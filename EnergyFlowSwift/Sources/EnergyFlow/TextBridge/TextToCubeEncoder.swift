import Foundation
import EFCore
import PyTorchSwift

public final class TextToCubeEncoder {
    public let energyConfig: EnergyConfig
    public let modelConfig: TextToCubeEncoderConfig
    public let surfaceDim: Int
    private var tokenizer: SimpleTokenizer

    private var embedding: Embedding

    // Shared TCN stack (encoder/decoder trunk)
    private var tcnStack: TCNStack

    // GPU projection (FP16)
    private var gpuProj: GraphLinear

    public init(energyConfig: EnergyConfig = createDebugConfig(),
                modelConfig: TextToCubeEncoderConfig = TextToCubeEncoderConfig(),
                vocabSize: Int = 30000) {
        self.energyConfig = energyConfig
        self.modelConfig = modelConfig
        self.surfaceDim = energyConfig.surfaceDim
        self.tokenizer = SimpleTokenizer()
        // Seeds per module
        let seedEmbed = Seed.derive(modelConfig.baseSeed, label: "embedding")
        self.embedding = Embedding(vocabSize: vocabSize, embeddingDim: modelConfig.hiddenDim, seed: seedEmbed)
        self.tcnStack = TCNStack(numBlocks: modelConfig.tcnBlocks,
                                  dim: modelConfig.hiddenDim,
                                  hidden: modelConfig.ffDim,
                                  kernelSize: modelConfig.kernelSize,
                                  dilationSchedule: modelConfig.dilationSchedule,
                                  seed: Seed.derive(modelConfig.baseSeed, label: "tcn"))
        self.gpuProj = GraphLinear(inFeatures: modelConfig.hiddenDim, outFeatures: modelConfig.outputDim, bias: true, seed: Seed.derive(modelConfig.baseSeed, label: "gpu_proj"))
    }

    public func encode(_ texts: [String],
                       on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        let logger = Logger.shared
        logger.debug("encode(texts) start: batch=\(texts.count) maxLen=\(modelConfig.maxLength) outputDim=\(modelConfig.outputDim)", category: Logger.Category.textBridge)
        // 1) Tokenize
        let batch = tokenizer.encodeBatch(texts, maxLength: modelConfig.maxLength)
        return try await encodeTokens(inputIDs: batch.ids,
                                      attentionMask: batch.attentionMask,
                                      on: gpu)
    }

    // Public API: allow pre-tokenized inputs
    public func encodeTokens(inputIDs: [[Int]],
                             attentionMask: [[Int]],
                             on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        let logger = Logger.shared
        let b = inputIDs.count
        let l = inputIDs.first?.count ?? 0
        logger.debug("encode(tokens) start: batch=\(b) L=\(l) maxLen=\(modelConfig.maxLength) outputDim=\(modelConfig.outputDim)", category: Logger.Category.textBridge)
        // 0) Pad/truncate to maxLength for fixed shapes
        let (idsFixed, maskFixed) = padOrTruncate(inputIDs: inputIDs, attentionMask: attentionMask, to: modelConfig.maxLength)
        logger.debug("token ids shape: [\(idsFixed.count), \(idsFixed.first?.count ?? 0)] mask shape: [\(maskFixed.count), \(maskFixed.first?.count ?? 0)]", category: Logger.Category.textBridge)
        // 1) Embedding [B,L,hidden]
        let embs = embedding.forward(ids: idsFixed)
        logger.debug("embeddings: \(embs.prettyShape)", category: Logger.Category.textBridge)
        // 2) TCN encoder (shared stack)
        let enc = try await tcnStack.forward(embs, mask: maskFixed, on: gpu)
        // 3) Masked-avg over sequence -> [B, hidden]
        let pooled = try await gpu.maskedMean(x: enc, mask: maskFixed)
        logger.debug("pooled: \(pooled.prettyShape) mean=\(mean(of: pooled)), std=\(std(of: pooled))", category: Logger.Category.textBridge)
        // 4) Projection to output via GPU
        var proj = gpuProj
        do {
            let readback = try await proj.forwardDeferred(pooled, on: gpu)
            gpuProj = proj
            let outGPU = try await readback.value()
            let out = modelConfig.useTanhOutput ? Activations.tanh(outGPU) : outGPU
            logger.debug("out: \(out.prettyShape) mean=\(mean(of: out)), std=\(std(of: out))", category: Logger.Category.textBridge)
            return out
        } catch {
            fatalError("GPU projection failed: \(error)")
        }
    }

    // Training forward: returns pooled and output before optional tanh
    public func forwardForTraining(inputIDs: [[Int]],
                                   attentionMask: [[Int]],
                                   on gpu: GPUActor = GPU.shared) async throws -> (pooled: Tensor, out: Tensor) {
        let (idsFixed, maskFixed) = padOrTruncate(inputIDs: inputIDs, attentionMask: attentionMask, to: modelConfig.maxLength)
        let embs = embedding.forward(ids: idsFixed)
        let enc = try await tcnStack.forward(embs, mask: maskFixed, on: gpu)
        let pooled = try await gpu.maskedMean(x: enc, mask: maskFixed)
        var proj = gpuProj
        do {
            let readback = try await proj.forwardDeferred(pooled, on: gpu)
            gpuProj = proj
            let outGPU = try await readback.value()
            return (pooled, outGPU)
        } catch {
            fatalError("GPU projection failed: \(error)")
        }
    }

    public struct LastTCNCache {
        public let xIn: Tensor   // [B,L,D]
        public let norm: Tensor  // [B,L,D]
        public let h1: Tensor    // [B,L,H]
        public let h1a: Tensor   // [B,L,H]
    }

    // Forward for training with cache for the last TCN block
    public func forwardForTrainingWithLastBlockCache(inputIDs: [[Int]],
                                                     attentionMask: [[Int]],
                                                     on gpu: GPUActor = GPU.shared) async throws -> (pooled: Tensor, out: Tensor, cache: LastTCNCache, maskFixed: [[Int]]) {
        let (idsFixed, maskFixed) = padOrTruncate(inputIDs: inputIDs, attentionMask: attentionMask, to: modelConfig.maxLength)
        let embs = embedding.forward(ids: idsFixed)
        let B = embs.shape[0]; let L = embs.shape[1]; let D = embs.shape[2]
        var x = embs
        // Run all but last block
        let nb = tcnStack.blocks.count
        precondition(nb > 0)
        if nb > 1 {
            for i in 0..<(nb - 1) {
                x = try await tcnStack.blocks[i].forward(x, mask: maskFixed, on: gpu)
            }
        }
        // Last block with caches
        let last = tcnStack.blocks[nb - 1]
        // LN on [B*L, D] using GPU LayerNorm via Metal
        let xFlat = x.reshaped([B * L, D])
        let normFlat = try await gpu.layerNormForward(x: xFlat,
                                                      gamma: last.ln.gamma,
                                                      beta: last.ln.beta,
                                                      eps: last.ln.eps)
        let norm = normFlat.reshaped([B, L, D])
        let h1 = try await last.conv1.forwardAsync(norm, on: gpu)
        let h1a = try await gpu.geluForward(x: h1)
        var y = try await last.conv2.forwardAsync(h1a, on: gpu)
        // Residual and mask on GPU
        y = try await gpu.residualAdd(y: y, x: x)
        y = try await gpu.maskZero(y: y, mask: maskFixed)
        let pooled = try await gpu.maskedMean(x: y, mask: maskFixed)
        var proj = gpuProj
        do {
            let readback = try await proj.forwardDeferred(pooled, on: gpu)
            gpuProj = proj
            let outGPU = try await readback.value()
            let cache = LastTCNCache(xIn: x, norm: norm, h1: h1, h1a: h1a)
            return (pooled, outGPU, cache, maskFixed)
        } catch {
            fatalError("GPU projection failed: \(error)")
        }
    }

    // Deferred variant: returns readbacks for pooled and out; cache and mask remain as-is
    public func forwardForTrainingWithLastBlockCacheDeferred(inputIDs: [[Int]],
                                                             attentionMask: [[Int]],
                                                             on gpu: GPUActor = GPU.shared) async throws -> (pooledHandleRB: GPUReadback<GPUTensorHandle>, outRB: GPUReadback<Tensor>, cache: LastTCNCache, maskFixed: [[Int]]) {
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: start B=\(inputIDs.count) L=\(inputIDs.first?.count ?? 0)", category: Logger.Category.training)
        let (idsFixed, maskFixed) = padOrTruncate(inputIDs: inputIDs, attentionMask: attentionMask, to: modelConfig.maxLength)
        let embs = embedding.forward(ids: idsFixed)
        let B = embs.shape[0]; let L = embs.shape[1]; let D = embs.shape[2]
        var x = embs
        // Run all but last block
        let nb = tcnStack.blocks.count
        precondition(nb > 0)
        if nb > 1 {
            for i in 0..<(nb - 1) {
                x = try await tcnStack.blocks[i].forward(x, mask: maskFixed, on: gpu)
            }
        }
        // Last block with caches
        let last = tcnStack.blocks[nb - 1]
        let xHandle = await gpu.tensorToHandle(x, label: "Encoder.lastBlock.input")
        let normRB = try await gpu.layerNormForwardHandleDeferred(xHandle: xHandle,
                                                                  gamma: last.ln.gamma,
                                                                  beta: last.ln.beta,
                                                                  eps: last.ln.eps,
                                                                  outputShape: [B, L, D],
                                                                  consumeInput: false)
        let normHandle = try await normRB.value()
        let norm = try await gpu.readHandleToTensor(normHandle)
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: LN done", category: Logger.Category.training)

        let h1RB = try await last.conv1.forwardFromHandleDeferred(normHandle,
                                                                  on: gpu,
                                                                  outputShape: [B, L, last.hidden],
                                                                  consumeInput: false)
        let h1Handle = try await h1RB.value()
        let h1 = try await gpu.readHandleToTensor(h1Handle)

        let h1aRB = try await gpu.geluForwardHandleDeferred(xHandle: h1Handle,
                                                            outputShape: [B, L, last.hidden],
                                                            consumeInput: false)
        let h1aHandle = try await h1aRB.value()
        let h1a = try await gpu.readHandleToTensor(h1aHandle)
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: GELU done", category: Logger.Category.training)

        let conv2RB = try await last.conv2.forwardFromHandleDeferred(h1aHandle,
                                                                     on: gpu,
                                                                     outputShape: [B, L, D],
                                                                     consumeInput: false)
        let conv2Handle = try await conv2RB.value()
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: conv2 done", category: Logger.Category.training)

        await gpu.releaseHandle(normHandle)
        await gpu.releaseHandle(h1Handle)
        await gpu.releaseHandle(h1aHandle)

        let residualRB = try await gpu.residualAddHandleDeferred(yHandle: conv2Handle,
                                                                 xHandle: xHandle,
                                                                 outputShape: [B, L, D],
                                                                 consumeInputs: true)
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: scheduled residualAdd deferred", category: Logger.Category.training)
        let residualHandle = try await residualRB.value()
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: residualAdd readback resolved", category: Logger.Category.training)

        let maskRB = try await gpu.maskZeroHandleDeferred(yHandle: residualHandle,
                                                          mask: maskFixed,
                                                          outputShape: [B, L, D],
                                                          consumeInput: true)
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: scheduled maskZero deferred", category: Logger.Category.training)
        let maskedHandle = try await maskRB.value()
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: maskZero readback resolved", category: Logger.Category.training)
        let y = try await gpu.readHandleToTensor(maskedHandle)
        await gpu.releaseHandle(maskedHandle)
        // Compute pooled in two forms:
        // 1) GPU handle for chaining into projection (no host copy)
        // 2) Host Tensor for trainers that need pooled (return as resolved readback)
        let pooledHandleRB = try await gpu.maskedMeanHandleDeferred(x: y, mask: maskFixed)
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: scheduled maskedMean handle deferred", category: Logger.Category.training)
        var proj = gpuProj
        // Resolve handle (no batch active, so this will await command completion only for maskedMean kernel)
        let pooledHandle = try await pooledHandleRB.value()
        let outRB = try await proj.forwardFromHandleDeferred(pooledHandle, on: gpu)
        gpuProj = proj
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: scheduled proj from handle deferred", category: Logger.Category.training)
        let cache = LastTCNCache(xIn: x, norm: norm, h1: h1, h1a: h1a)
        Logger.shared.info1("Encoder.forwardForTrainingWithLastBlockCacheDeferred: return", category: Logger.Category.training)
        return (pooledHandleRB, outRB, cache, maskFixed)
    }

    // Accessors for last TCN block params
    public func getLastBlockParams() -> (w1: Tensor, b1: Tensor?, w2: Tensor, b2: Tensor?, gamma: Tensor, beta: Tensor) {
        let nb = tcnStack.blocks.count
        precondition(nb > 0)
        let last = tcnStack.blocks[nb - 1]
        return (last.conv1.weight, last.conv1.bias ?? Tensor.zeros([0]), last.conv2.weight, last.conv2.bias ?? Tensor.zeros([0]), last.ln.gamma, last.ln.beta)
    }
    public func setLastBlockParams(w1: Tensor, b1: Tensor?, w2: Tensor, b2: Tensor?, gamma: Tensor, beta: Tensor) {
        let nb = tcnStack.blocks.count
        precondition(nb > 0)
        tcnStack.blocks[nb - 1].conv1.weight = w1
        tcnStack.blocks[nb - 1].conv1.bias = b1
        tcnStack.blocks[nb - 1].conv2.weight = w2
        tcnStack.blocks[nb - 1].conv2.bias = b2
        tcnStack.blocks[nb - 1].ln.gamma = gamma
        tcnStack.blocks[nb - 1].ln.beta = beta
    }
    public func invalidateLastBlockCaches() {
        let nb = tcnStack.blocks.count
        precondition(nb > 0)
        tcnStack.blocks[nb - 1].conv1.invalidateCache()
        tcnStack.blocks[nb - 1].conv2.invalidateCache()
    }

    // Training forward (text-mode): tokenize inside and return pooled/out
    public func forwardForTraining(texts: [String],
                                   on gpu: GPUActor = GPU.shared) async throws -> (pooled: Tensor, out: Tensor) {
        let batch = tokenizer.encodeBatch(texts, maxLength: modelConfig.maxLength)
        return try await forwardForTraining(inputIDs: batch.ids,
                                            attentionMask: batch.attentionMask,
                                            on: gpu)
    }

    // Training forward (text-mode) with last-block cache and fixed mask
    public func forwardForTrainingWithLastBlockCache(texts: [String],
                                                     on gpu: GPUActor = GPU.shared) async throws -> (pooled: Tensor, out: Tensor, cache: LastTCNCache, maskFixed: [[Int]]) {
        let batch = tokenizer.encodeBatch(texts, maxLength: modelConfig.maxLength)
        return try await forwardForTrainingWithLastBlockCache(inputIDs: batch.ids,
                                                              attentionMask: batch.attentionMask,
                                                              on: gpu)
    }

    // Projection param accessors for training
    public func getProjParams() -> (weight: Tensor, bias: Tensor?) {
        return (gpuProj.weight, gpuProj.bias)
    }
    public func setProjParams(weight: Tensor, bias: Tensor?) {
        gpuProj.weight = weight
        gpuProj.bias = bias
    }
    public func invalidateProjectionCache() {
        gpuProj.invalidateCache()
    }
    // Input gradients of projection: dX = dY @ W
    public func projectionInputGradientsGPU(dY: Tensor,
                                            on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        var proj = gpuProj
        let readback = try await proj.inputGradientsGPUDeferred(dY: dY, on: gpu)
        gpuProj = proj
        return try await readback.value()
    }

    public func projectionInputGradientsGPUDeferred(dY: Tensor,
                                                     on gpu: GPUActor = GPU.shared) async throws -> GPUReadback<Tensor> {
        var proj = gpuProj
        let readback = try await proj.inputGradientsGPUDeferred(dY: dY, on: gpu)
        gpuProj = proj
        return readback
    }

    // Project-only using current GPU projection (to evaluate post-update metrics without recomputing TCN)
    public func projectOnly(_ pooled: Tensor,
                            on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        var proj = gpuProj
        do {
            let readback = try await proj.forwardDeferred(pooled, on: gpu)
            gpuProj = proj
            return try await readback.value()
        } catch {
            fatalError("GPU projection failed: \(error)")
        }
    }

    // Projection gradients via GPU matmul wrapper
    public func projectionGradientsGPU(X: Tensor,
                                       dY: Tensor,
                                       on gpu: GPUActor = GPU.shared) async throws -> (Tensor, Tensor) {
        var proj = gpuProj
        let readback = try await proj.gradientsGPUDeferred(X: X, dY: dY, on: gpu)
        gpuProj = proj
        return try await readback.value()
    }

    public func projectionGradientsGPUDeferred(X: Tensor,
                                               dY: Tensor,
                                               on gpu: GPUActor = GPU.shared) async throws -> GPUReadback<(Tensor, Tensor)> {
        var proj = gpuProj
        let readback = try await proj.gradientsGPUDeferred(X: X, dY: dY, on: gpu)
        gpuProj = proj
        return readback
    }

    // Backward for masked mean: given upstream dPooled [B,H] and mask [B][L],
    // distribute gradient equally across masked positions per example.
    public func maskedMeanBackward(dPooled: Tensor,
                                   mask: [[Int]],
                                   seqLen: Int,
                                   on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        return try await gpu.maskedMeanBackward(dPooled: dPooled, mask: mask, seqLen: seqLen)
    }

    // Simple stats for debugging
    private func mean(of t: Tensor) -> Float {
        var s: Float = 0
        for v in t.data { s += v }
        return s / Float(max(t.count, 1))
    }
    private func std(of t: Tensor) -> Float {
        let m = mean(of: t)
        var acc: Float = 0
        for v in t.data { let d = v - m; acc += d*d }
        return sqrt(acc / Float(max(t.count, 1)))
    }

    // MARK: - Helpers

    private func padOrTruncate(inputIDs: [[Int]], attentionMask: [[Int]], to maxLen: Int) -> ([[Int]], [[Int]]) {
        precondition(inputIDs.count == attentionMask.count)
        var idsOut: [[Int]] = []
        var maskOut: [[Int]] = []
        idsOut.reserveCapacity(inputIDs.count)
        maskOut.reserveCapacity(attentionMask.count)
        for i in 0..<inputIDs.count {
            var ids = inputIDs[i]
            var m = attentionMask[i]
            if ids.count > maxLen {
                ids = Array(ids.prefix(maxLen))
                m = Array(m.prefix(maxLen))
            } else if ids.count < maxLen {
                let pad = maxLen - ids.count
                ids.append(contentsOf: Array(repeating: 0, count: pad))
                m.append(contentsOf: Array(repeating: 0, count: pad))
            }
            idsOut.append(ids)
            maskOut.append(m)
        }
        return (idsOut, maskOut)
    }


    private func maskedMean(_ x: Tensor, mask: [[Int]]) -> Tensor {
        // x: [B,L,H], mask: [B][L] -> [B,H]
        let b = x.shape[0], l = x.shape[1], h = x.shape[2]
        precondition(mask.count == b && mask.allSatisfy { $0.count == l })
        var out = Tensor.zeros([b, h])
        for bi in 0..<b {
            var denom: Float = 0
            for li in 0..<l { denom += Float(mask[bi][li]) }
            denom = max(denom, 1e-9)
            for li in 0..<l {
                let m = Float(mask[bi][li])
                if m == 0 { continue }
                let xBase = (bi * l + li) * h
                for di in 0..<h {
                    out.data[bi * h + di] += x.data[xBase + di] * m
                }
            }
            for di in 0..<h { out.data[bi * h + di] /= denom }
        }
        return out
    }

}
