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

    public func encode(_ texts: [String]) -> Tensor {
        let logger = Logger.shared
        logger.debug("encode(texts) start: batch=\(texts.count) maxLen=\(modelConfig.maxLength) outputDim=\(modelConfig.outputDim)", category: Logger.Category.textBridge)
        // 1) Tokenize
        let batch = tokenizer.encodeBatch(texts, maxLength: modelConfig.maxLength)
        return encodeTokens(inputIDs: batch.ids, attentionMask: batch.attentionMask)
    }

    // Public API: allow pre-tokenized inputs
    public func encodeTokens(inputIDs: [[Int]], attentionMask: [[Int]]) -> Tensor {
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
        let enc = tcnStack.forward(embs, mask: maskFixed)
        // 3) Masked-avg over sequence -> [B, hidden]
        let pooled = maskedMean(enc, mask: maskFixed)
        logger.debug("pooled: \(pooled.prettyShape) mean=\(mean(of: pooled)), std=\(std(of: pooled))", category: Logger.Category.textBridge)
        // 4) Projection to output via GPU
        var proj = gpuProj
        do {
            let outGPU = try proj.forward(pooled)
            self.gpuProj = proj
            var out = outGPU
            if modelConfig.useTanhOutput { out = Activations.tanh(out) }
            logger.debug("out: \(out.prettyShape) mean=\(mean(of: out)), std=\(std(of: out))", category: Logger.Category.textBridge)
            return out
        } catch {
            fatalError("GPU projection failed: \(error)")
        }
    }

    // Training forward: returns pooled and output before optional tanh
    public func forwardForTraining(inputIDs: [[Int]], attentionMask: [[Int]]) -> (pooled: Tensor, out: Tensor) {
        let (idsFixed, maskFixed) = padOrTruncate(inputIDs: inputIDs, attentionMask: attentionMask, to: modelConfig.maxLength)
        let embs = embedding.forward(ids: idsFixed)
        let enc = tcnStack.forward(embs, mask: maskFixed)
        let pooled = maskedMean(enc, mask: maskFixed)
        var proj = gpuProj
        do {
            let outGPU = try proj.forward(pooled)
            self.gpuProj = proj
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
    public func forwardForTrainingWithLastBlockCache(inputIDs: [[Int]], attentionMask: [[Int]]) -> (pooled: Tensor, out: Tensor, cache: LastTCNCache, maskFixed: [[Int]]) {
        let (idsFixed, maskFixed) = padOrTruncate(inputIDs: inputIDs, attentionMask: attentionMask, to: modelConfig.maxLength)
        let embs = embedding.forward(ids: idsFixed)
        let B = embs.shape[0]; let L = embs.shape[1]; let D = embs.shape[2]
        var x = embs
        // Run all but last block
        let nb = tcnStack.blocks.count
        precondition(nb > 0)
        if nb > 1 {
            for i in 0..<(nb - 1) {
                x = tcnStack.blocks[i].forward(x, mask: maskFixed)
            }
        }
        // Last block with caches
        let last = tcnStack.blocks[nb - 1]
        // LN on [B*L, D] using CPU LayerNorm (D is small; avoids MPSGraph)
        let xFlat = x.reshaped([B * L, D])
        let normFlat = last.ln.forward(xFlat)
        let norm = normFlat.reshaped([B, L, D])
        let h1 = last.conv1.forward(norm)
        let h1a = Activations.gelu(h1)
        var y = last.conv2.forward(h1a)
        // Residual
        for idx in 0..<(B*L*D) { y.data[idx] += x.data[idx] }
        // Zero masked positions
        for b in 0..<B { for t in 0..<L { if maskFixed[b][t] == 0 {
            let base = (b * L + t) * D
            for d in 0..<D { y.data[base + d] = 0 }
        } } }
        let pooled = maskedMean(y, mask: maskFixed)
        var proj = gpuProj
        do {
            let outGPU = try proj.forward(pooled)
            self.gpuProj = proj
            let cache = LastTCNCache(xIn: x, norm: norm, h1: h1, h1a: h1a)
            return (pooled, outGPU, cache, maskFixed)
        } catch {
            fatalError("GPU projection failed: \(error)")
        }
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
    public func forwardForTraining(texts: [String]) -> (pooled: Tensor, out: Tensor) {
        let batch = tokenizer.encodeBatch(texts, maxLength: modelConfig.maxLength)
        return forwardForTraining(inputIDs: batch.ids, attentionMask: batch.attentionMask)
    }

    // Training forward (text-mode) with last-block cache and fixed mask
    public func forwardForTrainingWithLastBlockCache(texts: [String]) -> (pooled: Tensor, out: Tensor, cache: LastTCNCache, maskFixed: [[Int]]) {
        let batch = tokenizer.encodeBatch(texts, maxLength: modelConfig.maxLength)
        return forwardForTrainingWithLastBlockCache(inputIDs: batch.ids, attentionMask: batch.attentionMask)
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
    public func projectionInputGradientsGPU(dY: Tensor) throws -> Tensor {
        return try gpuProj.inputGradientsGPU(dY: dY)
    }

    // Project-only using current GPU projection (to evaluate post-update metrics without recomputing TCN)
    public func projectOnly(_ pooled: Tensor) -> Tensor {
        var proj = gpuProj
        do {
            let outGPU = try proj.forward(pooled)
            self.gpuProj = proj
            return outGPU
        } catch {
            fatalError("GPU projection failed: \(error)")
        }
    }

    // Projection gradients via GPU matmul wrapper
    public func projectionGradientsGPU(X: Tensor, dY: Tensor) throws -> (Tensor, Tensor) {
        return try gpuProj.gradientsGPU(X: X, dY: dY)
    }

    // Backward for masked mean: given upstream dPooled [B,H] and mask [B][L],
    // distribute gradient equally across masked positions per example.
    public func maskedMeanBackward(dPooled: Tensor, mask: [[Int]], seqLen: Int) -> Tensor {
        let B = dPooled.shape[0]
        let H = dPooled.shape[1]
        precondition(mask.count == B)
        let L = seqLen
        var dEnc = Tensor.zeros([B, L, H])
        for b in 0..<B {
            precondition(mask[b].count == L)
            var denom: Float = 0
            for t in 0..<L { denom += Float(mask[b][t]) }
            denom = max(denom, 1e-9)
            for t in 0..<L {
                let m = Float(mask[b][t])
                if m == 0 { continue }
                let outBase = (b * L + t) * H
                let inBase = b * H
                for h in 0..<H { dEnc.data[outBase + h] = dPooled.data[inBase + h] * (m / denom) }
            }
        }
        return dEnc
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
