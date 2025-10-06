import Foundation
import EFCore
import PyTorchSwift

public final class TextToCubeEncoder {
    public let energyConfig: EnergyConfig
    public let modelConfig: TextToCubeEncoderConfig
    public let surfaceDim: Int
    private var tokenizer: SimpleTokenizer

    private var embedding: Embedding

    // TCN encoder (roadmap)
    private var tcnEncoder: TCNEncoder

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
        self.tcnEncoder = TCNEncoder(numBlocks: modelConfig.tcnBlocks,
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
        // 2) TCN encoder
        let enc = tcnEncoder.forward(embs, mask: maskFixed)
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
        let enc = tcnEncoder.forward(embs, mask: maskFixed)
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

    // Training forward (text-mode): tokenize inside and return pooled/out
    public func forwardForTraining(texts: [String]) -> (pooled: Tensor, out: Tensor) {
        let batch = tokenizer.encodeBatch(texts, maxLength: modelConfig.maxLength)
        return forwardForTraining(inputIDs: batch.ids, attentionMask: batch.attentionMask)
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
