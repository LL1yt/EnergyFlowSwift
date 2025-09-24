import Foundation
import EFCore
import PyTorchSwift

public final class TextToCubeEncoder {
    public let energyConfig: EnergyConfig
    public let modelConfig: TextToCubeEncoderConfig
    public let surfaceDim: Int
    private var tokenizer: SimpleTokenizer

    private var embedding: Embedding
    private var proj1: Linear
    private var ln: LayerNorm
    private var proj2: Linear

    // Transformer encoder (CPU reference, starts with single-head)
    private var encoder: TransformerEncoder

    // Sinusoidal positional encoding [maxPosition, hiddenDim]
    private var positionalEncoding: Tensor

    public init(energyConfig: EnergyConfig = createDebugConfig(),
                modelConfig: TextToCubeEncoderConfig = TextToCubeEncoderConfig(),
                vocabSize: Int = 30000) {
        self.energyConfig = energyConfig
        self.modelConfig = modelConfig
        self.surfaceDim = energyConfig.surfaceDim
        self.tokenizer = SimpleTokenizer()
        // Seeds per module
        let seedEmbed = Seed.derive(modelConfig.baseSeed, label: "embedding")
        let seedLin1  = Seed.derive(modelConfig.baseSeed, label: "linear1")
        let seedLin2  = Seed.derive(modelConfig.baseSeed, label: "linear2")
        self.embedding = Embedding(vocabSize: vocabSize, embeddingDim: modelConfig.hiddenDim, seed: seedEmbed)
        self.proj1 = Linear(inFeatures: modelConfig.hiddenDim, outFeatures: modelConfig.hiddenDim, seed: seedLin1)
        self.ln = LayerNorm(dim: modelConfig.hiddenDim)
        self.proj2 = Linear(inFeatures: modelConfig.hiddenDim, outFeatures: modelConfig.outputDim, seed: seedLin2)
        self.encoder = TransformerEncoder(numLayers: modelConfig.numLayers,
                                          hidden: modelConfig.hiddenDim,
                                          ffDim: modelConfig.ffDim,
                                          numHeads: modelConfig.numHeads,
                                          seed: Seed.derive(modelConfig.baseSeed, label: "encoder"))
        self.positionalEncoding = TextToCubeEncoder.makePositionalEncoding(maxLen: modelConfig.maxPosition, dim: modelConfig.hiddenDim, seed: Seed.derive(modelConfig.baseSeed, label: "posenc"))
    }

    public func encode(_ texts: [String]) -> Tensor {
        let logger = Logger.shared
        logger.debug("encode(texts) start: batch=\(texts.count) maxLen=\(modelConfig.maxLength) outputDim=\(modelConfig.outputDim)", category: Logger.Category.textBridge)
        // 1) Tokenize
        var tok = tokenizer
        let batch = tok.encodeBatch(texts, maxLength: modelConfig.maxLength)
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
        var embs = embedding.forward(ids: idsFixed)
        logger.debug("embeddings: \(embs.prettyShape)", category: Logger.Category.textBridge)
        // 2) Add positional encoding (truncate to seqLen)
        let seqLen = modelConfig.maxLength
        precondition(seqLen <= modelConfig.maxPosition, "seqLen exceeds maxPosition in positional encoding")
        let pe = positionalSlice(len: seqLen) // [L, hidden]
        addPE(to: &embs, pe: pe)
        logger.debug("embeddings+PE: \(embs.prettyShape)", category: Logger.Category.textBridge)
        // 3) Transformer encoder
        let enc = encoder.forward(embs, mask: maskFixed)
        // 4) Masked-avg over sequence -> [B, hidden]
        let pooled = maskedMean(enc, mask: maskFixed)
        logger.debug("pooled: \(pooled.prettyShape) mean=\(mean(of: pooled)), std=\(std(of: pooled))", category: Logger.Category.textBridge)
        // 5) Projection MLP
        var out = proj1.forward(pooled)
        out = Activations.gelu(out)
        out = ln.forward(out)
        out = proj2.forward(out)
        if modelConfig.useTanhOutput { out = Activations.tanh(out) }
        logger.debug("out: \(out.prettyShape) mean=\(mean(of: out)), std=\(std(of: out))", category: Logger.Category.textBridge)
        return out
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

    private func positionalSlice(len: Int) -> Tensor { // [L, hidden]
        precondition(len <= modelConfig.maxPosition)
        let hidden = modelConfig.hiddenDim
        var out = Tensor.zeros([len, hidden])
        for l in 0..<len {
            let srcBase = l * hidden
            for d in 0..<hidden { out.data[srcBase + d] = positionalEncoding.data[srcBase + d] }
        }
        return out
    }

    private func addPE(to embs: inout Tensor, pe: Tensor) {
        // embs: [B,L,H]; pe: [L,H]
        let b = embs.shape[0]
        let l = embs.shape[1]
        let h = embs.shape[2]
        precondition(pe.shape == [l, h])
        for bi in 0..<b {
            for li in 0..<l {
                let eBase = (bi * l + li) * h
                let pBase = li * h
                for di in 0..<h {
                    embs.data[eBase + di] += pe.data[pBase + di]
                }
            }
        }
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

    private static func makePositionalEncoding(maxLen: Int, dim: Int, seed: UInt64) -> Tensor {
        // seed is not strictly needed for sinusoidal, but we keep signature uniform for future variants
        _ = seed
        var pe = Tensor.zeros([maxLen, dim])
        let divTermBase = Float(log(10000.0) / Double(dim))
        for pos in 0..<maxLen {
            for i in stride(from: 0, to: dim, by: 2) {
                let div = expf(-Float(i) * divTermBase)
                pe.data[pos * dim + i] = sinf(Float(pos) * div)
                if i + 1 < dim {
                    pe.data[pos * dim + i + 1] = cosf(Float(pos) * div)
                }
            }
        }
        return pe
    }
}
