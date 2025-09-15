import Foundation
import EFCore
import PyTorchSwift

public final class TextToCubeEncoder {
    public let config: EnergyConfig
    public let surfaceDim: Int
    private var tokenizer: SimpleTokenizer

    private let hiddenDim: Int = 256
    private let maxPos: Int = 512

    private var embedding: Embedding
    private var proj1: Linear
    private var ln: LayerNorm
    private var proj2: Linear

    // Sinusoidal positional encoding [maxPos, hiddenDim]
    private var positionalEncoding: Tensor

    public init(config: EnergyConfig = createDebugConfig(), vocabSize: Int = 30000) {
        self.config = config
        self.surfaceDim = config.surfaceDim
        self.tokenizer = SimpleTokenizer()
        self.embedding = Embedding(vocabSize: vocabSize, embeddingDim: hiddenDim)
        self.proj1 = Linear(inFeatures: hiddenDim, outFeatures: hiddenDim)
        self.ln = LayerNorm(dim: hiddenDim)
        self.proj2 = Linear(inFeatures: hiddenDim, outFeatures: surfaceDim)
        self.positionalEncoding = TextToCubeEncoder.makePositionalEncoding(maxLen: maxPos, dim: hiddenDim)
    }

    public func encode(_ texts: [String], maxLength: Int = 128) -> Tensor {
        // 1) Tokenize
        var tok = tokenizer
        let batch = tok.encodeBatch(texts, maxLength: maxLength)
        // 2) Embedding [B,L,hidden]
        var embs = embedding.forward(ids: batch.ids)
        // 3) Add positional encoding (truncate to seqLen)
        let seqLen = maxLength
        precondition(seqLen <= maxPos, "seqLen exceeds maxPos in positional encoding")
        let pe = positionalSlice(len: seqLen) // [L, hidden]
        // add PE to embeddings
        addPE(to: &embs, pe: pe)
        // 4) (Placeholder) Transformer encoder — identity for Phase 1
        let enc = embs
        // 5) Masked-avg over sequence -> [B, hidden]
        let pooled = maskedMean(enc, mask: batch.attentionMask)
        // 6) Projection MLP: 256 -> 256 -> LN -> surfaceDim -> tanh
        var out = proj1.forward(pooled)
        out = Activations.gelu(out)
        out = ln.forward(out)
        out = proj2.forward(out)
        out = Activations.tanh(out)
        // shape: [B, surfaceDim]
        return out
    }

    // MARK: - Helpers

    private func positionalSlice(len: Int) -> Tensor { // [L, hidden]
        precondition(len <= maxPos)
        let hidden = hiddenDim
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

    private static func makePositionalEncoding(maxLen: Int, dim: Int) -> Tensor {
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
