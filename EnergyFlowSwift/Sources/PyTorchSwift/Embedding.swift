import Foundation
import EFCore

public struct Embedding {
    public let vocabSize: Int
    public let embeddingDim: Int
    public var weight: Tensor // [vocab, dim]

    public init(vocabSize: Int, embeddingDim: Int, seed: UInt64 = 42) {
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        // Normal init similar to Transformers embedding
        self.weight = Tensor.randomUniform([vocabSize, embeddingDim], min: -0.02, max: 0.02, seed: seed)
    }

    // ids: [B][L]
    public func forward(ids: [[Int]]) -> Tensor {
        let b = ids.count
        let l = ids.first?.count ?? 0
        precondition(ids.allSatisfy { $0.count == l }, "Embedding.forward: ragged input")
        var out = Tensor.zeros([b, l, embeddingDim])
        for bi in 0..<b {
            for li in 0..<l {
                let token = ids[bi][li]
                let idx = token >= 0 && token < vocabSize ? token : 0
                let wBase = idx * embeddingDim
                let oBase = (bi * l + li) * embeddingDim
                for d in 0..<embeddingDim {
                    out.data[oBase + d] = weight.data[wBase + d]
                }
            }
        }
        return out
    }
}
