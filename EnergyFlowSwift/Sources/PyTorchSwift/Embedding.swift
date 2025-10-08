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
        // GPU gather for performance
        return EmbeddingGPU.forward(ids: ids, weight: weight)
    }
}
