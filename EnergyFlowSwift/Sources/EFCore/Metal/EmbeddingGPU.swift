import Foundation

public enum EmbeddingGPU {
    public static func forward(ids: [[Int]], weight: Tensor) -> Tensor {
        do {
            return try GPU.blocking(label: "EmbeddingGPU.forward") { actor in
                try await actor.embeddingForward(ids: ids, weight: weight)
            }
        } catch {
            fatalError("EmbeddingGPU.forward failed: \(error)")
        }
    }
}
