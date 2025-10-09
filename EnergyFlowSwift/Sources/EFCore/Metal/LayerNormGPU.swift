import Foundation

public enum LayerNormGPU {
    public static func forward(x: Tensor, gamma: Tensor, beta: Tensor, eps: Float) -> Tensor {
        do {
            return try GPU.blocking(label: "LayerNormGPU.forward") { actor in
                try await actor.layerNormForward(x: x, gamma: gamma, beta: beta, eps: eps)
            }
        } catch {
            fatalError("LayerNormGPU.forward failed: \(error)")
        }
    }

    public static func backward(x: Tensor, g: Tensor, gamma: Tensor, eps: Float) -> (Tensor, Tensor, Tensor) {
        do {
            return try GPU.blocking(label: "LayerNormGPU.backward") { actor in
                try await actor.layerNormBackward(x: x, g: g, gamma: gamma, eps: eps)
            }
        } catch {
            fatalError("LayerNormGPU.backward failed: \(error)")
        }
    }
}
