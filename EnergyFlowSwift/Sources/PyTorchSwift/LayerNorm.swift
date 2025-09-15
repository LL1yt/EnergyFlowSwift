import Foundation
import EFCore

public struct LayerNorm {
    public let dim: Int
    public var gamma: Tensor // [dim]
    public var beta: Tensor  // [dim]
    public let eps: Float

    public init(dim: Int, eps: Float = 1e-5) {
        self.dim = dim
        self.gamma = Tensor(shape: [dim], data: Array(repeating: 1, count: dim))
        self.beta = Tensor(zeros: [dim])
        self.eps = eps
    }

    // x: [B, dim] -> y: [B, dim]
    public func forward(_ x: Tensor) -> Tensor {
        precondition(x.shape.count == 2 && x.shape[1] == dim, "LayerNorm.forward expects [B, dim]")
        let b = x.shape[0]
        var out = Tensor.zeros([b, dim])
        for i in 0..<b {
            let base = i * dim
            // mean
            var mean: Float = 0
            for j in 0..<dim { mean += x.data[base + j] }
            mean /= Float(dim)
            // variance
            var varAcc: Float = 0
            for j in 0..<dim { let d = x.data[base + j] - mean; varAcc += d*d }
            let varVal = varAcc / Float(dim)
            let invStd = 1.0 / Float(sqrt(Double(varVal + eps)))
            for j in 0..<dim {
                let norm = (x.data[base + j] - mean) * invStd
                out.data[base + j] = norm * gamma.data[j] + beta.data[j]
            }
        }
        return out
    }
}
