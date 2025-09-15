import Foundation
import EFCore

public enum Activations {
    // Gaussian Error Linear Unit (approximation used by many frameworks)
    public static func gelu(_ x: Tensor) -> Tensor {
        // 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
        let c: Float = 0.7978845608028654 // sqrt(2/pi)
        var out = x.data
        for i in 0..<out.count {
            let xi = out[i]
            let x3 = xi * xi * xi
            let t = c * (xi + 0.044715 * x3)
            out[i] = 0.5 * xi * (1 + Float(tanh(Double(t))))
        }
        return Tensor(shape: x.shape, data: out)
    }

    public static func tanh(_ x: Tensor) -> Tensor {
        var out = x.data
        for i in 0..<out.count {
            out[i] = Float(Darwin.tanh(Double(out[i])))
        }
        return Tensor(shape: x.shape, data: out)
    }
}
