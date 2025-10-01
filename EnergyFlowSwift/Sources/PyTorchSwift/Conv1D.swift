import Foundation
import EFCore

// Simple 1D causal convolution with dilation (CPU reference)
// Input x: [B, L, Cin]
// Weight: [Cout, Cin, K]
// Bias: [Cout]
public struct Conv1D {
    public let inChannels: Int
    public let outChannels: Int
    public let kernelSize: Int
    public let dilation: Int
    public var weight: Tensor  // [Cout, Cin, K]
    public var bias: Tensor?   // [Cout]

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int = 1, bias: Bool = true, seed: UInt64 = 42) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = max(1, dilation)
        let wCount = outChannels * inChannels * kernelSize
        self.weight = Tensor.randomUniform([outChannels, inChannels, kernelSize], min: -1.0/Float(inChannels*kernelSize), max: 1.0/Float(inChannels*kernelSize), seed: seed)
        self.bias = bias ? Tensor.randomUniform([outChannels], min: -0.001, max: 0.001, seed: seed &+ 1) : nil
    }

    // Forward causal conv: y[b,t,o] = sum_{i,k} x[b, t - k*d, i] * w[o, i, k] + b[o]
    // If (t - k*d) < 0, treat x as 0 (implicit left padding).
    public func forward(_ x: Tensor) -> Tensor {
        precondition(x.shape.count == 3)
        let B = x.shape[0], L = x.shape[1]
        let Cin = x.shape[2]
        precondition(Cin == inChannels)
        var y = Tensor.zeros([B, L, outChannels])
        for b in 0..<B {
            for t in 0..<L {
                for o in 0..<outChannels {
                    var acc: Float = 0
                    for i in 0..<inChannels {
                        for k in 0..<kernelSize {
                            let ti = t - k * dilation
                            if ti < 0 { continue }
                            let xIdx = (b * L + ti) * inChannels + i
                            let wIdx = (o * inChannels + i) * kernelSize + k
                            acc += x.data[xIdx] * weight.data[wIdx]
                        }
                    }
                    if let bT = bias { acc += bT.data[o] }
                    y.data[(b * L + t) * outChannels + o] = acc
                }
            }
        }
        return y
    }
}