import Foundation
import EFCore

public struct Linear {
    public let inFeatures: Int
    public let outFeatures: Int
    public var weight: Tensor // [out, in]
    public var bias: Tensor?  // [out]

    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, seed: UInt64 = 42) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = Tensor.randomUniform([outFeatures, inFeatures], min: -1.0/Float(inFeatures), max: 1.0/Float(inFeatures), seed: seed)
        self.bias = bias ? Tensor.randomUniform([outFeatures], min: -0.001, max: 0.001, seed: seed &+ 1) : nil
    }

    // x: [B, in] -> y: [B, out]
    public func forward(_ x: Tensor) -> Tensor {
        precondition(x.shape.count == 2 && x.shape[1] == inFeatures, "Linear.forward: input shape must be [B, inFeatures]")
        let batch = x.shape[0]
        var y = Tensor.zeros([batch, outFeatures])
        for b in 0..<batch {
            for o in 0..<outFeatures {
                var sum: Float = 0
                let wRowBase = o * inFeatures
                let xRowBase = b * inFeatures
                for i in 0..<inFeatures {
                    sum += x.data[xRowBase + i] * weight.data[wRowBase + i]
                }
                if let bTensor = bias {
                    sum += bTensor.data[o]
                }
                y.data[b * outFeatures + o] = sum
            }
        }
        return y
    }
}
