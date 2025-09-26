import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

// MARK: - Linear layer (GPU, forward only)
// y = x @ W^T + b
// x: [B, In], W: [Out, In], b: [Out] -> y: [B, Out]
public struct GraphLinear {
    public let inFeatures: Int
    public let outFeatures: Int
    public var weight: Tensor   // CPU-hosted weights; uploaded each call for now
    public var bias: Tensor?    // optional bias

    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, seed: UInt64 = 42) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = Tensor.randomUniform([outFeatures, inFeatures], min: -1.0/Float(inFeatures), max: 1.0/Float(inFeatures), seed: seed)
        self.bias = bias ? Tensor.randomUniform([outFeatures], min: -0.001, max: 0.001, seed: seed &+ 1) : nil
    }

    // Forward pass on GPU
    public func forward(_ x: Tensor) throws -> Tensor {
        let ctx = MPSGContext.shared
        precondition(x.shape.count == 2 && x.shape[1] == inFeatures, "GraphLinear.forward expects [B, inFeatures]")
        let b = x.shape[0]

        // Build a tiny graph: y = matmul(x, W^T) + b
        let g = MPSGraph()
        let xPlaceholder = g.placeholder(shape: [NSNumber(value: b), NSNumber(value: inFeatures)], dataType: .float32, name: "x")
        let wPlaceholder = g.placeholder(shape: [NSNumber(value: outFeatures), NSNumber(value: inFeatures)], dataType: .float32, name: "w")
        // Matmul expects [B, In] x [In, Out] -> we provide W^T by transposing placeholder logically
        let wT = g.transpose(wPlaceholder, permutation: [1, 0], name: "wT") // [In, Out]
        var y = g.matrixMultiplication(primary: xPlaceholder, secondary: wT, name: "xW") // [B, Out]
        var bPlaceholder: MPSGraphTensor? = nil
        if bias != nil {
            let bPHL = g.placeholder(shape: [NSNumber(value: outFeatures)], dataType: .float32, name: "b")
            // Add bias with broadcast across batch
            let bExp = g.expandDims(bPHL, axes: [0], name: "bExp") // [1, Out]
            y = g.addition(y, bExp, name: "addBias")
            bPlaceholder = bPHL
        }
        // Prepare feeds as MPSNDArray to avoid MPSGraphDevice intricacies
        // x tensor
        let xDesc = MPSNDArrayDescriptor(dataType: .float32, shape: [NSNumber(value: b), NSNumber(value: inFeatures)])
        let xArr = MPSNDArray(device: ctx.device, descriptor: xDesc)
        x.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress { xArr.writeBytes(UnsafeMutableRawPointer(mutating: base), strideBytes: nil) }
        }
        let xData = MPSGraphTensorData(xArr)
        // w tensor
        let wDesc = MPSNDArrayDescriptor(dataType: .float32, shape: [NSNumber(value: outFeatures), NSNumber(value: inFeatures)])
        let wArr = MPSNDArray(device: ctx.device, descriptor: wDesc)
        weight.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress { wArr.writeBytes(UnsafeMutableRawPointer(mutating: base), strideBytes: nil) }
        }
        let wData = MPSGraphTensorData(wArr)
        var feeds: [MPSGraphTensor: MPSGraphTensorData] = [
            xPlaceholder: xData,
            wPlaceholder: wData
        ]
        if let bPlaceholder, let bias = bias {
            let bDesc = MPSNDArrayDescriptor(dataType: .float32, shape: [NSNumber(value: outFeatures)])
            let bArr = MPSNDArray(device: ctx.device, descriptor: bDesc)
            bias.data.withUnsafeBytes { raw in
                if let base = raw.baseAddress { bArr.writeBytes(UnsafeMutableRawPointer(mutating: base), strideBytes: nil) }
            }
            let bData = MPSGraphTensorData(bArr)
            feeds[bPlaceholder] = bData
        }

        // Run synchronously on provided command queue
        let results = g.run(with: ctx.commandQueue, feeds: feeds, targetTensors: [y], targetOperations: nil)

        // Fetch result
        guard let out = results[y] else {
            throw MPSGError.commandBufferFailed
        }
        let outCount = b * outFeatures
        var outHost = [Float](repeating: 0, count: outCount)
        let ndarray = out.mpsndarray()
        withUnsafeMutableBytes(of: &outHost) { raw in
            if let base = raw.baseAddress { ndarray.readBytes(base, strideBytes: nil) }
        }
        return Tensor(shape: [b, outFeatures], data: outHost)
    }
}
