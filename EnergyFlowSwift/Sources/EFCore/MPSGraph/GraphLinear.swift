import Foundation
import Metal
import MetalPerformanceShaders

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
        let logger = Logger.shared
        let ctx = MPSGContext.shared
        precondition(x.shape.count == 2 && x.shape[1] == inFeatures, "GraphLinear.forward expects [B, inFeatures]")
        let b = x.shape[0]
        logger.info("GraphLinear.forward start B=\(b) In=\(inFeatures) Out=\(outFeatures)", category: Logger.Category.textBridge)

        // GPU matmul via MPSMatrixMultiplication: y = x @ W^T
        let elemSize = MemoryLayout<Float>.size
        let xCount = b * inFeatures
        let wCount = outFeatures * inFeatures
        let yCount = b * outFeatures

        guard let xBuf = ctx.device.makeBuffer(length: xCount * elemSize, options: .storageModeShared),
              let wBuf = ctx.device.makeBuffer(length: wCount * elemSize, options: .storageModeShared),
              let yBuf = ctx.device.makeBuffer(length: yCount * elemSize, options: .storageModeShared)
        else {
            throw MPSGError.commandBufferFailed
        }
        // Copy host data into shared buffers
        x.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(xBuf.contents(), base, xCount * elemSize) } }
        weight.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(wBuf.contents(), base, wCount * elemSize) } }
        memset(yBuf.contents(), 0, yCount * elemSize)

        // Descriptors (row-major): rowBytes = columns * sizeof(Float)
        let xDesc = MPSMatrixDescriptor(rows: b, columns: inFeatures, rowBytes: inFeatures * elemSize, dataType: .float32)
        let wDesc = MPSMatrixDescriptor(rows: outFeatures, columns: inFeatures, rowBytes: inFeatures * elemSize, dataType: .float32)
        let yDesc = MPSMatrixDescriptor(rows: b, columns: outFeatures, rowBytes: outFeatures * elemSize, dataType: .float32)

        let xMat = MPSMatrix(buffer: xBuf, descriptor: xDesc)
        let wMat = MPSMatrix(buffer: wBuf, descriptor: wDesc)
        let yMat = MPSMatrix(buffer: yBuf, descriptor: yDesc)

        let mm = MPSMatrixMultiplication(device: ctx.device,
                                         transposeLeft: false,
                                         transposeRight: true,
                                         resultRows: b,
                                         resultColumns: outFeatures,
                                         interiorColumns: inFeatures,
                                         alpha: 1.0,
                                         beta: 0.0)
        logger.info("GraphLinear.forward run() begin", category: Logger.Category.textBridge)
        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer() else { throw MPSGError.commandBufferFailed }
        mm.encode(commandBuffer: cmdBuf, leftMatrix: xMat, rightMatrix: wMat, resultMatrix: yMat)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        logger.info("GraphLinear.forward run() end", category: Logger.Category.textBridge)

        // Read back and add bias on CPU for exact parity with CPU test
        var outHost = [Float](repeating: 0, count: yCount)
        memcpy(&outHost, yBuf.contents(), yCount * elemSize)
        if let bias = bias {
            for bi in 0..<b {
                let base = bi * outFeatures
                for o in 0..<outFeatures { outHost[base + o] += bias.data[o] }
            }
        }
        let outShapeInts = [b, outFeatures]
        let result = Tensor(shape: outShapeInts, data: outHost)
        // Optional CPU validation to help diagnose mismatches during tests
        if outShapeInts.count == 2 && outShapeInts[0] == b && outShapeInts[1] == outFeatures {
            // Compute CPU reference: y[b,o] = sum_i x[b,i] * W[o,i] + b[o]
            var yCPU = [Float](repeating: 0, count: b * outFeatures)
            for bi in 0..<b {
                for o in 0..<outFeatures {
                    var s: Float = 0
                    let xBase = bi * inFeatures
                    let wBase = o * inFeatures
                    for i in 0..<inFeatures { s += x.data[xBase + i] * weight.data[wBase + i] }
                    if let bias = bias { s += bias.data[o] }
                    yCPU[bi * outFeatures + o] = s
                }
            }
            // Compare a few entries
            var maxAbs: Float = 0
            var firstDiffs: [(Int, Float, Float, Float)] = [] // idx, cpu, gpu, abs
            for idx in 0..<(b * outFeatures) {
                let d = abs(yCPU[idx] - outHost[idx])
                if d > maxAbs { maxAbs = d }
                if d > 1e-5 && firstDiffs.count < 5 {
                    firstDiffs.append((idx, yCPU[idx], outHost[idx], d))
                }
            }
            if !firstDiffs.isEmpty {
                logger.warn("GraphLinear CPU≠GPU maxAbs=\(maxAbs) diffs=\(firstDiffs)", category: Logger.Category.textBridge)
            } else {
                logger.debug("GraphLinear CPU≈GPU maxAbs=\(maxAbs)", category: Logger.Category.textBridge)
            }
        }
        logger.info("GraphLinear.forward done out=\(result.prettyShape)", category: Logger.Category.textBridge)
        return result
    }
}
