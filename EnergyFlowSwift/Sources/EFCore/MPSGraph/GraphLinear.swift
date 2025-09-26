import Foundation
import Metal
import MetalPerformanceShaders

// MARK: - Linear layer (GPU, forward only)
// y = x @ W^T + b
// x: [B, In], W: [Out, In], b: [Out] -> y: [B, Out]
public struct GraphLinear {
    public enum Precision {
        case fp32
        case fp16
    }

    public let inFeatures: Int
    public let outFeatures: Int
    public var weight: Tensor   // CPU-hosted weights
    public var bias: Tensor?    // optional bias
    public var precision: Precision

    // Cached GPU buffers for weights/bias
    private var wBufFP32: MTLBuffer?
    private var bBufFP32: MTLBuffer?
    private var wBufFP16: MTLBuffer?
    private var bBufFP16: MTLBuffer?

    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, seed: UInt64 = 42, precision: Precision = .fp32) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = Tensor.randomUniform([outFeatures, inFeatures], min: -1.0/Float(inFeatures), max: 1.0/Float(inFeatures), seed: seed)
        self.bias = bias ? Tensor.randomUniform([outFeatures], min: -0.001, max: 0.001, seed: seed &+ 1) : nil
        self.precision = precision
        self.wBufFP32 = nil
        self.bBufFP32 = nil
        self.wBufFP16 = nil
        self.bBufFP16 = nil
    }

    // Forward pass on GPU
    public mutating func forward(_ x: Tensor) throws -> Tensor {
        let logger = Logger.shared
        let ctx = MPSGContext.shared
        precondition(x.shape.count == 2 && x.shape[1] == inFeatures, "GraphLinear.forward expects [B, inFeatures]")
        let b = x.shape[0]
        logger.info("GraphLinear.forward start B=\(b) In=\(inFeatures) Out=\(outFeatures) precision=\(precision)", category: Logger.Category.textBridge)

        switch precision {
        case .fp32:
            // GPU matmul via MPSMatrixMultiplication: y = x @ W^T
            let elemSize = MemoryLayout<Float>.size
            let xCount = b * inFeatures
            let wCount = outFeatures * inFeatures
            let yCount = b * outFeatures

            guard let xBuf = ctx.device.makeBuffer(length: xCount * elemSize, options: .storageModeShared),
                  let yBuf = ctx.device.makeBuffer(length: yCount * elemSize, options: .storageModeShared)
            else { throw MPSGError.commandBufferFailed }

            // Ensure/copy weight buffer once (cached)
            if wBufFP32 == nil {
                wBufFP32 = ctx.device.makeBuffer(length: wCount * elemSize, options: .storageModeShared)
                guard let wb = wBufFP32 else { throw MPSGError.commandBufferFailed }
                weight.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(wb.contents(), base, wCount * elemSize) } }
                if let bias = bias {
                    bBufFP32 = ctx.device.makeBuffer(length: outFeatures * elemSize, options: .storageModeShared)
                    if let bb = bBufFP32 { bias.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(bb.contents(), base, outFeatures * elemSize) } } }
                }
            }
            guard let wBuf = wBufFP32 else { throw MPSGError.commandBufferFailed }

            // Copy host input into shared buffer
            x.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(xBuf.contents(), base, xCount * elemSize) } }
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
            cmdBuf.commit(); cmdBuf.waitUntilCompleted()
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
            // CPU validation kept as before
            if outShapeInts.count == 2 && outShapeInts[0] == b && outShapeInts[1] == outFeatures {
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
                var maxAbs: Float = 0
                var firstDiffs: [(Int, Float, Float, Float)] = []
                for idx in 0..<(b * outFeatures) {
                    let d = abs(yCPU[idx] - outHost[idx])
                    if d > maxAbs { maxAbs = d }
                    if d > 1e-5 && firstDiffs.count < 5 { firstDiffs.append((idx, yCPU[idx], outHost[idx], d)) }
                }
                if !firstDiffs.isEmpty { logger.warn("GraphLinear CPU≠GPU maxAbs=\(maxAbs) diffs=\(firstDiffs)", category: Logger.Category.textBridge) }
                else { logger.debug("GraphLinear CPU≈GPU maxAbs=\(maxAbs)", category: Logger.Category.textBridge) }
            }
            logger.info("GraphLinear.forward done out=\(result.prettyShape)", category: Logger.Category.textBridge)
            return result

        case .fp16:
            // FP16 path: convert x and cached W/b to Float16, run matmul, read back and convert to Float
            let elemSizeH = MemoryLayout<Float16>.size
            let xCount = b * inFeatures
            let wCount = outFeatures * inFeatures
            let yCount = b * outFeatures

            // Ensure/copy weight & bias buffers once (cached) in FP16
            if wBufFP16 == nil {
                wBufFP16 = ctx.device.makeBuffer(length: wCount * elemSizeH, options: .storageModeShared)
                guard let wb = wBufFP16 else { throw MPSGError.commandBufferFailed }
                // Convert weight to Float16
                var wHalf = [Float16](repeating: 0, count: wCount)
                for i in 0..<wCount { wHalf[i] = Float16(weight.data[i]) }
                wHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(wb.contents(), base, wCount * elemSizeH) } }
                if let bias = bias {
                    bBufFP16 = ctx.device.makeBuffer(length: outFeatures * elemSizeH, options: .storageModeShared)
                    if let bb = bBufFP16 {
                        var bHalf = [Float16](repeating: 0, count: outFeatures)
                        for i in 0..<outFeatures { bHalf[i] = Float16(bias.data[i]) }
                        bHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(bb.contents(), base, outFeatures * elemSizeH) } }
                    }
                }
            }
            guard let wBuf = wBufFP16 else { throw MPSGError.commandBufferFailed }

            // Allocate and fill x in FP16
            guard let xBuf = ctx.device.makeBuffer(length: xCount * elemSizeH, options: .storageModeShared),
                  let yBuf = ctx.device.makeBuffer(length: yCount * elemSizeH, options: .storageModeShared)
            else { throw MPSGError.commandBufferFailed }
            var xHalf = [Float16](repeating: 0, count: xCount)
            for i in 0..<xCount { xHalf[i] = Float16(x.data[i]) }
            xHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(xBuf.contents(), base, xCount * elemSizeH) } }
            memset(yBuf.contents(), 0, yCount * elemSizeH)

            // Descriptors (row-major) in FP16
            let xDesc = MPSMatrixDescriptor(rows: b, columns: inFeatures, rowBytes: inFeatures * elemSizeH, dataType: .float16)
            let wDesc = MPSMatrixDescriptor(rows: outFeatures, columns: inFeatures, rowBytes: inFeatures * elemSizeH, dataType: .float16)
            let yDesc = MPSMatrixDescriptor(rows: b, columns: outFeatures, rowBytes: outFeatures * elemSizeH, dataType: .float16)

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
            logger.info("GraphLinear.forward run() begin [fp16]", category: Logger.Category.textBridge)
            guard let cmdBuf = ctx.commandQueue.makeCommandBuffer() else { throw MPSGError.commandBufferFailed }
            mm.encode(commandBuffer: cmdBuf, leftMatrix: xMat, rightMatrix: wMat, resultMatrix: yMat)
            cmdBuf.commit(); cmdBuf.waitUntilCompleted()
            logger.info("GraphLinear.forward run() end [fp16]", category: Logger.Category.textBridge)

            // Read back as Float16 then convert to Float
            var yHalf = [Float16](repeating: 0, count: yCount)
            memcpy(&yHalf, yBuf.contents(), yCount * elemSizeH)
            var outHost = [Float](repeating: 0, count: yCount)
            for i in 0..<yCount { outHost[i] = Float(yHalf[i]) }
            if let bias = bias {
                for bi in 0..<b {
                    let base = bi * outFeatures
                    for o in 0..<outFeatures { outHost[base + o] += bias.data[o] }
                }
            }
            let outShapeInts = [b, outFeatures]
            let result = Tensor(shape: outShapeInts, data: outHost)
            logger.info("GraphLinear.forward done out=\(result.prettyShape) [fp16]", category: Logger.Category.textBridge)
            return result
        }
    }
}
