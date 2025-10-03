import Foundation
import Metal
import MetalPerformanceShaders

// MARK: - Linear layer (GPU, forward only)
// y = x @ W^T + b
// x: [B, In], W: [Out, In], b: [Out] -> y: [B, Out]
public struct GraphLinear {
    public let inFeatures: Int
    public let outFeatures: Int
    public var weight: Tensor   // CPU-hosted weights
    public var bias: Tensor?    // optional bias

    // Cached GPU buffers for FP16 weights/bias
    private var wBufFP16: MTLBuffer?
    private var bBufFP16: MTLBuffer?

    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, seed: UInt64 = 42) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = Tensor.randomUniform([outFeatures, inFeatures], min: -1.0/Float(inFeatures), max: 1.0/Float(inFeatures), seed: seed)
        self.bias = bias ? Tensor.randomUniform([outFeatures], min: -0.001, max: 0.001, seed: seed &+ 1) : nil
        self.wBufFP16 = nil
        self.bBufFP16 = nil
    }

    // Invalidate GPU caches (call after weight updates)
    public mutating func invalidateCache() {
        self.wBufFP16 = nil
        self.bBufFP16 = nil
    }

    // Compute gradients on GPU for dW using matmul: dW = dY^T · X. dB on CPU by sum rows of dY.
    public func gradientsGPU(X: Tensor, dY: Tensor) throws -> (dW: Tensor, dB: Tensor) {
        precondition(X.shape.count == 2 && dY.shape.count == 2, "gradientsGPU expects 2D tensors")
        let B = X.shape[0]
        precondition(X.shape[1] == inFeatures && dY.shape[1] == outFeatures && dY.shape[0] == B, "shape mismatch: X [B,In], dY [B,Out]")
        let ctx = MPSGContext.shared
        let device = ctx.device
        let elemH = MemoryLayout<Float16>.size
        // Allocate FP16 buffers for dY^T and X
        let rowsL = outFeatures    // dY^T rows
        let colsL = B              // dY^T cols
        let rowsR = B              // X rows
        let colsR = inFeatures     // X cols
        let rowsY = outFeatures
        let colsY = inFeatures
        guard let lBuf = device.makeBuffer(length: rowsL * colsL * elemH, options: .storageModeShared),
              let rBuf = device.makeBuffer(length: rowsR * colsR * elemH, options: .storageModeShared),
              let yBuf = device.makeBuffer(length: rowsY * colsY * elemH, options: .storageModeShared)
        else { throw MPSGError.commandBufferFailed }
        // Pack dY^T (Out x B) and X (B x In) into Float16
        var lHalf = [Float16](repeating: 0, count: rowsL * colsL)
        for b in 0..<B {
            for o in 0..<outFeatures {
                // l[o, b] = dY[b, o]
                lHalf[o * colsL + b] = Float16(dY.data[b * outFeatures + o])
            }
        }
        lHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(lBuf.contents(), base, rowsL * colsL * elemH) } }
        var rHalf = [Float16](repeating: 0, count: rowsR * colsR)
        for b in 0..<B {
            let xBase = b * inFeatures
            let rBase = b * colsR
            for i in 0..<inFeatures { rHalf[rBase + i] = Float16(X.data[xBase + i]) }
        }
        rHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(rBuf.contents(), base, rowsR * colsR * elemH) } }
        memset(yBuf.contents(), 0, rowsY * colsY * elemH)
        // Descriptors
        let lDesc = MPSMatrixDescriptor(rows: rowsL, columns: colsL, rowBytes: colsL * elemH, dataType: .float16)
        let rDesc = MPSMatrixDescriptor(rows: rowsR, columns: colsR, rowBytes: colsR * elemH, dataType: .float16)
        let yDesc = MPSMatrixDescriptor(rows: rowsY, columns: colsY, rowBytes: colsY * elemH, dataType: .float16)
        let lMat = MPSMatrix(buffer: lBuf, descriptor: lDesc)
        let rMat = MPSMatrix(buffer: rBuf, descriptor: rDesc)
        let yMat = MPSMatrix(buffer: yBuf, descriptor: yDesc)
        // Y = L * R (no transpose flags here because we packed L as dY^T)
        let mm = MPSMatrixMultiplication(device: device,
                                         transposeLeft: false,
                                         transposeRight: false,
                                         resultRows: rowsY,
                                         resultColumns: colsY,
                                         interiorColumns: colsL,
                                         alpha: 1.0,
                                         beta: 0.0)
        guard let cmd = ctx.commandQueue.makeCommandBuffer() else { throw MPSGError.commandBufferFailed }
        mm.encode(commandBuffer: cmd, leftMatrix: lMat, rightMatrix: rMat, resultMatrix: yMat)
        cmd.commit(); cmd.waitUntilCompleted()
        // Read back dW as Float
        var yHalf = [Float16](repeating: 0, count: rowsY * colsY)
        memcpy(&yHalf, yBuf.contents(), rowsY * colsY * elemH)
        var dWHost = [Float](repeating: 0, count: rowsY * colsY)
        for i in 0..<(rowsY * colsY) { dWHost[i] = Float(yHalf[i]) }
        let dW = Tensor(shape: [rowsY, colsY], data: dWHost)
        // Bias gradient on CPU
        var dB = Tensor.zeros([outFeatures])
        for b in 0..<B {
            let base = b * outFeatures
            for o in 0..<outFeatures { dB.data[o] += dY.data[base + o] }
        }
        return (dW, dB)
    }

    // Forward pass on GPU
    public mutating func forward(_ x: Tensor) throws -> Tensor {
        let logger = Logger.shared
        let ctx = MPSGContext.shared
        precondition(x.shape.count == 2 && x.shape[1] == inFeatures, "GraphLinear.forward expects [B, inFeatures]")
        let b = x.shape[0]
        logger.info("GraphLinear.forward start B=\(b) In=\(inFeatures) Out=\(outFeatures)", category: Logger.Category.textBridge)

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
