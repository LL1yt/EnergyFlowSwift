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
    private var wRowBytesFP16: Int = 0
    private var bBufFP16: MTLBuffer?

    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, seed: UInt64 = 42) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = Tensor.randomUniform([outFeatures, inFeatures], min: -1.0/Float(inFeatures), max: 1.0/Float(inFeatures), seed: seed)
        self.bias = bias ? Tensor.randomUniform([outFeatures], min: -0.001, max: 0.001, seed: seed &+ 1) : nil
        self.wBufFP16 = nil
        self.bBufFP16 = nil
    }

    // Helper: 16-byte aligned rowBytes for MPSMatrix (required)
    @inline(__always) private func alignedRowBytes(columns: Int, elem: Int) -> Int {
        let raw = columns * elem
        return ((raw + 15) / 16) * 16
    }

    // DEBUG-only: validate matrix layout and buffer bounds before encoding GPU work
    #if DEBUG
    @inline(__always) private func debugAssertMatrixLayout(name: String,
                                                           rows: Int,
                                                           cols: Int,
                                                           elem: Int,
                                                           rowBytes: Int,
                                                           buffer: MTLBuffer,
                                                           offset: Int = 0,
                                                           category: String = Logger.Category.textBridge) {
        let logger = Logger.shared
        let minRowBytes = cols * elem
        precondition(rowBytes >= minRowBytes, "[GraphLinear] \(name).rowBytes=\(rowBytes) < min required=\(minRowBytes) for cols=\(cols), elem=\(elem)")
        precondition(rowBytes % 16 == 0, "[GraphLinear] \(name).rowBytes must be 16-byte aligned, got \(rowBytes)")
        let needBytes = (rows - 1) * rowBytes + minRowBytes
        precondition(offset + needBytes <= buffer.length,
                     "[GraphLinear] OOB for \(name): need=\(needBytes) have=\(buffer.length - offset) rows=\(rows) cols=\(cols) rowBytes=\(rowBytes) elem=\(elem)")
        logger.debug("[GL][\(name)] rows=\(rows) cols=\(cols) elem=\(elem) rowBytes=\(rowBytes) need=\(needBytes) buf.len=\(buffer.length) off=\(offset)", category: category)
    }
    #endif

    // Invalidate GPU caches (call after weight updates)
    public mutating func invalidateCache() {
        self.wBufFP16 = nil
        self.wRowBytesFP16 = 0
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
        // Use instance helper alignedRowBytes
        @inline(__always) func copyRowsToBuffer(src: UnsafeRawPointer, dst: UnsafeMutableRawPointer, rows: Int, cols: Int, elem: Int, rowBytes: Int) {
            let rowSize = cols * elem
            for r in 0..<rows { memcpy(dst.advanced(by: r * rowBytes), src.advanced(by: r * rowSize), rowSize) }
        }
        @inline(__always) func copyRowsFromBuffer(dst: UnsafeMutableRawPointer, src: UnsafeRawPointer, rows: Int, cols: Int, elem: Int, rowBytes: Int) {
            let rowSize = cols * elem
            for r in 0..<rows { memcpy(dst.advanced(by: r * rowSize), src.advanced(by: r * rowBytes), rowSize) }
        }
        // Allocate FP16 buffers for dY^T and X
        let rowsL = outFeatures    // dY^T rows
        let colsL = B              // dY^T cols
        let rowsR = B              // X rows
        let colsR = inFeatures     // X cols
        let rowsY = outFeatures
        let colsY = inFeatures
        let lRowBytes = alignedRowBytes(columns: colsL, elem: elemH)
        let rRowBytes = alignedRowBytes(columns: colsR, elem: elemH)
        let yRowBytes = alignedRowBytes(columns: colsY, elem: elemH)
        let lBuf = BufferPool.buffer(device: device, length: rowsL * lRowBytes, label: "GraphLinear.grad.L")
        let rBuf = BufferPool.buffer(device: device, length: rowsR * rRowBytes, label: "GraphLinear.grad.R")
        let yBuf = BufferPool.buffer(device: device, length: rowsY * yRowBytes, label: "GraphLinear.grad.Y")
        #if DEBUG
        debugAssertMatrixLayout(name: "dY^T", rows: rowsL, cols: colsL, elem: elemH, rowBytes: lRowBytes, buffer: lBuf)
        debugAssertMatrixLayout(name: "X", rows: rowsR, cols: colsR, elem: elemH, rowBytes: rRowBytes, buffer: rBuf)
        debugAssertMatrixLayout(name: "dW(out,in)", rows: rowsY, cols: colsY, elem: elemH, rowBytes: yRowBytes, buffer: yBuf)
        #endif
        // Pack dY^T (Out x B) and X (B x In) into Float16
        var lHalf = [Float16](repeating: 0, count: rowsL * colsL)
        for b in 0..<B {
            for o in 0..<outFeatures {
                // l[o, b] = dY[b, o]
                lHalf[o * colsL + b] = Float16(dY.data[b * outFeatures + o])
            }
        }
        memset(lBuf.contents(), 0, rowsL * lRowBytes)
        lHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { copyRowsToBuffer(src: base, dst: lBuf.contents(), rows: rowsL, cols: colsL, elem: elemH, rowBytes: lRowBytes) } }
        var rHalf = [Float16](repeating: 0, count: rowsR * colsR)
        for b in 0..<B {
            let xBase = b * inFeatures
            let rBase = b * colsR
            for i in 0..<inFeatures { rHalf[rBase + i] = Float16(X.data[xBase + i]) }
        }
        memset(rBuf.contents(), 0, rowsR * rRowBytes)
        rHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { copyRowsToBuffer(src: base, dst: rBuf.contents(), rows: rowsR, cols: colsR, elem: elemH, rowBytes: rRowBytes) } }
        memset(yBuf.contents(), 0, rowsY * yRowBytes)
        // Descriptors
        let lDesc = MPSMatrixDescriptor(rows: rowsL, columns: colsL, rowBytes: lRowBytes, dataType: .float16)
        let rDesc = MPSMatrixDescriptor(rows: rowsR, columns: colsR, rowBytes: rRowBytes, dataType: .float16)
        let yDesc = MPSMatrixDescriptor(rows: rowsY, columns: colsY, rowBytes: yRowBytes, dataType: .float16)
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
        mm.label = "GraphLinear.mm.gradW"
        guard let cmd = ctx.commandQueue.makeCommandBuffer() else { throw MPSGError.commandBufferFailed }
        cmd.label = "GraphLinear.gradientsGPU"
        mm.encode(commandBuffer: cmd, leftMatrix: lMat, rightMatrix: rMat, resultMatrix: yMat)
        cmd.commit(); cmd.waitUntilCompleted()
        #if DEBUG
        let logger = Logger.shared
        switch cmd.status {
        case .completed:
            logger.debug("GraphLinear.gradientsGPU commandBuffer completed", category: Logger.Category.textBridge)
        case .error:
            let errDesc = cmd.error?.localizedDescription ?? "unknown"
            logger.debug("GraphLinear.gradientsGPU commandBuffer error: \(errDesc)", category: Logger.Category.textBridge)
        default:
            logger.debug("GraphLinear.gradientsGPU commandBuffer status=\(cmd.status.rawValue)", category: Logger.Category.textBridge)
        }
        #endif
        // Read back dW as Float
    var yHalf = [Float16](repeating: 0, count: rowsY * colsY)
    yHalf.withUnsafeMutableBytes { raw in if let base = raw.baseAddress { copyRowsFromBuffer(dst: base, src: yBuf.contents(), rows: rowsY, cols: colsY, elem: elemH, rowBytes: yRowBytes) } }
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
        logger.debug("GraphLinear.forward start B=\(b) In=\(inFeatures) Out=\(outFeatures)", category: Logger.Category.textBridge)

        // FP16 path: convert x and cached W/b to Float16, run matmul, read back and convert to Float
        let elemSizeH = MemoryLayout<Float16>.size
        let xCount = b * inFeatures
        let wCount = outFeatures * inFeatures
        let yCount = b * outFeatures
        // Use instance helper alignedRowBytes
        @inline(__always) func copyRowsToBuffer(src: UnsafeRawPointer, dst: UnsafeMutableRawPointer, rows: Int, cols: Int, elem: Int, rowBytes: Int) {
            let rowSize = cols * elem
            for r in 0..<rows { memcpy(dst.advanced(by: r * rowBytes), src.advanced(by: r * rowSize), rowSize) }
        }
        @inline(__always) func copyRowsFromBuffer(dst: UnsafeMutableRawPointer, src: UnsafeRawPointer, rows: Int, cols: Int, elem: Int, rowBytes: Int) {
            let rowSize = cols * elem
            for r in 0..<rows { memcpy(dst.advanced(by: r * rowSize), src.advanced(by: r * rowBytes), rowSize) }
        }

        // Ensure/copy weight & bias buffers once (cached) in FP16 with aligned rowBytes
        if wBufFP16 == nil {
            // Compute aligned rowBytes for W rows
            let wRB = alignedRowBytes(columns: inFeatures, elem: elemSizeH)
            self.wRowBytesFP16 = wRB
            wBufFP16 = ctx.device.makeBuffer(length: outFeatures * wRB, options: .storageModeShared)
            guard let wb = wBufFP16 else { throw MPSGError.commandBufferFailed }
            wb.label = "GraphLinear.W.fp16"
            // Convert weight to Float16 then pack rows with stride
            var wHalf = [Float16](repeating: 0, count: wCount)
            for i in 0..<wCount { wHalf[i] = Float16(weight.data[i]) }
            wHalf.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    let rowSize = inFeatures * elemSizeH
                    for r in 0..<outFeatures {
                        memcpy(wb.contents().advanced(by: r * wRB), base.advanced(by: r * rowSize), rowSize)
                    }
                }
            }
            // Bias buffer (packed tightly, no row stride)
            if let bias = bias {
                bBufFP16 = ctx.device.makeBuffer(length: outFeatures * elemSizeH, options: .storageModeShared)
                if let bb = bBufFP16 {
                    bb.label = "GraphLinear.b.fp16"
                    var bHalf = [Float16](repeating: 0, count: outFeatures)
                    for i in 0..<outFeatures { bHalf[i] = Float16(bias.data[i]) }
                    bHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(bb.contents(), base, outFeatures * elemSizeH) } }
                }
            }
        }
        guard let wBuf = wBufFP16 else { throw MPSGError.commandBufferFailed }

        // Allocate and fill x in FP16
        let xRowBytes = alignedRowBytes(columns: inFeatures, elem: elemSizeH)
        let yRowBytes = alignedRowBytes(columns: outFeatures, elem: elemSizeH)
        let xBuf = BufferPool.buffer(device: ctx.device, length: b * xRowBytes, label: "GraphLinear.fwd.X")
        let yBuf = BufferPool.buffer(device: ctx.device, length: b * yRowBytes, label: "GraphLinear.fwd.Y")
        #if DEBUG
        debugAssertMatrixLayout(name: "X", rows: b, cols: inFeatures, elem: elemSizeH, rowBytes: xRowBytes, buffer: xBuf, offset: 0)
        debugAssertMatrixLayout(name: "W", rows: outFeatures, cols: inFeatures, elem: elemSizeH, rowBytes: self.wRowBytesFP16 > 0 ? self.wRowBytesFP16 : alignedRowBytes(columns: inFeatures, elem: elemSizeH), buffer: wBuf, offset: 0)
        debugAssertMatrixLayout(name: "Y", rows: b, cols: outFeatures, elem: elemSizeH, rowBytes: yRowBytes, buffer: yBuf, offset: 0)
        #endif
        var xHalf = [Float16](repeating: 0, count: xCount)
        for i in 0..<xCount { xHalf[i] = Float16(x.data[i]) }
        memset(xBuf.contents(), 0, b * xRowBytes)
        xHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { copyRowsToBuffer(src: base, dst: xBuf.contents(), rows: b, cols: inFeatures, elem: elemSizeH, rowBytes: xRowBytes) } }
        memset(yBuf.contents(), 0, b * yRowBytes)

        // Descriptors (row-major) in FP16
        let xDesc = MPSMatrixDescriptor(rows: b, columns: inFeatures, rowBytes: xRowBytes, dataType: .float16)
        let wDesc = MPSMatrixDescriptor(rows: outFeatures, columns: inFeatures, rowBytes: self.wRowBytesFP16 > 0 ? self.wRowBytesFP16 : ((inFeatures * elemSizeH + 15) / 16) * 16, dataType: .float16)
        let yDesc = MPSMatrixDescriptor(rows: b, columns: outFeatures, rowBytes: yRowBytes, dataType: .float16)

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
        mm.label = "GraphLinear.mm.fwd"
        logger.debug("GraphLinear.forward run() begin [fp16]", category: Logger.Category.textBridge)
        guard let cmdBuf = ctx.commandQueue.makeCommandBuffer() else { throw MPSGError.commandBufferFailed }
        cmdBuf.label = "GraphLinear.forward"
        mm.encode(commandBuffer: cmdBuf, leftMatrix: xMat, rightMatrix: wMat, resultMatrix: yMat)
        cmdBuf.commit(); cmdBuf.waitUntilCompleted()
        #if DEBUG
        switch cmdBuf.status {
        case .completed:
            logger.debug("GraphLinear.forward commandBuffer completed", category: Logger.Category.textBridge)
        case .error:
            let errDesc = cmdBuf.error?.localizedDescription ?? "unknown"
            logger.debug("GraphLinear.forward commandBuffer error: \(errDesc)", category: Logger.Category.textBridge)
        default:
            logger.debug("GraphLinear.forward commandBuffer status=\(cmdBuf.status.rawValue)", category: Logger.Category.textBridge)
        }
        #endif
        logger.debug("GraphLinear.forward run() end [fp16]", category: Logger.Category.textBridge)

        // Read back as Float16 then convert to Float
        var yHalf = [Float16](repeating: 0, count: yCount)
        yHalf.withUnsafeMutableBytes { raw in if let base = raw.baseAddress { copyRowsFromBuffer(dst: base, src: yBuf.contents(), rows: b, cols: outFeatures, elem: elemSizeH, rowBytes: yRowBytes) } }
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
        logger.debug("GraphLinear.forward done out=\(result.prettyShape) [fp16]", category: Logger.Category.textBridge)
        return result
    }

    // MARK: - Backward: input gradients dX = dY @ W
    public func inputGradientsGPU(dY: Tensor) throws -> Tensor {
        precondition(dY.shape.count == 2 && dY.shape[1] == outFeatures, "dY must be [B, Out]")
        let B = dY.shape[0]
        let ctx = MPSGContext.shared
        let device = ctx.device
        let elemH = MemoryLayout<Float16>.size
    // Use instance helper alignedRowBytes
        @inline(__always) func copyRowsToBuffer(src: UnsafeRawPointer, dst: UnsafeMutableRawPointer, rows: Int, cols: Int, elem: Int, rowBytes: Int) { let rowSize = cols * elem; for r in 0..<rows { memcpy(dst.advanced(by: r * rowBytes), src.advanced(by: r * rowSize), rowSize) } }
        @inline(__always) func copyRowsFromBuffer(dst: UnsafeMutableRawPointer, src: UnsafeRawPointer, rows: Int, cols: Int, elem: Int, rowBytes: Int) { let rowSize = cols * elem; for r in 0..<rows { memcpy(dst.advanced(by: r * rowSize), src.advanced(by: r * rowBytes), rowSize) } }
        // Ensure weight buffer exists; if not, build a local aligned FP16 copy
        
        let wRB = self.wRowBytesFP16 > 0 ? self.wRowBytesFP16 : alignedRowBytes(columns: inFeatures, elem: elemH)
        let localWB: MTLBuffer
        if let cached = wBufFP16 {
            localWB = cached
        } else {
            // Build a transient aligned buffer for W
            let wCount = outFeatures * inFeatures
            guard let wb = device.makeBuffer(length: outFeatures * wRB, options: .storageModeShared) else { throw MPSGError.commandBufferFailed }
            wb.label = "GraphLinear.W.fp16.local"
            var wHalf = [Float16](repeating: 0, count: wCount)
            for i in 0..<wCount { wHalf[i] = Float16(weight.data[i]) }
            wHalf.withUnsafeBytes { raw in if let base = raw.baseAddress {
                let rowSize = inFeatures * elemH
                for r in 0..<outFeatures { memcpy(wb.contents().advanced(by: r * wRB), base.advanced(by: r * rowSize), rowSize) }
            }}
            localWB = wb
        }
        // Allocate buffers for dY and dX (FP16)
        let dyCount = B * outFeatures
        let dxCount = B * inFeatures
        let dyRowBytes = alignedRowBytes(columns: outFeatures, elem: elemH)
        let dxRowBytes = alignedRowBytes(columns: inFeatures, elem: elemH)
        let dyBuf = BufferPool.buffer(device: device, length: B * dyRowBytes, label: "GraphLinear.dX.DY")
        let dxBuf = BufferPool.buffer(device: device, length: B * dxRowBytes, label: "GraphLinear.dX.DX")
        #if DEBUG
        debugAssertMatrixLayout(name: "dY[B,Out]", rows: B, cols: outFeatures, elem: elemH, rowBytes: dyRowBytes, buffer: dyBuf)
        debugAssertMatrixLayout(name: "W[Out,In]", rows: outFeatures, cols: inFeatures, elem: elemH, rowBytes: wRB, buffer: localWB)
        debugAssertMatrixLayout(name: "dX[B,In]", rows: B, cols: inFeatures, elem: elemH, rowBytes: dxRowBytes, buffer: dxBuf)
        #endif
        var dyHalf = [Float16](repeating: 0, count: dyCount)
        for i in 0..<dyCount { dyHalf[i] = Float16(dY.data[i]) }
        memset(dyBuf.contents(), 0, B * dyRowBytes)
        dyHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { copyRowsToBuffer(src: base, dst: dyBuf.contents(), rows: B, cols: outFeatures, elem: elemH, rowBytes: dyRowBytes) } }
        memset(dxBuf.contents(), 0, B * dxRowBytes)
        // Descriptors: dY [B, Out], W [Out, In], dX [B, In]
        let dyDesc = MPSMatrixDescriptor(rows: B, columns: outFeatures, rowBytes: dyRowBytes, dataType: .float16)
        let wDesc = MPSMatrixDescriptor(rows: outFeatures, columns: inFeatures, rowBytes: wRB, dataType: .float16)
        let dxDesc = MPSMatrixDescriptor(rows: B, columns: inFeatures, rowBytes: dxRowBytes, dataType: .float16)
        let dyMat = MPSMatrix(buffer: dyBuf, descriptor: dyDesc)
        let wMat = MPSMatrix(buffer: localWB, descriptor: wDesc)
        let dxMat = MPSMatrix(buffer: dxBuf, descriptor: dxDesc)
        let mm = MPSMatrixMultiplication(device: device,
                                         transposeLeft: false,
                                         transposeRight: false,
                                         resultRows: B,
                                         resultColumns: inFeatures,
                                         interiorColumns: outFeatures,
                                         alpha: 1.0,
                                         beta: 0.0)
        mm.label = "GraphLinear.mm.dX"
        guard let cmd = ctx.commandQueue.makeCommandBuffer() else { throw MPSGError.commandBufferFailed }
        cmd.label = "GraphLinear.inputGradientsGPU"
        mm.encode(commandBuffer: cmd, leftMatrix: dyMat, rightMatrix: wMat, resultMatrix: dxMat)
        cmd.commit(); cmd.waitUntilCompleted()
        #if DEBUG
        let logger = Logger.shared
        switch cmd.status {
        case .completed:
            logger.debug("GraphLinear.inputGradientsGPU commandBuffer completed", category: Logger.Category.textBridge)
        case .error:
            let errDesc = cmd.error?.localizedDescription ?? "unknown"
            logger.debug("GraphLinear.inputGradientsGPU commandBuffer error: \(errDesc)", category: Logger.Category.textBridge)
        default:
            logger.debug("GraphLinear.inputGradientsGPU commandBuffer status=\(cmd.status.rawValue)", category: Logger.Category.textBridge)
        }
        #endif
        var dxHalf = [Float16](repeating: 0, count: dxCount)
        dxHalf.withUnsafeMutableBytes { raw in if let base = raw.baseAddress { copyRowsFromBuffer(dst: base, src: dxBuf.contents(), rows: B, cols: inFeatures, elem: elemH, rowBytes: dxRowBytes) } }
        var dxHost = [Float](repeating: 0, count: dxCount)
        for i in 0..<dxCount { dxHost[i] = Float(dxHalf[i]) }
        return Tensor(shape: [B, inFeatures], data: dxHost)
    }
}
