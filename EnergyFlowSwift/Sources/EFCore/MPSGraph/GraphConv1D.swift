import Foundation
import Metal
import MetalPerformanceShaders

// MARK: - GraphConv1D (FP16, GPU) via im2col + GEMM
// Input x: [B, L, Cin]
// Weight: [Cout, Cin, K]
// Bias: [Cout]
// Forward causal conv with dilation: y[b,t,o] = sum_{i,k} x[b, t - k*d, i] * w[o, i, k] + b[o]
public final class GraphConv1D {
    public let inChannels: Int
    public let outChannels: Int
    public let kernelSize: Int
    public let dilation: Int
    public var weight: Tensor   // [Cout, Cin, K] on host (Float32)
    public var bias: Tensor?    // [Cout] on host (Float32)

    // Cached FP16 weight matrix Wcol [Cout, Cin*K] on GPU
    private var wcolFP16: MTLBuffer?
    // Cached 2D weight for pointwise (K=1) path, host-side [Cout, Cin]
    private var w2DCache: Tensor? = nil

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int = 1, bias: Bool = true, seed: UInt64 = 42) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = max(1, dilation)
        self.weight = Tensor.randomUniform([outChannels, inChannels, kernelSize], min: -1.0/Float(inChannels*kernelSize), max: 1.0/Float(inChannels*kernelSize), seed: seed)
        self.bias = bias ? Tensor.randomUniform([outChannels], min: -0.001, max: 0.001, seed: seed &+ 1) : nil
        self.wcolFP16 = nil
    }

    // Helper: 64-byte aligned rowBytes (conservative for MPSMatrix and better for bandwidth)
    @inline(__always) private func alignedRowBytes(columns: Int, elem: Int) -> Int {
        let raw = columns * elem
        return ((raw + 63) / 64) * 64
    }

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
        precondition(rowBytes >= minRowBytes, "[GraphConv1D] \(name).rowBytes=\(rowBytes) < min required=\(minRowBytes) for cols=\(cols), elem=\(elem)")
        precondition(rowBytes % 64 == 0, "[GraphConv1D] \(name).rowBytes must be 64-byte aligned, got \(rowBytes)")
        let needBytes = (rows - 1) * rowBytes + minRowBytes
        precondition(offset + needBytes <= buffer.length,
                     "[GraphConv1D] OOB for \(name): need=\(needBytes) have=\(buffer.length - offset) rows=\(rows) cols=\(cols) rowBytes=\(rowBytes) elem=\(elem)")
        logger.debug("[GC1D][\(name)] rows=\(rows) cols=\(cols) elem=\(elem) rowBytes=\(rowBytes) need=\(needBytes) buf.len=\(buffer.length) off=\(offset)", category: category)
    }
    #endif

    // Build/copy cached Wcol FP16 buffer [Cout, Cin*K] with aligned rowBytes
    private func ensureWcolFP16(device: MTLDevice) {
        if wcolFP16 != nil { return }
        let CinK = inChannels * kernelSize
        let wCount = outChannels * CinK
        let elemH = MemoryLayout<Float16>.size
        let rowBytes = alignedRowBytes(columns: CinK, elem: elemH)
        wcolFP16 = device.makeBuffer(length: outChannels * rowBytes, options: .storageModeShared)
        guard let wbuf = wcolFP16 else { return }
        wbuf.label = "GraphConv1D.Wcol.fp16"
        // Repack weight [Cout, Cin, K] -> Wcol [Cout, Cin*K]
        var wHalf = [Float16](repeating: 0, count: wCount)
        for o in 0..<outChannels {
            for i in 0..<inChannels {
                for k in 0..<kernelSize {
                    let src = (o * inChannels + i) * kernelSize + k
                    let dst = o * CinK + (i * kernelSize + k)
                    wHalf[dst] = Float16(weight.data[src])
                }
            }
        }
        // Pack row-wise with stride
        wHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                let rowSize = CinK * elemH
                for r in 0..<outChannels {
                    memcpy(wbuf.contents().advanced(by: r * rowBytes), base.advanced(by: r * rowSize), rowSize)
                }
            }
        }
    }

    // Forward on GPU (FP16):
    // 1) Build Xcol [B*L, Cin*K] in FP16 with causal padding and dilation
    // 2) Y = Xcol * Wcol^T -> [B*L, Cout]
    // 3) Add bias on host, reshape -> [B, L, Cout]
public func forward(_ x: Tensor) -> Tensor {
        precondition(x.shape.count == 3, "GraphConv1D.forward expects [B,L,Cin]")
        let B = x.shape[0]
        let L = x.shape[1]
        let Cin = x.shape[2]
        precondition(Cin == inChannels, "Cin mismatch: got \(Cin), expected \(inChannels)")

        let ctx = MPSGContext.shared
        let device = ctx.device
        ensureWcolFP16(device: device)
        guard let wbuf = wcolFP16 else {
            fatalError("GraphConv1D: failed to create Wcol FP16 buffer")
        }

        let CinK = inChannels * kernelSize
        let rows = B * L
        let colsX = CinK
        let colsY = outChannels
        let elemH = MemoryLayout<Float16>.size
    let xRB = alignedRowBytes(columns: colsX, elem: elemH)
    let yRB = alignedRowBytes(columns: colsY, elem: elemH)
        // Allocate Xcol and Y buffers (FP16) with aligned rowBytes
        let xcolBuf = BufferPool.buffer(device: device, length: rows * xRB, label: "GraphConv1D.Xcol")
        let yBuf = BufferPool.buffer(device: device, length: rows * yRB, label: "GraphConv1D.Y")
        #if DEBUG
        debugAssertMatrixLayout(name: "Xcol", rows: rows, cols: colsX, elem: elemH, rowBytes: xRB, buffer: xcolBuf)
        if let wtmp = wcolFP16 {
            debugAssertMatrixLayout(name: "Wcol", rows: outChannels, cols: colsX, elem: elemH, rowBytes: alignedRowBytes(columns: colsX, elem: elemH), buffer: wtmp)
        }
        debugAssertMatrixLayout(name: "Y", rows: rows, cols: colsY, elem: elemH, rowBytes: yRB, buffer: yBuf)
        #endif

        // Build Xcol on CPU as Float16 with causal padding and pack row-wise
        var xcolHalf = [Float16](repeating: 0, count: rows * colsX)
        for b in 0..<B {
            for t in 0..<L {
                let row = b * L + t
                let rowBase = row * colsX
                for i in 0..<inChannels {
                    for k in 0..<kernelSize {
                        let ti = t - k * dilation
                        let dst = rowBase + i * kernelSize + k
                        if ti < 0 {
                            xcolHalf[dst] = Float16(0)
                        } else {
                            let xIdx = (b * L + ti) * inChannels + i
                            xcolHalf[dst] = Float16(x.data[xIdx])
                        }
                    }
                }
            }
        }
        memset(xcolBuf.contents(), 0, rows * xRB)
        xcolHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                let rowSize = colsX * elemH
                for r in 0..<rows { memcpy(xcolBuf.contents().advanced(by: r * xRB), base.advanced(by: r * rowSize), rowSize) }
            }
        }
        memset(yBuf.contents(), 0, rows * yRB)

        // Matrix descriptors with aligned rowBytes
        let xDesc = MPSMatrixDescriptor(rows: rows, columns: colsX, rowBytes: xRB, dataType: .float16)
    let wDesc = MPSMatrixDescriptor(rows: colsY, columns: colsX, rowBytes: alignedRowBytes(columns: colsX, elem: elemH), dataType: .float16)
        let yDesc = MPSMatrixDescriptor(rows: rows, columns: colsY, rowBytes: yRB, dataType: .float16)

        let xMat = MPSMatrix(buffer: xcolBuf, descriptor: xDesc)
        let wMat = MPSMatrix(buffer: wbuf, descriptor: wDesc)
        let yMat = MPSMatrix(buffer: yBuf, descriptor: yDesc)

        let mm = MPSMatrixMultiplication(device: device,
                                         transposeLeft: false,
                                         transposeRight: true,
                                         resultRows: rows,
                                         resultColumns: colsY,
                                         interiorColumns: colsX,
                                         alpha: 1.0,
                                         beta: 0.0)
        mm.label = "GraphConv1D.mm.fwd"
        guard let cmd = ctx.commandQueue.makeCommandBuffer() else {
            fatalError("GraphConv1D: failed to make command buffer")
        }
        cmd.label = "GraphConv1D.forward"
        mm.encode(commandBuffer: cmd, leftMatrix: xMat, rightMatrix: wMat, resultMatrix: yMat)
        cmd.commit(); cmd.waitUntilCompleted()
        #if DEBUG
        let logger = Logger.shared
        switch cmd.status {
        case .completed:
            logger.debug("GraphConv1D.forward commandBuffer completed", category: Logger.Category.textBridge)
        case .error:
            let errDesc = cmd.error?.localizedDescription ?? "unknown"
            logger.debug("GraphConv1D.forward commandBuffer error: \(errDesc)", category: Logger.Category.textBridge)
        default:
            logger.debug("GraphConv1D.forward commandBuffer status=\(cmd.status.rawValue)", category: Logger.Category.textBridge)
        }
        #endif

        // Read back and convert to Float32 (row-wise)
        var yHalf = [Float16](repeating: 0, count: rows * colsY)
        let rowSize = colsY * elemH
        yHalf.withUnsafeMutableBytes { raw in
            if let base = raw.baseAddress {
                for r in 0..<rows {
                    memcpy(base.advanced(by: r * rowSize), yBuf.contents().advanced(by: r * yRB), rowSize)
                }
            }
        }
        var yHost = [Float](repeating: 0, count: rows * colsY)
        for i in 0..<(rows * colsY) { yHost[i] = Float(yHalf[i]) }
        if let bias = bias {
            for r in 0..<rows {
                let base = r * colsY
                for o in 0..<colsY { yHost[base + o] += bias.data[o] }
            }
        }
        return Tensor(shape: [B, L, colsY], data: yHost)
    }

    // Invalidate GPU cached weight matrix (call after weight updates)
    public func invalidateCache() {
        self.wcolFP16 = nil
        self.w2DCache = nil
    }

    // Return repacked 2D weight for pointwise conv (K must be 1). Cached until invalidateCache is called.
    public func weight2DPointwise() -> Tensor {
        precondition(kernelSize == 1, "weight2DPointwise requires kernelSize == 1")
        if let w = w2DCache { return w }
        let Cin = inChannels
        let Cout = outChannels
        var w = Tensor.zeros([Cout, Cin])
        for o in 0..<Cout {
            for i in 0..<Cin {
                w.data[o * Cin + i] = weight.data[(o * Cin + i) * 1 + 0]
            }
        }
        self.w2DCache = w
        return w
    }
}
