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

    public init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int = 1, bias: Bool = true, seed: UInt64 = 42) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = max(1, dilation)
        self.weight = Tensor.randomUniform([outChannels, inChannels, kernelSize], min: -1.0/Float(inChannels*kernelSize), max: 1.0/Float(inChannels*kernelSize), seed: seed)
        self.bias = bias ? Tensor.randomUniform([outChannels], min: -0.001, max: 0.001, seed: seed &+ 1) : nil
        self.wcolFP16 = nil
    }

    // Build/copy cached Wcol FP16 buffer [Cout, Cin*K]
private func ensureWcolFP16(device: MTLDevice) {
        if wcolFP16 != nil { return }
        let CinK = inChannels * kernelSize
        let wCount = outChannels * CinK
        wcolFP16 = device.makeBuffer(length: wCount * MemoryLayout<Float16>.size, options: .storageModeShared)
        guard let wbuf = wcolFP16 else { return }
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
        wHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(wbuf.contents(), base, wCount * MemoryLayout<Float16>.size)
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

        // Allocate Xcol and Y buffers (FP16)
        guard let xcolBuf = device.makeBuffer(length: rows * colsX * elemH, options: .storageModeShared),
              let yBuf = device.makeBuffer(length: rows * colsY * elemH, options: .storageModeShared)
        else { fatalError("GraphConv1D: failed to allocate buffers") }

        // Build Xcol on CPU as Float16 with causal padding
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
        xcolHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress { memcpy(xcolBuf.contents(), base, rows * colsX * elemH) }
        }
        memset(yBuf.contents(), 0, rows * colsY * elemH)

        // Matrix descriptors
        let xDesc = MPSMatrixDescriptor(rows: rows, columns: colsX, rowBytes: colsX * elemH, dataType: .float16)
        let wDesc = MPSMatrixDescriptor(rows: colsY, columns: colsX, rowBytes: colsX * elemH, dataType: .float16)
        let yDesc = MPSMatrixDescriptor(rows: rows, columns: colsY, rowBytes: colsY * elemH, dataType: .float16)

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
        guard let cmd = ctx.commandQueue.makeCommandBuffer() else {
            fatalError("GraphConv1D: failed to make command buffer")
        }
        mm.encode(commandBuffer: cmd, leftMatrix: xMat, rightMatrix: wMat, resultMatrix: yMat)
        cmd.commit(); cmd.waitUntilCompleted()

        // Read back and convert to Float32
        var yHalf = [Float16](repeating: 0, count: rows * colsY)
        memcpy(&yHalf, yBuf.contents(), rows * colsY * elemH)
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
    }
}
