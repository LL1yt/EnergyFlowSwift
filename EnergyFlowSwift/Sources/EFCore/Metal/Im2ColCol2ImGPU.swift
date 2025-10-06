import Foundation
import Metal

// GPU helpers for 1D causal dilated im2col/col2im using Metal compute shaders (float32)
// Shapes:
// - X:    [B, L, Cin]
// - Xcol: [B*L, Cin*K]
// - dXcol same as Xcol
// - dX:   [B, L, Cin]

public enum Im2ColCol2ImGPU {
    // Lazy-compiled pipelines
    nonisolated(unsafe) private static var library: MTLLibrary? = nil
    nonisolated(unsafe) private static var pIm2Col: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pCol2Im: MTLComputePipelineState? = nil

    private static func ensurePipelines(device: MTLDevice) throws {
        if let _ = pIm2Col, let _ = pCol2Im { return }
        let source = Self.metalSource
        if library == nil {
            library = try device.makeLibrary(source: source, options: nil)
        }
        guard let lib = library else { throw MPSGError.contextUnavailable }
        if pIm2Col == nil {
            let fn = lib.makeFunction(name: "im2col_1d_causal_f32")!
            pIm2Col = try device.makeComputePipelineState(function: fn)
        }
        if pCol2Im == nil {
            let fn = lib.makeFunction(name: "col2im_1d_causal_f32")!
            pCol2Im = try device.makeComputePipelineState(function: fn)
        }
    }

    public static func im2col(X: Tensor, B: Int, L: Int, Cin: Int, K: Int, dilation: Int) throws -> Tensor {
        precondition(X.shape == [B, L, Cin], "im2col expects X [B,L,Cin]")
        let ctx = MPSGContext.shared
        try ensurePipelines(device: ctx.device)
        guard let pipe = pIm2Col, let cmd = ctx.commandQueue.makeCommandBuffer() else { throw MPSGError.commandBufferFailed }
        let rows = B * L
        let colsX = Cin * K
        let outCount = rows * colsX
        let inCount = B * L * Cin
        let elem = MemoryLayout<Float>.size
        let inBuf = BufferPool.buffer(device: ctx.device, length: inCount * elem, label: "im2col.in")
        let outBuf = BufferPool.buffer(device: ctx.device, length: outCount * elem, label: "im2col.out")
        // Copy X to GPU
        X.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(inBuf.contents(), base, inCount * elem) } }
        memset(outBuf.contents(), 0, outCount * elem)
        // Encode kernel
        guard let enc = cmd.makeComputeCommandEncoder() else { throw MPSGError.commandBufferFailed }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vCin = Int32(Cin), vK = Int32(K), vDil = Int32(max(1, dilation))
        enc.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        enc.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 4)
        enc.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 5)
        enc.setBytes(&vDil, length: MemoryLayout<Int32>.size, index: 6)
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreads = outCount
        let threadgroups = MTLSize(width: (numThreads + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        // Read back
        var host = [Float](repeating: 0, count: outCount)
        memcpy(&host, outBuf.contents(), outCount * elem)
        return Tensor(shape: [rows, colsX], data: host)
    }

    public static func col2im(dXcol: Tensor, B: Int, L: Int, Cin: Int, K: Int, dilation: Int) throws -> Tensor {
        precondition(dXcol.shape == [B * L, Cin * K], "col2im expects dXcol [B*L, Cin*K]")
        let ctx = MPSGContext.shared
        try ensurePipelines(device: ctx.device)
        guard let pipe = pCol2Im, let cmd = ctx.commandQueue.makeCommandBuffer() else { throw MPSGError.commandBufferFailed }
        let rows = B * L
        let colsX = Cin * K
        let inCount = rows * colsX
        let outCount = B * L * Cin
        let elem = MemoryLayout<Float>.size
        let inBuf = BufferPool.buffer(device: ctx.device, length: inCount * elem, label: "col2im.in")
        let outBuf = BufferPool.buffer(device: ctx.device, length: outCount * elem, label: "col2im.out")
        dXcol.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(inBuf.contents(), base, inCount * elem) } }
        memset(outBuf.contents(), 0, outCount * elem)
        guard let enc = cmd.makeComputeCommandEncoder() else { throw MPSGError.commandBufferFailed }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vCin = Int32(Cin), vK = Int32(K), vDil = Int32(max(1, dilation))
        enc.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        enc.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 4)
        enc.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 5)
        enc.setBytes(&vDil, length: MemoryLayout<Int32>.size, index: 6)
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreads = outCount
        let threadgroups = MTLSize(width: (numThreads + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        var host = [Float](repeating: 0, count: outCount)
        memcpy(&host, outBuf.contents(), outCount * elem)
        return Tensor(shape: [B, L, Cin], data: host)
    }

    // Metal compute kernels (float32)
    private static let metalSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void im2col_1d_causal_f32(
        const device float* X          [[buffer(0)]],
        device float*       Xcol       [[buffer(1)]],
        constant int&       B          [[buffer(2)]],
        constant int&       L          [[buffer(3)]],
        constant int&       Cin        [[buffer(4)]],
        constant int&       K          [[buffer(5)]],
        constant int&       dilation   [[buffer(6)]],
        uint                gid        [[thread_position_in_grid]])
    {
        // total elements = rows * colsX = (B*L) * (Cin*K)
        int rows = B * L;
        int colsX = Cin * K;
        if ((int)gid >= rows * colsX) return;
        int row = gid / colsX;       // 0..B*L-1
        int col = gid % colsX;       // 0..Cin*K-1
        int i = col / K;             // input channel 0..Cin-1
        int k = col % K;             // kernel index 0..K-1
        int b = row / L;             // batch 0..B-1
        int t = row % L;             // time 0..L-1
        int ti = t - k * dilation;   // source time index
        float val = 0.0f;
        if (ti >= 0) {
            int xIndex = (b * L + ti) * Cin + i;
            val = X[xIndex];
        }
        Xcol[gid] = val;
    }

    kernel void col2im_1d_causal_f32(
        const device float* dXcol      [[buffer(0)]],
        device float*       dX         [[buffer(1)]],
        constant int&       B          [[buffer(2)]],
        constant int&       L          [[buffer(3)]],
        constant int&       Cin        [[buffer(4)]],
        constant int&       K          [[buffer(5)]],
        constant int&       dilation   [[buffer(6)]],
        uint                gid        [[thread_position_in_grid]])
    {
        // Each thread computes one output element dX[b,t,i]
        int outCount = B * L * Cin;
        if ((int)gid >= outCount) return;
        int rem = gid;
        int b = rem / (L * Cin);
        rem = rem % (L * Cin);
        int t = rem / Cin;
        int i = rem % Cin;
        float sum = 0.0f;
        // iterate over kernel positions k contributing to (b,t,i)
        for (int kIdx = 0; kIdx < K; ++kIdx) {
            int r = t + kIdx * dilation; // row in Xcol
            if (r >= L) continue;
            int row = b * L + r;
            int col = i * K + kIdx;
            int idx = row * (Cin * K) + col;
            sum += dXcol[idx];
        }
        dX[gid] = sum;
    }
    """
}
