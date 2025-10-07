import Foundation
import Metal

public enum LayerNormGPU {
    nonisolated(unsafe) private static var library: MTLLibrary? = nil
    nonisolated(unsafe) private static var pStats: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pNorm: MTLComputePipelineState? = nil
    // Backward pipelines
    nonisolated(unsafe) private static var pBwdRow: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pBwdDX: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pBwdDG: MTLComputePipelineState? = nil

    private static func ensurePipelines(device: MTLDevice) throws {
        if let _ = pStats, let _ = pNorm, let _ = pBwdRow, let _ = pBwdDX, let _ = pBwdDG { return }
        if library == nil {
            library = try device.makeLibrary(source: metalSource, options: nil)
        }
        guard let lib = library else { throw MPSGError.contextUnavailable }
        if pStats == nil {
            let fn = lib.makeFunction(name: "ln_compute_stats_fp16")!
            pStats = try device.makeComputePipelineState(function: fn)
        }
        if pNorm == nil {
            let fn = lib.makeFunction(name: "ln_normalize_affine_fp16")!
            pNorm = try device.makeComputePipelineState(function: fn)
        }
        if pBwdRow == nil {
            let fn = lib.makeFunction(name: "ln_bwd_row_sums_fp16")!
            pBwdRow = try device.makeComputePipelineState(function: fn)
        }
        if pBwdDX == nil {
            let fn = lib.makeFunction(name: "ln_bwd_dx_fp16")!
            pBwdDX = try device.makeComputePipelineState(function: fn)
        }
        if pBwdDG == nil {
            let fn = lib.makeFunction(name: "ln_bwd_dgamma_dbeta_f32")!
            pBwdDG = try device.makeComputePipelineState(function: fn)
        }
    }

    public static func forward(x: Tensor, gamma: Tensor, beta: Tensor, eps: Float) -> Tensor {
        precondition(x.shape.count == 2, "LayerNormGPU.forward expects [N,D]")
        precondition(gamma.shape == [x.shape[1]] && beta.shape == [x.shape[1]], "gamma/beta shape mismatch")
        let N = x.shape[0]
        let D = x.shape[1]
        let ctx = MPSGContext.shared
        do { try ensurePipelines(device: ctx.device) } catch { fatalError("LayerNormGPU: pipeline error: \(error)") }
        guard let pStats = pStats, let pNorm = pNorm else { fatalError("LayerNormGPU: pipelines unavailable") }
        guard let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("LayerNormGPU: command buffer failed") }
        cmd.label = "LayerNormGPU.forward"

        // Buffers
        let elemH = MemoryLayout<Float16>.size
        let xBuf = BufferPool.buffer(device: ctx.device, length: N * D * elemH, label: "LN.x.fp16")
        let yBuf = BufferPool.buffer(device: ctx.device, length: N * D * elemH, label: "LN.y.fp16")
        let meanBuf = BufferPool.buffer(device: ctx.device, length: N * MemoryLayout<Float>.size, label: "LN.mean.f32")
        let invStdBuf = BufferPool.buffer(device: ctx.device, length: N * MemoryLayout<Float>.size, label: "LN.invstd.f32")
        let gammaBuf = BufferPool.buffer(device: ctx.device, length: D * MemoryLayout<Float>.size, label: "LN.gamma.f32")
        let betaBuf = BufferPool.buffer(device: ctx.device, length: D * MemoryLayout<Float>.size, label: "LN.beta.f32")

        // Pack X as FP16 (contiguous)
        var xHalf = [Float16](repeating: 0, count: N * D)
        for i in 0..<(N*D) { xHalf[i] = Float16(x.data[i]) }
        xHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(xBuf.contents(), base, N * D * elemH) } }
        // Copy gamma/beta (float)
        gamma.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(gammaBuf.contents(), base, D * MemoryLayout<Float>.size) } }
        beta.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(betaBuf.contents(), base, D * MemoryLayout<Float>.size) } }
        memset(meanBuf.contents(), 0, N * MemoryLayout<Float>.size)
        memset(invStdBuf.contents(), 0, N * MemoryLayout<Float>.size)
        memset(yBuf.contents(), 0, N * D * elemH)

        // Encode stats kernel: one thread per row
        guard let enc1 = cmd.makeComputeCommandEncoder() else { fatalError("LayerNormGPU: encoder 1 failed") }
        enc1.label = "LN.stats"
        enc1.setComputePipelineState(pStats)
        enc1.setBuffer(xBuf, offset: 0, index: 0)
        enc1.setBuffer(meanBuf, offset: 0, index: 1)
        enc1.setBuffer(invStdBuf, offset: 0, index: 2)
        var vN = Int32(N), vD = Int32(D), vEps = eps
        enc1.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 3)
        enc1.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 4)
        enc1.setBytes(&vEps, length: MemoryLayout<Float>.size, index: 5)
        let tpt1 = MTLSize(width: 256, height: 1, depth: 1)
        let tg1 = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        enc1.dispatchThreadgroups(tg1, threadsPerThreadgroup: tpt1)
        enc1.endEncoding()

        // Encode normalize+affine kernel: one thread per row
        guard let enc2 = cmd.makeComputeCommandEncoder() else { fatalError("LayerNormGPU: encoder 2 failed") }
        enc2.label = "LN.norm"
        enc2.setComputePipelineState(pNorm)
        enc2.setBuffer(xBuf, offset: 0, index: 0)
        enc2.setBuffer(yBuf, offset: 0, index: 1)
        enc2.setBuffer(gammaBuf, offset: 0, index: 2)
        enc2.setBuffer(betaBuf, offset: 0, index: 3)
        enc2.setBuffer(meanBuf, offset: 0, index: 4)
        enc2.setBuffer(invStdBuf, offset: 0, index: 5)
        enc2.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 6)
        enc2.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 7)
        let tpt2 = MTLSize(width: 256, height: 1, depth: 1)
        let tg2 = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        enc2.dispatchThreadgroups(tg2, threadsPerThreadgroup: tpt2)
        enc2.endEncoding()

        cmd.commit(); cmd.waitUntilCompleted()

        // Read back y (FP16) and convert to Float
        var yHalf = [Float16](repeating: 0, count: N * D)
        memcpy(&yHalf, yBuf.contents(), N * D * elemH)
        var yHost = [Float](repeating: 0, count: N * D)
        for i in 0..<(N*D) { yHost[i] = Float(yHalf[i]) }
        return Tensor(shape: [N, D], data: yHost)
    }

    // Backward for LayerNorm (row-wise): given x [N,D], upstream g [N,D], gamma [D]
    // returns (dx [N,D], dGamma [D], dBeta [D])
    public static func backward(x: Tensor, g: Tensor, gamma: Tensor, eps: Float = 1e-5) -> (Tensor, Tensor, Tensor) {
        precondition(x.shape.count == 2 && g.shape == x.shape, "LayerNormGPU.backward expects x,g [N,D]")
        precondition(gamma.shape.count == 1 && gamma.shape[0] == x.shape[1], "gamma shape mismatch")
        let N = x.shape[0]
        let D = x.shape[1]
        let ctx = MPSGContext.shared
        do { try ensurePipelines(device: ctx.device) } catch { fatalError("LayerNormGPU: pipeline error: \(error)") }
        guard let pBwdRow = pBwdRow, let pBwdDX = pBwdDX, let pBwdDG = pBwdDG else { fatalError("LayerNormGPU: backward pipelines unavailable") }
        guard let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("LayerNormGPU: command buffer failed") }
        cmd.label = "LayerNormGPU.backward"
        // Buffers
        let elemH = MemoryLayout<Float16>.size
        let xBuf = BufferPool.buffer(device: ctx.device, length: N * D * elemH, label: "LN.bwd.x.fp16")
        let gBuf = BufferPool.buffer(device: ctx.device, length: N * D * elemH, label: "LN.bwd.g.fp16")
        let gammaBuf = BufferPool.buffer(device: ctx.device, length: D * MemoryLayout<Float>.size, label: "LN.bwd.gamma.f32")
        let meanBuf = BufferPool.buffer(device: ctx.device, length: N * MemoryLayout<Float>.size, label: "LN.bwd.mean.f32")
        let invStdBuf = BufferPool.buffer(device: ctx.device, length: N * MemoryLayout<Float>.size, label: "LN.bwd.invstd.f32")
        let sumGBuf = BufferPool.buffer(device: ctx.device, length: N * MemoryLayout<Float>.size, label: "LN.bwd.sumG.f32")
        let sumGXhatBuf = BufferPool.buffer(device: ctx.device, length: N * MemoryLayout<Float>.size, label: "LN.bwd.sumGXhat.f32")
        let dxBuf = BufferPool.buffer(device: ctx.device, length: N * D * elemH, label: "LN.bwd.dx.fp16")
        let dGammaBuf = BufferPool.buffer(device: ctx.device, length: D * MemoryLayout<Float>.size, label: "LN.bwd.dgamma.f32")
        let dBetaBuf = BufferPool.buffer(device: ctx.device, length: D * MemoryLayout<Float>.size, label: "LN.bwd.dbeta.f32")
        // Pack x,g as FP16; gamma as float
        var xHalf = [Float16](repeating: 0, count: N * D)
        var gHalf = [Float16](repeating: 0, count: N * D)
        for i in 0..<(N*D) { xHalf[i] = Float16(x.data[i]); gHalf[i] = Float16(g.data[i]) }
        xHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(xBuf.contents(), base, N * D * elemH) } }
        gHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(gBuf.contents(), base, N * D * elemH) } }
        gamma.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(gammaBuf.contents(), base, D * MemoryLayout<Float>.size) } }
        memset(meanBuf.contents(), 0, N * MemoryLayout<Float>.size)
        memset(invStdBuf.contents(), 0, N * MemoryLayout<Float>.size)
        memset(sumGBuf.contents(), 0, N * MemoryLayout<Float>.size)
        memset(sumGXhatBuf.contents(), 0, N * MemoryLayout<Float>.size)
        memset(dxBuf.contents(), 0, N * D * elemH)
        memset(dGammaBuf.contents(), 0, D * MemoryLayout<Float>.size)
        memset(dBetaBuf.contents(), 0, D * MemoryLayout<Float>.size)
        // Kernel 1: row sums and stats
        guard let enc1 = cmd.makeComputeCommandEncoder() else { fatalError("LayerNormGPU: encoder bwd row failed") }
        enc1.label = "LN.bwd.row"
        enc1.setComputePipelineState(pBwdRow)
        enc1.setBuffer(xBuf, offset: 0, index: 0)
        enc1.setBuffer(gBuf, offset: 0, index: 1)
        enc1.setBuffer(gammaBuf, offset: 0, index: 2)
        enc1.setBuffer(meanBuf, offset: 0, index: 3)
        enc1.setBuffer(invStdBuf, offset: 0, index: 4)
        enc1.setBuffer(sumGBuf, offset: 0, index: 5)
        enc1.setBuffer(sumGXhatBuf, offset: 0, index: 6)
        var vN = Int32(N), vD = Int32(D), vEps = eps
        enc1.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 7)
        enc1.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 8)
        enc1.setBytes(&vEps, length: MemoryLayout<Float>.size, index: 9)
        let tpt1 = MTLSize(width: 256, height: 1, depth: 1)
        let tg1 = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        enc1.dispatchThreadgroups(tg1, threadsPerThreadgroup: tpt1)
        enc1.endEncoding()
        // Kernel 2: dx per row
        guard let enc2 = cmd.makeComputeCommandEncoder() else { fatalError("LayerNormGPU: encoder bwd dx failed") }
        enc2.label = "LN.bwd.dx"
        enc2.setComputePipelineState(pBwdDX)
        enc2.setBuffer(xBuf, offset: 0, index: 0)
        enc2.setBuffer(gBuf, offset: 0, index: 1)
        enc2.setBuffer(gammaBuf, offset: 0, index: 2)
        enc2.setBuffer(meanBuf, offset: 0, index: 3)
        enc2.setBuffer(invStdBuf, offset: 0, index: 4)
        enc2.setBuffer(sumGBuf, offset: 0, index: 5)
        enc2.setBuffer(sumGXhatBuf, offset: 0, index: 6)
        enc2.setBuffer(dxBuf, offset: 0, index: 7)
        enc2.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 8)
        enc2.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 9)
        let tpt2 = MTLSize(width: 256, height: 1, depth: 1)
        let tg2 = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        enc2.dispatchThreadgroups(tg2, threadsPerThreadgroup: tpt2)
        enc2.endEncoding()
        // Kernel 3: dGamma/dBeta across rows
        guard let enc3 = cmd.makeComputeCommandEncoder() else { fatalError("LayerNormGPU: encoder bwd dgb failed") }
        enc3.label = "LN.bwd.dgb"
        enc3.setComputePipelineState(pBwdDG)
        enc3.setBuffer(xBuf, offset: 0, index: 0)
        enc3.setBuffer(gBuf, offset: 0, index: 1)
        enc3.setBuffer(meanBuf, offset: 0, index: 2)
        enc3.setBuffer(invStdBuf, offset: 0, index: 3)
        enc3.setBuffer(dGammaBuf, offset: 0, index: 4)
        enc3.setBuffer(dBetaBuf, offset: 0, index: 5)
        enc3.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 6)
        enc3.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 7)
        let tpt3 = MTLSize(width: 256, height: 1, depth: 1)
        let tg3 = MTLSize(width: (D + 255) / 256, height: 1, depth: 1)
        enc3.dispatchThreadgroups(tg3, threadsPerThreadgroup: tpt3)
        enc3.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        // Read back outputs
        var dxHalf = [Float16](repeating: 0, count: N * D)
        memcpy(&dxHalf, dxBuf.contents(), N * D * elemH)
        var dxHost = [Float](repeating: 0, count: N * D)
        for i in 0..<(N*D) { dxHost[i] = Float(dxHalf[i]) }
        var dGammaHost = [Float](repeating: 0, count: D)
        dGammaHost.withUnsafeMutableBytes { raw in if let base = raw.baseAddress { memcpy(base, dGammaBuf.contents(), D * MemoryLayout<Float>.size) } }
        var dBetaHost = [Float](repeating: 0, count: D)
        dBetaHost.withUnsafeMutableBytes { raw in if let base = raw.baseAddress { memcpy(base, dBetaBuf.contents(), D * MemoryLayout<Float>.size) } }
        return (Tensor(shape: [N, D], data: dxHost), Tensor(shape: [D], data: dGammaHost), Tensor(shape: [D], data: dBetaHost))
    }

    // Two-pass LayerNorm kernels with FP16 IO and FP32 accumulators
    private static let metalSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void ln_compute_stats_fp16(
        const device half*  x     [[buffer(0)]],
        device float*       mean  [[buffer(1)]],
        device float*       invst [[buffer(2)]],
        constant int&       N     [[buffer(3)]],
        constant int&       D     [[buffer(4)]],
        constant float&     eps   [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        int base = (int)gid * D;
        float s = 0.0f;
        float ss = 0.0f;
        for (int j = 0; j < D; ++j) {
            float v = (float)x[base + j];
            s  += v;
            ss += v * v;
        }
        float m = s / (float)D;
        float var = ss / (float)D - m * m;
        mean[gid] = m;
        invst[gid] = rsqrt(var + eps);
    }

    kernel void ln_normalize_affine_fp16(
        const device half*   x     [[buffer(0)]],
        device half*         y     [[buffer(1)]],
        const device float*  gamma [[buffer(2)]],
        const device float*  beta  [[buffer(3)]],
        const device float*  mean  [[buffer(4)]],
        const device float*  invst [[buffer(5)]],
        constant int&        N     [[buffer(6)]],
        constant int&        D     [[buffer(7)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        int base = (int)gid * D;
        float m = mean[gid];
        float is = invst[gid];
        for (int j = 0; j < D; ++j) {
            float vx = (float)x[base + j];
            float nrm = (vx - m) * is;
            float out = nrm * gamma[j] + beta[j];
            y[base + j] = (half)out;
        }
    }

    // Backward kernels
    // Kernel 1: per-row stats and sums
    kernel void ln_bwd_row_sums_fp16(
        const device half*   x      [[buffer(0)]],
        const device half*   g      [[buffer(1)]],
        const device float*  gamma  [[buffer(2)]],
        device float*        mean    [[buffer(3)]],
        device float*        invst   [[buffer(4)]],
        device float*        sumG    [[buffer(5)]],
        device float*        sumGXh  [[buffer(6)]],
        constant int&        N       [[buffer(7)]],
        constant int&        D       [[buffer(8)]],
        constant float&      eps     [[buffer(9)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        int base = (int)gid * D;
        // pass 1: mean/var
        float s = 0.0f;
        float ss = 0.0f;
        for (int j = 0; j < D; ++j) {
            float v = (float)x[base + j];
            s  += v;
            ss += v * v;
        }
        float m = s / (float)D;
        float var = ss / (float)D - m * m;
        float is = rsqrt(var + eps);
        mean[gid] = m;
        invst[gid] = is;
        // pass 2: sums for gy and gy*xhat
        float sg = 0.0f;
        float sgx = 0.0f;
        for (int j = 0; j < D; ++j) {
            float v = (float)x[base + j];
            float xh = (v - m) * is;
            float gy = (float)g[base + j] * gamma[j];
            sg += gy;
            sgx += gy * xh;
        }
        sumG[gid] = sg;
        sumGXh[gid] = sgx;
    }

    // Kernel 2: dx per row
    kernel void ln_bwd_dx_fp16(
        const device half*   x      [[buffer(0)]],
        const device half*   g      [[buffer(1)]],
        const device float*  gamma  [[buffer(2)]],
        const device float*  mean   [[buffer(3)]],
        const device float*  invst  [[buffer(4)]],
        const device float*  sumG   [[buffer(5)]],
        const device float*  sumGXh [[buffer(6)]],
        device half*         dx     [[buffer(7)]],
        constant int&        N      [[buffer(8)]],
        constant int&        D      [[buffer(9)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        int base = (int)gid * D;
        float m = mean[gid];
        float is = invst[gid];
        float sg = sumG[gid];
        float sgx = sumGXh[gid];
        float invD = 1.0f / (float)D;
        for (int j = 0; j < D; ++j) {
            float v = (float)x[base + j];
            float xh = (v - m) * is;
            float gy = (float)g[base + j] * gamma[j];
            float term = (float)D * gy - sg - xh * sgx;
            float dxi = invD * is * term;
            dx[base + j] = (half)dxi;
        }
    }

    // Kernel 3: dGamma/dBeta per feature (reduce across rows)
    kernel void ln_bwd_dgamma_dbeta_f32(
        const device half*   x      [[buffer(0)]],
        const device half*   g      [[buffer(1)]],
        const device float*  mean   [[buffer(2)]],
        const device float*  invst  [[buffer(3)]],
        device float*        dGamma [[buffer(4)]],
        device float*        dBeta  [[buffer(5)]],
        constant int&        N      [[buffer(6)]],
        constant int&        D      [[buffer(7)]],
        uint gid [[thread_position_in_grid]])
    {
        int j = (int)gid;
        if (j >= D) return;
        float dgb = 0.0f;
        float db = 0.0f;
        for (int n = 0; n < N; ++n) {
            int idx = n * D + j;
            float v = (float)x[idx];
            float m = mean[n];
            float is = invst[n];
            float xh = (v - m) * is;
            float gy = (float)g[idx];
            dgb += gy * xh;
            db += gy;
        }
        dGamma[j] = dgb;
        dBeta[j] = db;
    }
    """
}
