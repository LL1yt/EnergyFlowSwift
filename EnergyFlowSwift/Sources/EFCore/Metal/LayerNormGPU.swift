import Foundation
import Metal

public enum LayerNormGPU {
    nonisolated(unsafe) private static var library: MTLLibrary? = nil
    nonisolated(unsafe) private static var pStats: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pNorm: MTLComputePipelineState? = nil

    private static func ensurePipelines(device: MTLDevice) throws {
        if let _ = pStats, let _ = pNorm { return }
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
    """
}
