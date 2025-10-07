import Foundation
import Metal

public enum GELUGPU {
    nonisolated(unsafe) private static var library: MTLLibrary? = nil
    nonisolated(unsafe) private static var pGelu: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pGeluBwd: MTLComputePipelineState? = nil

    private static func ensurePipeline(device: MTLDevice) throws {
        if let _ = pGelu, let _ = pGeluBwd { return }
        if library == nil {
            library = try device.makeLibrary(source: metalSource, options: nil)
        }
        guard let lib = library else { throw MPSGError.contextUnavailable }
        if pGelu == nil {
            let fn = lib.makeFunction(name: "gelu_tanh_fp16")!
            pGelu = try device.makeComputePipelineState(function: fn)
        }
        if pGeluBwd == nil {
            let fnb = lib.makeFunction(name: "gelu_tanh_bwd_fp16")!
            pGeluBwd = try device.makeComputePipelineState(function: fnb)
        }
    }

    // Forward GELU with FP16 IO and tanh approximation. Works on any shaped tensor (flattened on device).
    public static func forward(_ x: Tensor) -> Tensor {
        let count = x.data.count
        if count == 0 { return x }
        let ctx = MPSGContext.shared
        do { try ensurePipeline(device: ctx.device) } catch { fatalError("GELUGPU: pipeline error: \(error)") }
        guard let pGelu = pGelu, let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("GELUGPU: command buffer failed") }
        cmd.label = "GELUGPU.forward"
        let elemH = MemoryLayout<Float16>.size
        // Buffers
        let xBuf = BufferPool.buffer(device: ctx.device, length: count * elemH, label: "GELU.x.fp16")
        let yBuf = BufferPool.buffer(device: ctx.device, length: count * elemH, label: "GELU.y.fp16")
        // Pack input as FP16
        var xHalf = [Float16](repeating: 0, count: count)
        for i in 0..<count { xHalf[i] = Float16(x.data[i]) }
        xHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(xBuf.contents(), base, count * elemH) } }
        memset(yBuf.contents(), 0, count * elemH)
        // Encode kernel
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("GELUGPU: encoder failed") }
        enc.label = "GELU.tanh"
        enc.setComputePipelineState(pGelu)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(yBuf, offset: 0, index: 1)
        var n = Int32(count)
        enc.setBytes(&n, length: MemoryLayout<Int32>.size, index: 2)
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        // Read back y (FP16) into Float
        var yHalf = [Float16](repeating: 0, count: count)
        memcpy(&yHalf, yBuf.contents(), count * elemH)
        var yHost = [Float](repeating: 0, count: count)
        for i in 0..<count { yHost[i] = Float(yHalf[i]) }
        return Tensor(shape: x.shape, data: yHost)
    }

    // Backward: given x and upstream dY, compute dX using derivative of tanh-approx GELU
    public static func backward(x: Tensor, dY: Tensor) -> Tensor {
        precondition(x.shape == dY.shape, "GELUGPU.backward shape mismatch")
        let count = x.data.count
        if count == 0 { return dY }
        let ctx = MPSGContext.shared
        do { try ensurePipeline(device: ctx.device) } catch { fatalError("GELUGPU: pipeline error: \(error)") }
        guard let pB = pGeluBwd, let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("GELUGPU: command buffer failed") }
        cmd.label = "GELUGPU.backward"
        let elemH = MemoryLayout<Float16>.size
        let xBuf = BufferPool.buffer(device: ctx.device, length: count * elemH, label: "GELU.bwd.x.fp16")
        let dyBuf = BufferPool.buffer(device: ctx.device, length: count * elemH, label: "GELU.bwd.dy.fp16")
        let dxBuf = BufferPool.buffer(device: ctx.device, length: count * elemH, label: "GELU.bwd.dx.fp16")
        // Pack x and dY as FP16
        var xHalf = [Float16](repeating: 0, count: count)
        var dyHalf = [Float16](repeating: 0, count: count)
        for i in 0..<count { xHalf[i] = Float16(x.data[i]); dyHalf[i] = Float16(dY.data[i]) }
        xHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(xBuf.contents(), base, count * elemH) } }
        dyHalf.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(dyBuf.contents(), base, count * elemH) } }
        memset(dxBuf.contents(), 0, count * elemH)
        // Encode kernel
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("GELUGPU: encoder failed") }
        enc.label = "GELU.tanh.bwd"
        enc.setComputePipelineState(pB)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(dyBuf, offset: 0, index: 1)
        enc.setBuffer(dxBuf, offset: 0, index: 2)
        var n = Int32(count)
        enc.setBytes(&n, length: MemoryLayout<Int32>.size, index: 3)
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        // Read back dx (FP16) into Float
        var dxHalf = [Float16](repeating: 0, count: count)
        memcpy(&dxHalf, dxBuf.contents(), count * elemH)
        var dxHost = [Float](repeating: 0, count: count)
        for i in 0..<count { dxHost[i] = Float(dxHalf[i]) }
        return Tensor(shape: x.shape, data: dxHost)
    }

    // Metal kernels: tanh-approx GELU (fwd/bwd) with FP16 IO
    private static let metalSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void gelu_tanh_fp16(
        const device half*  x   [[buffer(0)]],
        device half*        y   [[buffer(1)]],
        constant int&       N   [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        float v = (float)x[gid];
        // tanh approximation: 0.5 * v * (1 + tanh( sqrt(2/pi) * (v + 0.044715 v^3) ))
        const float c = 0.7978845608028654f; // sqrt(2/pi)
        float v3 = v * v * v;
        float u = c * (v + 0.044715f * v3);
        float t = tanh(u);
        float out = 0.5f * v * (1.0f + t);
        y[gid] = (half)out;
    }

    kernel void gelu_tanh_bwd_fp16(
        const device half*  x    [[buffer(0)]],
        const device half*  dy   [[buffer(1)]],
        device half*        dx   [[buffer(2)]],
        constant int&       N    [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        float v = (float)x[gid];
        float gy = (float)dy[gid];
        const float c = 0.7978845608028654f; // sqrt(2/pi)
        const float a = 0.044715f;
        float v2 = v * v;
        float v3 = v2 * v;
        float u = c * (v + a * v3);
        float th = tanh(u);
        float sech2 = 1.0f - th * th; // sech^2(u)
        float du_dx = c * (1.0f + 3.0f * a * v2);
        float dgelu = 0.5f * (1.0f + th) + 0.5f * v * sech2 * du_dx;
        dx[gid] = (half)(gy * dgelu);
    }
    """
}