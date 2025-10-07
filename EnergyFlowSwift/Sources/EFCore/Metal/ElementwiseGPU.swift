import Foundation
import Metal

public enum ElementwiseGPU {
    nonisolated(unsafe) private static var library: MTLLibrary? = nil
    nonisolated(unsafe) private static var pAdd: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pMask: MTLComputePipelineState? = nil

    private static func ensurePipelines(device: MTLDevice) throws {
        if let _ = pAdd, let _ = pMask { return }
        if library == nil {
            library = try device.makeLibrary(source: metalSource, options: nil)
        }
        guard let lib = library else { throw MPSGError.contextUnavailable }
        if pAdd == nil {
            let fn = lib.makeFunction(name: "residual_add_f32")!
            pAdd = try device.makeComputePipelineState(function: fn)
        }
        if pMask == nil {
            let fn = lib.makeFunction(name: "mask_zero_f32")!
            pMask = try device.makeComputePipelineState(function: fn)
        }
    }

    // y = y + x (elementwise), both Float32
    public static func residualAdd(y: Tensor, x: Tensor) -> Tensor {
        precondition(y.shape == x.shape, "residualAdd shape mismatch")
        let N = y.count
        if N == 0 { return y }
        let ctx = MPSGContext.shared
        do { try ensurePipelines(device: ctx.device) } catch { fatalError("ElementwiseGPU: pipeline error: \(error)") }
        guard let pAdd = pAdd, let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("ElementwiseGPU: command buffer failed") }
        cmd.label = "ElementwiseGPU.residualAdd"
        let elem = MemoryLayout<Float>.size
        let xBuf = BufferPool.buffer(device: ctx.device, length: N * elem, label: "Elem.add.x")
        let yBuf = BufferPool.buffer(device: ctx.device, length: N * elem, label: "Elem.add.y")
        x.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(xBuf.contents(), base, N * elem) } }
        y.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(yBuf.contents(), base, N * elem) } }
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("ElementwiseGPU: encoder failed") }
        enc.label = "Elem.add.enc"
        enc.setComputePipelineState(pAdd)
        enc.setBuffer(yBuf, offset: 0, index: 0)
        enc.setBuffer(xBuf, offset: 0, index: 1)
        var n = Int32(N)
        enc.setBytes(&n, length: MemoryLayout<Int32>.size, index: 2)
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        var yOut = [Float](repeating: 0, count: N)
        memcpy(&yOut, yBuf.contents(), N * elem)
        return Tensor(shape: y.shape, data: yOut)
    }

    // Zero masked positions: y[b,t,:] = 0 if mask[b][t] == 0
    public static func maskZero(y: Tensor, mask: [[Int]]) -> Tensor {
        precondition(y.shape.count == 3, "maskZero expects y [B,L,D]")
        let B = y.shape[0], L = y.shape[1], D = y.shape[2]
        precondition(mask.count == B && mask.allSatisfy { $0.count == L }, "mask shape mismatch")
        let N = B * L * D
        if N == 0 { return y }
        let ctx = MPSGContext.shared
        do { try ensurePipelines(device: ctx.device) } catch { fatalError("ElementwiseGPU: pipeline error: \(error)") }
        guard let pMask = pMask, let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("ElementwiseGPU: command buffer failed") }
        cmd.label = "ElementwiseGPU.maskZero"
        let elem = MemoryLayout<Float>.size
        let yBuf = BufferPool.buffer(device: ctx.device, length: N * elem, label: "Elem.mask.y")
        // Flatten mask to Int32 [B*L]
        var maskFlat = [Int32](repeating: 0, count: B * L)
        for b in 0..<B { for t in 0..<L { maskFlat[b * L + t] = Int32(mask[b][t]) } }
        let mBuf = BufferPool.buffer(device: ctx.device, length: B * L * MemoryLayout<Int32>.size, label: "Elem.mask.m")
        y.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(yBuf.contents(), base, N * elem) } }
        maskFlat.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(mBuf.contents(), base, B * L * MemoryLayout<Int32>.size) } }
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("ElementwiseGPU: encoder failed") }
        enc.label = "Elem.mask.enc"
        enc.setComputePipelineState(pMask)
        enc.setBuffer(yBuf, offset: 0, index: 0)
        enc.setBuffer(mBuf, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vD = Int32(D)
        enc.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        enc.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 4)
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        var yOut = [Float](repeating: 0, count: N)
        memcpy(&yOut, yBuf.contents(), N * elem)
        return Tensor(shape: y.shape, data: yOut)
    }

    private static let metalSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void residual_add_f32(
        device float*       y   [[buffer(0)]],
        const device float* x   [[buffer(1)]],
        constant int&       N   [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        y[gid] += x[gid];
    }

    kernel void mask_zero_f32(
        device float*        y    [[buffer(0)]],
        const device int*    mask [[buffer(1)]],
        constant int&        B    [[buffer(2)]],
        constant int&        L    [[buffer(3)]],
        constant int&        D    [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        int N = B * L * D;
        if ((int)gid >= N) return;
        int rem = (int)gid;
        int b = rem / (L * D);
        rem = rem % (L * D);
        int t = rem / D;
        int m = mask[b * L + t];
        if (m == 0) {
            y[gid] = 0.0f;
        }
    }
    """
}