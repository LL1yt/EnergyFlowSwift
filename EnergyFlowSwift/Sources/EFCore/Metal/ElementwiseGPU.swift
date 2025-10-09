import Foundation
import Metal

public enum ElementwiseGPU {
    nonisolated(unsafe) private static var library: MTLLibrary? = nil
    nonisolated(unsafe) private static var pMask: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pMean: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pMeanBwd: MTLComputePipelineState? = nil
    
    private static func ensurePipelines(device: MTLDevice) throws {
        if let _ = pMask, let _ = pMean, let _ = pMeanBwd { return }
        if library == nil {
            library = try device.makeLibrary(source: ElementwiseMetalLibrary.source, options: nil)
        }
        guard let lib = library else { throw MPSGError.contextUnavailable }
        if pMask == nil {
            let fn = lib.makeFunction(name: "mask_zero_f32")!
            pMask = try device.makeComputePipelineState(function: fn)
        }
        if pMean == nil {
            let fn = lib.makeFunction(name: "masked_mean_f32")!
            pMean = try device.makeComputePipelineState(function: fn)
        }
        if pMeanBwd == nil {
            let fn = lib.makeFunction(name: "masked_mean_bwd_f32")!
            pMeanBwd = try device.makeComputePipelineState(function: fn)
        }
    }
    
    // y = y + x (elementwise), both Float32
    public static func residualAdd(y: Tensor, x: Tensor) -> Tensor {
        do {
            return try GPU.blocking(label: "ElementwiseGPU.residualAdd") { actor in
                try await actor.residualAdd(y: y, x: x)
            }
        } catch {
            fatalError("ElementwiseGPU.residualAdd failed: \(error)")
        }
    }
    
    // Add broadcast: y[b,t,d] += add[b,d] for all t in 0..L-1
    public static func addBroadcast2DInto3D(y: Tensor, addBD: Tensor, L: Int) -> Tensor {
        precondition(y.shape.count == 3, "addBroadcast2DInto3D expects y [B,L,D]")
        let B = y.shape[0], D = y.shape[2]
        precondition(addBD.shape == [B, D], "addBroadcast2DInto3D expects add [B,D]")
        let N = B * L * D
        if N == 0 { return y }
        let ctx = MPSGContext.shared
        do { try ensurePipelines(device: ctx.device) } catch { fatalError("ElementwiseGPU: pipeline error: \(error)") }
        guard let lib = library else { fatalError("ElementwiseGPU: library not available") }
        // Build pipeline on demand for add_broadcast_f32
        var pAddBC: MTLComputePipelineState
        if let existing = lib.makeFunction(name: "add_broadcast_2d_into_3d_f32"), let pipe = try? ctx.device.makeComputePipelineState(function: existing) {
            pAddBC = pipe
        } else {
            // Recreate library and pipeline in case not compiled yet
            do { library = try ctx.device.makeLibrary(source: ElementwiseMetalLibrary.source, options: nil) } catch { fatalError("ElementwiseGPU: makeLibrary failed: \(error)") }
            guard let fn = library?.makeFunction(name: "add_broadcast_2d_into_3d_f32") else { fatalError("ElementwiseGPU: kernel not found") }
            pAddBC = try! ctx.device.makeComputePipelineState(function: fn)
        }
        guard let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("ElementwiseGPU: command buffer failed") }
        cmd.label = "ElementwiseGPU.addBroadcast2DInto3D"
        let elem = MemoryLayout<Float>.size
        let yCount = N
        let addCount = B * D
        let yBuf = BufferPool.buffer(device: ctx.device, length: yCount * elem, label: "Elem.addbc.y")
        let aBuf = BufferPool.buffer(device: ctx.device, length: addCount * elem, label: "Elem.addbc.a")
        y.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(yBuf.contents(), base, yCount * elem) } }
        addBD.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(aBuf.contents(), base, addCount * elem) } }
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("ElementwiseGPU: encoder failed") }
        enc.label = "Elem.addbc.enc"
        enc.setComputePipelineState(pAddBC)
        enc.setBuffer(yBuf, offset: 0, index: 0)
        enc.setBuffer(aBuf, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vD = Int32(D)
        enc.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        enc.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 4)
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (yCount + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        var yOut = [Float](repeating: 0, count: yCount)
        memcpy(&yOut, yBuf.contents(), yCount * elem)
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
    
    // Masked mean over sequence dimension: x [B,L,H], mask [[B][L]] -> [B,H]
    public static func maskedMean(x: Tensor, mask: [[Int]]) -> Tensor {
        precondition(x.shape.count == 3, "maskedMean expects x [B,L,H]")
        let B = x.shape[0], L = x.shape[1], H = x.shape[2]
        precondition(mask.count == B && mask.allSatisfy { $0.count == L }, "mask shape mismatch")
        let ctx = MPSGContext.shared
        do { try ensurePipelines(device: ctx.device) } catch { fatalError("ElementwiseGPU: pipeline error") }
        guard let pMean = pMean, let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("ElementwiseGPU: command buffer failed") }
        cmd.label = "ElementwiseGPU.maskedMean"
        // Buffers
        let elem = MemoryLayout<Float>.size
        let xCount = B * L * H
        let yCount = B * H
        let xBuf = BufferPool.buffer(device: ctx.device, length: xCount * elem, label: "Elem.mean.x")
        let yBuf = BufferPool.buffer(device: ctx.device, length: yCount * elem, label: "Elem.mean.y")
        var maskFlat = [Int32](repeating: 0, count: B * L)
        for b in 0..<B { for t in 0..<L { maskFlat[b * L + t] = Int32(mask[b][t]) } }
        let mBuf = BufferPool.buffer(device: ctx.device, length: B * L * MemoryLayout<Int32>.size, label: "Elem.mean.m")
        x.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(xBuf.contents(), base, xCount * elem) } }
        memset(yBuf.contents(), 0, yCount * elem)
        maskFlat.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(mBuf.contents(), base, B * L * MemoryLayout<Int32>.size) } }
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("ElementwiseGPU: encoder failed") }
        enc.label = "Elem.mean.enc"
        enc.setComputePipelineState(pMean)
        enc.setBuffer(xBuf, offset: 0, index: 0)
        enc.setBuffer(mBuf, offset: 0, index: 1)
        enc.setBuffer(yBuf, offset: 0, index: 2)
        var vB = Int32(B), vL = Int32(L), vH = Int32(H), vEps: Float = 1e-9
        enc.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 4)
        enc.setBytes(&vH, length: MemoryLayout<Int32>.size, index: 5)
        enc.setBytes(&vEps, length: MemoryLayout<Float>.size, index: 6)
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (yCount + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        var yHost = [Float](repeating: 0, count: yCount)
        memcpy(&yHost, yBuf.contents(), yCount * elem)
        return Tensor(shape: [B, H], data: yHost)
    }
    
    // Backward for masked mean: given dPooled [B,H] and mask [B][L],
    // produce dEnc [B,L,H] with dEnc[b,t,h] = (mask[b,t]/denom_b) * dPooled[b,h]
    public static func maskedMeanBackward(dPooled: Tensor, mask: [[Int]], seqLen: Int) -> Tensor {
        precondition(dPooled.shape.count == 2, "maskedMeanBackward expects [B,H]")
        let B = dPooled.shape[0], H = dPooled.shape[1]
        let L = seqLen
        precondition(mask.count == B && mask.allSatisfy { $0.count == L }, "mask shape mismatch")
        let ctx = MPSGContext.shared
        do { try ensurePipelines(device: ctx.device) } catch { fatalError("ElementwiseGPU: pipeline error") }
        guard let pB = pMeanBwd, let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("ElementwiseGPU: command buffer failed") }
        cmd.label = "ElementwiseGPU.maskedMeanBackward"
        // Buffers
        let elem = MemoryLayout<Float>.size
        let dyCount = B * H
        let dxCount = B * L * H
        let dyBuf = BufferPool.buffer(device: ctx.device, length: dyCount * elem, label: "Elem.meanbwd.dy")
        let dxBuf = BufferPool.buffer(device: ctx.device, length: dxCount * elem, label: "Elem.meanbwd.dx")
        var maskFlat = [Int32](repeating: 0, count: B * L)
        for b in 0..<B { for t in 0..<L { maskFlat[b * L + t] = Int32(mask[b][t]) } }
        let mBuf = BufferPool.buffer(device: ctx.device, length: B * L * MemoryLayout<Int32>.size, label: "Elem.meanbwd.m")
        dPooled.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(dyBuf.contents(), base, dyCount * elem) } }
        memset(dxBuf.contents(), 0, dxCount * elem)
        maskFlat.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(mBuf.contents(), base, B * L * MemoryLayout<Int32>.size) } }
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("ElementwiseGPU: encoder failed") }
        enc.label = "Elem.meanbwd.enc"
        enc.setComputePipelineState(pB)
        enc.setBuffer(dyBuf, offset: 0, index: 0)
        enc.setBuffer(mBuf, offset: 0, index: 1)
        enc.setBuffer(dxBuf, offset: 0, index: 2)
        var vB = Int32(B), vL = Int32(L), vH = Int32(H), vEps: Float = 1e-9
        enc.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 4)
        enc.setBytes(&vH, length: MemoryLayout<Int32>.size, index: 5)
        enc.setBytes(&vEps, length: MemoryLayout<Float>.size, index: 6)
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (dxCount + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        var dxHost = [Float](repeating: 0, count: dxCount)
        memcpy(&dxHost, dxBuf.contents(), dxCount * elem)
        return Tensor(shape: [B, L, H], data: dxHost)
    }
}
