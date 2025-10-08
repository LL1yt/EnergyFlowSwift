import Foundation
import Metal
import EFCore

public enum EmbeddingGPU {
    nonisolated(unsafe) private static var library: MTLLibrary? = nil
    nonisolated(unsafe) private static var pGather: MTLComputePipelineState? = nil

    private static func ensurePipeline(device: MTLDevice) throws {
        if let _ = pGather { return }
        if library == nil {
            library = try device.makeLibrary(source: metalSource, options: nil)
        }
        guard let lib = library else { throw MPSGError.contextUnavailable }
        if pGather == nil {
            let fn = lib.makeFunction(name: "embedding_gather_f32")!
            pGather = try device.makeComputePipelineState(function: fn)
        }
    }

    // ids: [[Int]] -> out [B,L,D]; weight: [V,D]
    public static func forward(ids: [[Int]], weight: Tensor) -> Tensor {
        let B = ids.count
        let L = ids.first?.count ?? 0
        precondition(ids.allSatisfy { $0.count == L }, "EmbeddingGPU.forward: ragged input")
        precondition(weight.shape.count == 2, "EmbeddingGPU.forward: weight must be [V,D]")
        let V = weight.shape[0]
        let D = weight.shape[1]
        if B == 0 || L == 0 { return Tensor.zeros([B, L, D]) }
        let ctx = MPSGContext.shared
        do { try ensurePipeline(device: ctx.device) } catch { fatalError("EmbeddingGPU: pipeline error: \(error)") }
        guard let p = pGather, let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("EmbeddingGPU: command buffer failed") }
        cmd.label = "EmbeddingGPU.forward"
        // Buffers
        let idsCount = B * L
        var idsFlat = [Int32](repeating: 0, count: idsCount)
        for b in 0..<B { for t in 0..<L { idsFlat[b * L + t] = Int32(ids[b][t]) } }
        let idsBuf = BufferPool.buffer(device: ctx.device, length: idsCount * MemoryLayout<Int32>.size, label: "Emb.ids")
        idsFlat.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(idsBuf.contents(), base, idsCount * MemoryLayout<Int32>.size) } }
        let wCount = V * D
        let wBuf = BufferPool.buffer(device: ctx.device, length: wCount * MemoryLayout<Float>.size, label: "Emb.w")
        weight.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(wBuf.contents(), base, wCount * MemoryLayout<Float>.size) } }
        let outCount = B * L * D
        let outBuf = BufferPool.buffer(device: ctx.device, length: outCount * MemoryLayout<Float>.size, label: "Emb.out")
        memset(outBuf.contents(), 0, outCount * MemoryLayout<Float>.size)
        // Encode
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("EmbeddingGPU: encoder failed") }
        enc.label = "Emb.gather"
        enc.setComputePipelineState(p)
        enc.setBuffer(idsBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var vB = Int32(B), vL = Int32(L), vV = Int32(V), vD = Int32(D)
        enc.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 4)
        enc.setBytes(&vV, length: MemoryLayout<Int32>.size, index: 5)
        enc.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 6)
        let threads = outCount
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (threads + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        // Read back
        var outHost = [Float](repeating: 0, count: outCount)
        memcpy(&outHost, outBuf.contents(), outCount * MemoryLayout<Float>.size)
        return Tensor(shape: [B, L, D], data: outHost)
    }

    private static let metalSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    // Gather embeddings: out[b,l,d] = W[ ids[b,l], d ] with out-of-range clamped to 0
    kernel void embedding_gather_f32(
        const device int*    ids   [[buffer(0)]],
        const device float*  W     [[buffer(1)]],
        device float*        out   [[buffer(2)]],
        constant int&        B     [[buffer(3)]],
        constant int&        L     [[buffer(4)]],
        constant int&        V     [[buffer(5)]],
        constant int&        D     [[buffer(6)]],
        uint gid [[thread_position_in_grid]])
    {
        int N = B * L * D;
        if ((int)gid >= N) return;
        int rem = (int)gid;
        int b = rem / (L * D);
        rem = rem % (L * D);
        int t = rem / D;
        int d = rem % D;
        int token = ids[b * L + t];
        if (token < 0 || token >= V) token = 0;
        out[gid] = W[token * D + d];
    }
    """
}