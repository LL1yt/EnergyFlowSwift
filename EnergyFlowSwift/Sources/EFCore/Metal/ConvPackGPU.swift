import Foundation
import Metal

public enum ConvPackGPU {
    nonisolated(unsafe) private static var library: MTLLibrary? = nil
    nonisolated(unsafe) private static var pPack: MTLComputePipelineState? = nil
    nonisolated(unsafe) private static var pUnpack: MTLComputePipelineState? = nil

    private static func ensure(device: MTLDevice) throws {
        if let _ = pPack, let _ = pUnpack { return }
        if library == nil {
            library = try device.makeLibrary(source: metalSource, options: nil)
        }
        guard let lib = library else { throw MPSGError.contextUnavailable }
        if pPack == nil {
            let fn = lib.makeFunction(name: "pack_w_to_col_f32")!
            pPack = try device.makeComputePipelineState(function: fn)
        }
        if pUnpack == nil {
            let fn = lib.makeFunction(name: "unpack_dwcol_f32")!
            pUnpack = try device.makeComputePipelineState(function: fn)
        }
    }

    // Pack W [Cout, Cin, K] -> Wcol [Cout, Cin*K]
    public static func packWToCol(W: Tensor, Cout: Int, Cin: Int, K: Int) -> Tensor {
        precondition(W.shape == [Cout, Cin, K])
        let ctx = MPSGContext.shared
        do { try ensure(device: ctx.device) } catch { fatalError("ConvPackGPU: pipeline error: \(error)") }
        guard let p = pPack, let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("ConvPackGPU: command buffer failed") }
        cmd.label = "ConvPackGPU.pack"
        let outCols = Cin * K
        let outCount = Cout * outCols
        let elem = MemoryLayout<Float>.size
        let inBuf = BufferPool.buffer(device: ctx.device, length: Cout * Cin * K * elem, label: "ConvPack.W.in")
        let outBuf = BufferPool.buffer(device: ctx.device, length: outCount * elem, label: "ConvPack.Wcol.out")
        W.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(inBuf.contents(), base, Cout * Cin * K * elem) } }
        memset(outBuf.contents(), 0, outCount * elem)
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("ConvPackGPU: encoder failed") }
        enc.label = "ConvPack.pack.enc"
        enc.setComputePipelineState(p)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var vCout = Int32(Cout), vCin = Int32(Cin), vK = Int32(K)
        enc.setBytes(&vCout, length: MemoryLayout<Int32>.size, index: 2)
        enc.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 4)
        let total = Cout * Cin * K
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (total + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        var outHost = [Float](repeating: 0, count: outCount)
        memcpy(&outHost, outBuf.contents(), outCount * elem)
        return Tensor(shape: [Cout, outCols], data: outHost)
    }

    // Unpack dWcol [Cout, Cin*K] -> dW [Cout, Cin, K]
    public static func unpackDWCol(dWcol: Tensor, Cout: Int, Cin: Int, K: Int) -> Tensor {
        precondition(dWcol.shape == [Cout, Cin * K])
        let ctx = MPSGContext.shared
        do { try ensure(device: ctx.device) } catch { fatalError("ConvPackGPU: pipeline error: \(error)") }
        guard let p = pUnpack, let cmd = ctx.commandQueue.makeCommandBuffer() else { fatalError("ConvPackGPU: command buffer failed") }
        cmd.label = "ConvPackGPU.unpack"
        let inCount = Cout * Cin * K
        let elem = MemoryLayout<Float>.size
        let inBuf = BufferPool.buffer(device: ctx.device, length: Cout * Cin * K * elem, label: "ConvPack.dWcol.in")
        let outBuf = BufferPool.buffer(device: ctx.device, length: inCount * elem, label: "ConvPack.dW.out")
        dWcol.data.withUnsafeBytes { raw in if let base = raw.baseAddress { memcpy(inBuf.contents(), base, Cout * Cin * K * elem) } }
        memset(outBuf.contents(), 0, inCount * elem)
        guard let enc = cmd.makeComputeCommandEncoder() else { fatalError("ConvPackGPU: encoder failed") }
        enc.label = "ConvPack.unpack.enc"
        enc.setComputePipelineState(p)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var vCout = Int32(Cout), vCin = Int32(Cin), vK = Int32(K)
        enc.setBytes(&vCout, length: MemoryLayout<Int32>.size, index: 2)
        enc.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 3)
        enc.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 4)
        let total = Cout * Cin * K
        let tpt = MTLSize(width: 256, height: 1, depth: 1)
        let tg = MTLSize(width: (total + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: tpt)
        enc.endEncoding()
        cmd.commit(); cmd.waitUntilCompleted()
        var outHost = [Float](repeating: 0, count: total)
        memcpy(&outHost, outBuf.contents(), total * elem)
        return Tensor(shape: [Cout, Cin, K], data: outHost)
    }

    private static let metalSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    // in: W[o,i,k], out: Wcol[o, i*K + k]
    kernel void pack_w_to_col_f32(
        const device float*  W     [[buffer(0)]],
        device float*        Wcol  [[buffer(1)]],
        constant int&        Cout  [[buffer(2)]],
        constant int&        Cin   [[buffer(3)]],
        constant int&        K     [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        int total = Cout * Cin * K;
        if ((int)gid >= total) return;
        int rem = (int)gid;
        int o = rem / (Cin * K);
        rem = rem % (Cin * K);
        int i = rem / K;
        int k = rem % K;
        int src = (o * Cin + i) * K + k;
        int dst = o * (Cin * K) + (i * K + k);
        Wcol[dst] = W[src];
    }

    // in: dWcol[o, i*K + k], out: dW[o,i,k]
    kernel void unpack_dwcol_f32(
        const device float*  dWcol [[buffer(0)]],
        device float*        dW    [[buffer(1)]],
        constant int&        Cout  [[buffer(2)]],
        constant int&        Cin   [[buffer(3)]],
        constant int&        K     [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        int total = Cout * Cin * K;
        if ((int)gid >= total) return;
        int rem = (int)gid;
        int o = rem / (Cin * K);
        rem = rem % (Cin * K);
        int i = rem / K;
        int k = rem % K;
        int src = o * (Cin * K) + (i * K + k);
        int dst = (o * Cin + i) * K + k;
        dW[dst] = dWcol[src];
    }
    """
}