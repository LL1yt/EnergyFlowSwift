import Foundation
@preconcurrency import Metal

public enum Im2ColCol2ImGPU {
    public static func im2col(X: Tensor, B: Int, L: Int, Cin: Int, K: Int, dilation: Int) throws -> Tensor {
        try GPU.blocking(label: "Im2ColCol2ImGPU.im2col") { actor in
            try await actor.im2col(X: X, B: B, L: L, Cin: Cin, K: K, dilation: dilation)
        }
    }

    public static func col2im(dXcol: Tensor, B: Int, L: Int, Cin: Int, K: Int, dilation: Int) throws -> Tensor {
        try GPU.blocking(label: "Im2ColCol2ImGPU.col2im") { actor in
            try await actor.col2im(dXcol: dXcol, B: B, L: L, Cin: Cin, K: K, dilation: dilation)
        }
    }

    public static func im2colFP16ToBuffer(X: Tensor,
                                          B: Int,
                                          L: Int,
                                          Cin: Int,
                                          K: Int,
                                          dilation: Int,
                                          outBuf: MTLBuffer,
                                          outRowBytes: Int,
                                          outColsTotal: Int) throws {
        try GPU.blocking(label: "Im2ColCol2ImGPU.im2colFP16ToBuffer") { actor in
            try await actor.im2colFP16ToBuffer(X: X,
                                               B: B,
                                               L: L,
                                               Cin: Cin,
                                               K: K,
                                               dilation: dilation,
                                               outBuffer: outBuf,
                                               outRowBytes: outRowBytes,
                                               outColsTotal: outColsTotal)
        }
    }

    public static func fillBiasColumnFP16(outBuf: MTLBuffer, rows: Int, outRowBytes: Int, biasIndex: Int) throws {
        try GPU.blocking(label: "Im2ColCol2ImGPU.fillBiasColumnFP16") { actor in
            try await actor.fillBiasColumnFP16(outBuffer: outBuf,
                                              rows: rows,
                                              outRowBytes: outRowBytes,
                                              biasIndex: biasIndex)
        }
    }
}
