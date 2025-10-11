import Foundation
import EFCore

// Backward for last TCN block in decoder (mirrors encoder's LastTCNBackward)
// Inputs:
// - cache: xIn [B,L,D], norm [B,L,D], h1 [B,L,H], h1a [B,L,H]
// - dOut: [B,L,D] upstream gradient from vocab head (after residual)
// - params: last block parameters (conv1, conv2, LN)
// Returns gradients for w1,b1,w2,b2,gamma,beta
public func decoderLastTCNBackward(cache: TextDecoder.LastTCNCache,
                                   dOut: Tensor,
                                   kernelSize: Int,
                                   dilation: Int,
                                   params: LastTCNParams,
                                   on gpu: GPUActor = GPU.shared) async throws -> LastTCNGrads {
    let B = cache.xIn.shape[0]
    let L = cache.xIn.shape[1]
    let D = cache.xIn.shape[2]
    precondition(dOut.shape == [B, L, D])
    // Conv2 backward via GraphLinear on flattened [B*L, H]
    let H = cache.h1a.shape[2]
    let Xf = cache.h1a.reshaped([B * L, H])
    let dYf = dOut.reshaped([B * L, D])
    var gl = GraphLinear(inFeatures: H, outFeatures: D, bias: params.b2 != nil, seed: 0)
    gl.weight = params.w2.reshaped([D, H])
    gl.bias = params.b2
    _ = try? await gl.forwardAsync(Tensor.zeros([1, H]), on: gpu)
    let (dW2lin, dB2) = try await gl.gradientsGPUAsync(X: Xf, dY: dYf, on: gpu)
    let dX2f = try await gl.inputGradientsGPUAsync(dY: dYf, on: gpu)
    let dX2 = dX2f.reshaped([B, L, H])
    // GELU backward on h1 (GPU)
    let dH1 = try await gpu.geluBackward(x: cache.h1, dY: dX2)
    // Conv1 backward (GPU GEMM + GPU im2col/col2im)
    let dil = max(1, dilation)
    let K1 = kernelSize
    let Cin1 = cache.norm.shape[2]
    let Cout1 = params.w1.shape[0]
    // Build Xcol [B*L, Cin*K]
    let rows = B * L
    // Avoid capturing non-Sendable 'cache' by copying the needed tensor into a local
    let norm = cache.norm
    let xcol = try await gpu.im2col(X: norm, B: B, L: L, Cin: Cin1, K: K1, dilation: dil)
    // Repack W1 -> Wcol [Cout, Cin*K] on GPU
    // Avoid capturing non-Sendable 'params' by copying the needed tensor into a local
    let W1 = params.w1
    let wcol = try await gpu.packWToCol(W: W1, Cout: Cout1, Cin: Cin1, K: K1)
    var gl1 = GraphLinear(inFeatures: Cin1 * K1, outFeatures: Cout1, bias: params.b1 != nil, seed: 0)
    gl1.weight = wcol
    gl1.bias = params.b1
    _ = try? await gl1.forwardAsync(Tensor.zeros([1, Cin1 * K1]), on: gpu)
    let dY1 = dH1.reshaped([rows, Cout1])
    let (dW1col, dB1gpu) = try await gl1.gradientsGPUAsync(X: xcol, dY: dY1, on: gpu)
    let dXcol = try await gl1.inputGradientsGPUAsync(dY: dY1, on: gpu)
    // Map dW1col -> dW1 [Cout, Cin, K] on GPU
    let dW1 = try await gpu.unpackDWCol(dWcol: dW1col, Cout: Cout1, Cin: Cin1, K: K1)
    // col2im: dXcol [rows, Cin*K] -> dX [B, L, Cin]
    let dX1 = try await gpu.col2im(dXcol: dXcol, B: B, L: L, Cin: Cin1, K: K1, dilation: dil)
    // LN backward (row-wise on [B*L, D]) using dX1
    let xFlat = cache.xIn.reshaped([B * L, D])
    let gNormFlat = dX1.reshaped([B * L, D])
    let (dxFlat, dGamma, dBeta) = try await gpu.layerNormBackward(x: xFlat,
                                                                  g: gNormFlat,
                                                                  gamma: params.gamma,
                                                                  eps: 1e-5)
    _ = dxFlat
    // Shape dW2 back to [D,H,1]
    let dW2 = dW2lin.reshaped([D, H, 1])
    return LastTCNGrads(dW1: dW1,
                        dB1: params.b1 != nil ? dB1gpu : nil,
                        dW2: dW2,
                        dB2: params.b2 != nil ? dB2 : nil,
                        dGamma: dGamma,
                        dBeta: dBeta)
}
