import Foundation
import EFCore

public struct LastTCNParams {
    public var w1: Tensor
    public var b1: Tensor?
    public var w2: Tensor
    public var b2: Tensor?
    public var gamma: Tensor
    public var beta: Tensor
    public init(w1: Tensor, b1: Tensor?, w2: Tensor, b2: Tensor?, gamma: Tensor, beta: Tensor) {
        self.w1 = w1; self.b1 = b1; self.w2 = w2; self.b2 = b2; self.gamma = gamma; self.beta = beta
    }
}

public struct LastTCNGrads {
    public var dW1: Tensor
    public var dB1: Tensor?
    public var dW2: Tensor
    public var dB2: Tensor?
    public var dGamma: Tensor
    public var dBeta: Tensor
    public init(dW1: Tensor, dB1: Tensor?, dW2: Tensor, dB2: Tensor?, dGamma: Tensor, dBeta: Tensor) {
        self.dW1 = dW1; self.dB1 = dB1; self.dW2 = dW2; self.dB2 = dB2; self.dGamma = dGamma; self.dBeta = dBeta
    }
}

// Backward for last TCN block.
// Inputs:
// - cache: xIn [B,L,D], norm [B,L,D], h1 [B,L,H], h1a [B,L,H]
// - mask: [B][L] (fixed to maxLen)
// - dOut: [B,L,D] upstream gradient from masked-mean backward
// - modelCfg: TextToCubeEncoderConfig (to get dilation schedule)
// - params: current parameters of last block
// Returns grads for w1,b1,w2,b2,gamma,beta. Upstream dx before last block is not returned.
public func lastTCNBackward(cache: TextToCubeEncoder.LastTCNCache,
                            mask: [[Int]],
                            dOut: Tensor,
                            modelCfg: TextToCubeEncoderConfig,
                            params: LastTCNParams) throws -> LastTCNGrads {
    let B = cache.xIn.shape[0]
    let L = cache.xIn.shape[1]
    let D = cache.xIn.shape[2]
    precondition(dOut.shape == [B, L, D])
    precondition(mask.count == B && mask.allSatisfy { $0.count == L })
    // Zero masked positions in dOut
    var dY = dOut
    for b in 0..<B { for t in 0..<L { if mask[b][t] == 0 {
        let base = (b * L + t) * D
        for di in 0..<D { dY.data[base + di] = 0 }
    } } }
    // Conv2 backward via GraphLinear on flattened [B*L, H]
    let H = cache.h1a.shape[2]
    let Xf = cache.h1a.reshaped([B * L, H])
    let dYf = dY.reshaped([B * L, D])
    var gl = GraphLinear(inFeatures: H, outFeatures: D, bias: params.b2 != nil, seed: 0)
    gl.weight = params.w2.reshaped([D, H])
    gl.bias = params.b2
    _ = try? gl.forward(Tensor.zeros([1, H])) // populate GPU caches
    let (dW2lin, dB2) = try gl.gradientsGPU(X: Xf, dY: dYf)
    let dX2f = try gl.inputGradientsGPU(dY: dYf)
    let dX2 = dX2f.reshaped([B, L, H])
    // GELU backward on h1
    let dH1 = dGELU(x: cache.h1, upstream: dX2)
    // Conv1 backward (CPU ref with dilation)
    let dil = modelCfg.kernelSize == 1 ? 1 : (modelCfg.dilationSchedule.last ?? 1)
    let conv1 = Conv1DGrad.backward(X: cache.norm, W: params.w1, dY: dH1, dilation: dil)
    // LN backward (row-wise on [B*L, D])
    let xFlat = cache.xIn.reshaped([B * L, D])
    let gNormFlat = conv1.dX.reshaped([B * L, D])
    let (dxFlat, dGamma, dBeta) = layerNormBackward(x: xFlat, upstream: gNormFlat, gamma: params.gamma)
    _ = dxFlat // not propagated further here
    // Shape dW2 back to [D,H,1]
    let dW2 = dW2lin.reshaped([D, H, 1])
    return LastTCNGrads(dW1: conv1.dW,
                        dB1: params.b1 != nil ? conv1.dB : nil,
                        dW2: dW2,
                        dB2: params.b2 != nil ? dB2 : nil,
                        dGamma: dGamma,
                        dBeta: dBeta)
}
