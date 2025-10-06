import Foundation
import EFCore
import EnergyFlow

// Scales a tensor in-place
private func scaleTensor(_ t: inout Tensor, by s: Float) {
    if s == 1.0 { return }
    for i in 0..<t.count { t.data[i] *= s }
}

public struct OptimStepInputs {
    public var projGradW: Tensor
    public var projGradB: Tensor?
    public var lastGrads: LastTCNGrads?
    public var lrNow: Float
    public var scale: Float
    public var clipNorm: Float
}

// Perform one optimizer step over projection params and optionally last TCN block params.
// - Scales grads by `scale`
// - Optionally applies global L2 grad clipping
// - Sets opt.lr to lrNow and calls opt.step
// - Writes back parameters into encoder and invalidates caches
public func optimizerStepProjectionAndLastBlock(enc: TextToCubeEncoder,
                                                opt: AdamW,
                                                inputs: OptimStepInputs) {
    var params: [Tensor] = []
    var grads: [Tensor] = []
    // Projection params
    let (projW, projB) = enc.getProjParams()
    params.append(projW)
    var gW = inputs.projGradW
    scaleTensor(&gW, by: inputs.scale)
    grads.append(gW)
    if let bParam = projB {
        params.append(bParam)
        if var gB = inputs.projGradB { scaleTensor(&gB, by: inputs.scale); grads.append(gB) } else { grads.append(Tensor.zeros([bParam.count])) }
    }
    // Last block params (optional)
    if let lg = inputs.lastGrads {
        let lp = enc.getLastBlockParams()
        // w1, (b1?), w2, (b2?), gamma, beta
        params.append(lp.w1)
        var dW1 = lg.dW1; scaleTensor(&dW1, by: inputs.scale); grads.append(dW1)
        if let b1p = lp.b1, var dB1 = lg.dB1 { params.append(b1p); scaleTensor(&dB1, by: inputs.scale); grads.append(dB1) }
        params.append(lp.w2)
        var dW2 = lg.dW2; scaleTensor(&dW2, by: inputs.scale); grads.append(dW2)
        if let b2p = lp.b2, var dB2 = lg.dB2 { params.append(b2p); scaleTensor(&dB2, by: inputs.scale); grads.append(dB2) }
        params.append(lp.gamma)
        var dG = lg.dGamma; scaleTensor(&dG, by: inputs.scale); grads.append(dG)
        params.append(lp.beta)
        var dBt = lg.dBeta; scaleTensor(&dBt, by: inputs.scale); grads.append(dBt)
    }
    // Clip
    if inputs.clipNorm > 0 {
        var gl = grads
        _ = GradClip.clipGlobalL2Norm(tensors: &gl, maxNorm: inputs.clipNorm, eps: 1e-6)
        grads = gl
    }
    // Step
    if opt.lr != inputs.lrNow { opt.lr = inputs.lrNow }
    var paramsCopy = params
    opt.step(params: &paramsCopy, grads: grads)
    // Write back
    var idx = 0
    let newW = paramsCopy[idx]; idx += 1
    var newB: Tensor? = nil
    if projB != nil { newB = paramsCopy[idx]; idx += 1 }
    enc.setProjParams(weight: newW, bias: newB)
    enc.invalidateProjectionCache()
    if inputs.lastGrads != nil {
        // order: w1,(b1),w2,(b2),gamma,beta
        let w1 = paramsCopy[idx]; idx += 1
        var b1: Tensor? = nil
        if let _ = enc.getLastBlockParams().b1 { b1 = paramsCopy[idx]; idx += 1 }
        let w2 = paramsCopy[idx]; idx += 1
        var b2: Tensor? = nil
        if let _ = enc.getLastBlockParams().b2 { b2 = paramsCopy[idx]; idx += 1 }
        let gamma = paramsCopy[idx]; idx += 1
        let beta = paramsCopy[idx]; idx += 1
        enc.setLastBlockParams(w1: w1, b1: b1, w2: w2, b2: b2, gamma: gamma, beta: beta)
        enc.invalidateLastBlockCaches()
    }
}
