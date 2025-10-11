import Foundation
import EFCore
import PyTorchSwift

public final class DecoderTrainer {
    public let config: TextDecoderConfig
    public var decoder: TextDecoder
    public var opt: AdamW
    public let gpu: GPUActor

    public init(config: TextDecoderConfig,
                lr: Float = 5e-3,
                weightDecay: Float = 0.0,
                seed: UInt64 = 0xDCD010,
                gpu: GPUActor = GPU.shared) {
        self.config = config
        self.decoder = TextDecoder(config: config, seed: seed)
        self.opt = AdamW(lr: lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: weightDecay)
        self.gpu = gpu
    }

    // One training step (projection-only): returns CE loss
    public func step(ids: [[Int]],
                     zTeacher: Tensor,
                     targets: [[Int]],
                     unfreezeLastTCN: Bool = false) async throws -> Float {
        await gpu.beginBatch()
        let (flat, logits, cache) = try await decoder.forwardForTraining(ids: ids,
                                                                         z: zTeacher,
                                                                         on: gpu)
        let ceReadback = try await gpu.crossEntropyMeanDeferred(logits: logits, targets: targets)
        let dLogits = CrossEntropyLoss.gradLogits(logits: logits, targets: targets)
        let (w0, b0) = decoder.getOutProjParams()
        var gl = GraphLinear(inFeatures: config.dim,
                             outFeatures: config.vocabSize,
                             bias: b0 != nil,
                             seed: 0)
        gl.weight = w0
        gl.bias = b0
        _ = try await gl.forwardAsync(Tensor.zeros([1, config.dim]), on: gpu)
        let gradReadback = try await gl.gradientsGPUDeferred(X: flat, dY: dLogits, on: gpu)
        let dXflat = try await gl.inputGradientsGPUAsync(dY: dLogits, on: gpu)

        var lastParams: (w1: Tensor, b1: Tensor?, w2: Tensor, b2: Tensor?, gamma: Tensor, beta: Tensor)? = nil
        var lastGrads: LastTCNGrads? = nil
        if unfreezeLastTCN {
            let B = ids.count, L = config.maxLength, D = config.dim
            let dOut = dXflat.reshaped([B, L, D])
            let p = decoder.getLastBlockParams()
            let gradsLast = try await decoderLastTCNBackward(cache: cache,
                                                             dOut: dOut,
                                                             kernelSize: config.kernelSize,
                                                             dilation: (config.dilationSchedule.last ?? 1),
                                                             params: LastTCNParams(w1: p.w1, b1: p.b1, w2: p.w2, b2: p.b2, gamma: p.gamma, beta: p.beta),
                                                             on: gpu)
            lastParams = p
            lastGrads = gradsLast
        }

        await gpu.syncBatch(label: "decoderTrainer.step")
        let (dW, dB) = try await gradReadback.value()
        let ce = try await ceReadback.value()

        var params: [Tensor] = [w0]
        var grads: [Tensor] = [dW]
        if let b = b0, dB.count > 0 { params.append(b); grads.append(dB) }
        if unfreezeLastTCN, let lp = lastParams, let lg = lastGrads {
            params.append(lp.w1); grads.append(lg.dW1)
            if let b1 = lp.b1, let db1 = lg.dB1 { params.append(b1); grads.append(db1) }
            params.append(lp.w2); grads.append(lg.dW2)
            if let b2 = lp.b2, let db2 = lg.dB2 { params.append(b2); grads.append(db2) }
            params.append(lp.gamma); grads.append(lg.dGamma)
            params.append(lp.beta); grads.append(lg.dBeta)
        }

        var pack = params
        opt.step(params: &pack, grads: grads)

        var cursor = 0
        let newW = pack[cursor]; cursor += 1
        var newB: Tensor? = nil
        if b0 != nil { newB = pack[cursor]; cursor += 1 }
        decoder.setOutProjParams(weight: newW, bias: newB)
        decoder.invalidateOutProjCache()

        if unfreezeLastTCN, let lp = lastParams {
            let newW1 = pack[cursor]; cursor += 1
            var newB1: Tensor? = nil
            if lp.b1 != nil { newB1 = pack[cursor]; cursor += 1 }
            let newW2 = pack[cursor]; cursor += 1
            var newB2: Tensor? = nil
            if lp.b2 != nil { newB2 = pack[cursor]; cursor += 1 }
            let newGamma = pack[cursor]; cursor += 1
            let newBeta = pack[cursor]; cursor += 1
            decoder.setLastBlockParams(w1: newW1, b1: newB1, w2: newW2, b2: newB2, gamma: newGamma, beta: newBeta)
            decoder.invalidateLastBlockCaches()
        }

        return ce
    }

    // Scaled training step with mixed-precision support and overflow detection.
    // scale: multiply upstream dLogits by this factor; grads are unscaled internally before opt.step.
    // clipNorm: optional global L2 grad clip across outProj and optional last TCN block
    // Returns (ce, overflow)
    public func stepScaled(ids: [[Int]],
                           zTeacher: Tensor,
                           targets: [[Int]],
                           unfreezeLastTCN: Bool = false,
                           scale: Float = 1.0,
                           clipNorm: Float = 0.0) async throws -> (Float, Bool) {
        await gpu.beginBatch()
        // Forward
        let (flat, logits, cache) = try await decoder.forwardForTraining(ids: ids,
                                                                         z: zTeacher,
                                                                         on: gpu)
        let ceReadback = try await gpu.crossEntropyMeanDeferred(logits: logits, targets: targets)
        var dLogits = CrossEntropyLoss.gradLogits(logits: logits, targets: targets)
        if scale != 1.0 {
            for i in 0..<dLogits.count { dLogits.data[i] *= scale }
        }
        let (w0, b0) = decoder.getOutProjParams()
        var gl = GraphLinear(inFeatures: config.dim,
                             outFeatures: config.vocabSize,
                             bias: b0 != nil,
                             seed: 0)
        gl.weight = w0
        gl.bias = b0
        _ = try await gl.forwardAsync(Tensor.zeros([1, config.dim]), on: gpu)
        let gradReadback = try await gl.gradientsGPUDeferred(X: flat, dY: dLogits, on: gpu)
        let dXflat = try await gl.inputGradientsGPUAsync(dY: dLogits, on: gpu)
        func hasNaNOrInf(_ t: Tensor) -> Bool { for v in t.data { if !v.isFinite { return true } } ; return false }
        var overflow = false
        var lastParams: (w1: Tensor, b1: Tensor?, w2: Tensor, b2: Tensor?, gamma: Tensor, beta: Tensor)? = nil
        var lastGrads: LastTCNGrads? = nil
        if unfreezeLastTCN {
            let B = ids.count, L = config.maxLength, D = config.dim
            let dOut = dXflat.reshaped([B, L, D])
            let p = decoder.getLastBlockParams()
            let gradsLast = try await decoderLastTCNBackward(cache: cache,
                                                             dOut: dOut,
                                                             kernelSize: config.kernelSize,
                                                             dilation: (config.dilationSchedule.last ?? 1),
                                                             params: LastTCNParams(w1: p.w1, b1: p.b1, w2: p.w2, b2: p.b2, gamma: p.gamma, beta: p.beta),
                                                             on: gpu)
            lastParams = p
            lastGrads = gradsLast
            if hasNaNOrInf(gradsLast.dW1) { overflow = true }
            if let db1 = gradsLast.dB1, hasNaNOrInf(db1) { overflow = true }
            if hasNaNOrInf(gradsLast.dW2) { overflow = true }
            if let db2 = gradsLast.dB2, hasNaNOrInf(db2) { overflow = true }
            if hasNaNOrInf(gradsLast.dGamma) || hasNaNOrInf(gradsLast.dBeta) { overflow = true }
        }
        await gpu.syncBatch(label: "decoderTrainer.stepScaled")
        let (dW, dB) = try await gradReadback.value()
        if hasNaNOrInf(dW) || hasNaNOrInf(dB) { overflow = true }
        let ce = try await ceReadback.value()
        if overflow {
            return (ce, true)
        }

        var params: [Tensor] = [w0]
        var grads: [Tensor] = [dW]
        if let b = b0, dB.count > 0 { params.append(b); grads.append(dB) }
        if unfreezeLastTCN, let lp = lastParams, let lg = lastGrads {
            params.append(lp.w1); grads.append(lg.dW1)
            if let b1 = lp.b1, let db1 = lg.dB1 { params.append(b1); grads.append(db1) }
            params.append(lp.w2); grads.append(lg.dW2)
            if let b2 = lp.b2, let db2 = lg.dB2 { params.append(b2); grads.append(db2) }
            params.append(lp.gamma); grads.append(lg.dGamma)
            params.append(lp.beta); grads.append(lg.dBeta)
        }

        let inv = (scale == 0) ? 1.0 : (1.0 / scale)
        if inv != 1.0 {
            for i in 0..<grads.count { for j in 0..<grads[i].count { grads[i].data[j] *= inv } }
        }
        if clipNorm > 0 {
            var gl = grads
            _ = GradClip.clipGlobalL2Norm(tensors: &gl, maxNorm: clipNorm, eps: 1e-6)
            grads = gl
        }

        var pack = params
        opt.step(params: &pack, grads: grads)

        var cursor = 0
        let newW = pack[cursor]; cursor += 1
        var newB: Tensor? = nil
        if b0 != nil { newB = pack[cursor]; cursor += 1 }
        decoder.setOutProjParams(weight: newW, bias: newB)
        decoder.invalidateOutProjCache()

        if unfreezeLastTCN, let lp = lastParams {
            let newW1 = pack[cursor]; cursor += 1
            var newB1: Tensor? = nil
            if lp.b1 != nil { newB1 = pack[cursor]; cursor += 1 }
            let newW2 = pack[cursor]; cursor += 1
            var newB2: Tensor? = nil
            if lp.b2 != nil { newB2 = pack[cursor]; cursor += 1 }
            let newGamma = pack[cursor]; cursor += 1
            let newBeta = pack[cursor]; cursor += 1
            decoder.setLastBlockParams(w1: newW1, b1: newB1, w2: newW2, b2: newB2, gamma: newGamma, beta: newBeta)
            decoder.invalidateLastBlockCaches()
        }

        return (ce, false)
    }
}
