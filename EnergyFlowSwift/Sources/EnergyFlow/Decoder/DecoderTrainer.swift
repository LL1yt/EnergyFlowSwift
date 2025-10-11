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
        // Forward
        let (flat, logits, cache) = try await decoder.forwardForTraining(ids: ids,
                                                                         z: zTeacher,
                                                                         on: gpu)
        let ceTask = Task<Float, Error> {
            try await gpu.crossEntropyMean(logits: logits, targets: targets)
        }
        defer { ceTask.cancel() }
        // Gradients on outProj
        let dLogits = CrossEntropyLoss.gradLogits(logits: logits, targets: targets) // [B*L, V]
        let (w0, b0) = decoder.getOutProjParams()
        var gl = GraphLinear(inFeatures: config.dim,
                             outFeatures: config.vocabSize,
                             bias: b0 != nil,
                             seed: 0)
        gl.weight = w0
        gl.bias = b0
        _ = try await gl.forwardAsync(Tensor.zeros([1, config.dim]), on: gpu)
        let (dW, dB) = try await gl.gradientsGPUAsync(X: flat, dY: dLogits, on: gpu)
        let dXflat = try await gl.inputGradientsGPUAsync(dY: dLogits, on: gpu) // upstream for last block
        // Build params/grads pack
        var params: [Tensor] = [w0]
        var grads: [Tensor] = [dW]
        if let b = b0, dB.count > 0 { params.append(b); grads.append(dB) }
        // Optionally unfreeze last TCN block
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
            // Append last-block grads to packs
            params.append(p.w1); grads.append(gradsLast.dW1)
            if let b1 = p.b1, let db1 = gradsLast.dB1 { params.append(b1); grads.append(db1) }
            params.append(p.w2); grads.append(gradsLast.dW2)
            if let b2 = p.b2, let db2 = gradsLast.dB2 { params.append(b2); grads.append(db2) }
            params.append(p.gamma); grads.append(gradsLast.dGamma)
            params.append(p.beta); grads.append(gradsLast.dBeta)
        }
        var pack = params
        opt.step(params: &pack, grads: grads)
        let newW = pack[0]
        var cursor = 1
        let newB: Tensor? = (b0 != nil) ? pack[cursor] : nil
        if b0 != nil { cursor += 1 }
        // If unfreeze, write back last block params
        if unfreezeLastTCN {
            let p = decoder.getLastBlockParams()
            let newW1 = pack[cursor]; cursor += 1
            var newB1: Tensor? = nil
            if p.b1 != nil { newB1 = pack[cursor]; cursor += 1 }
            let newW2 = pack[cursor]; cursor += 1
            var newB2: Tensor? = nil
            if p.b2 != nil { newB2 = pack[cursor]; cursor += 1 }
            let newGamma = pack[cursor]; cursor += 1
            let newBeta = pack[cursor]; cursor += 1
            decoder.setLastBlockParams(w1: newW1, b1: newB1, w2: newW2, b2: newB2, gamma: newGamma, beta: newBeta)
            decoder.invalidateLastBlockCaches()
        }
        decoder.setOutProjParams(weight: newW, bias: newB)
        decoder.invalidateOutProjCache()
        await gpu.syncBatch(label: "decoderTrainer.step")
        let ce = try await ceTask.value
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
        let ceTask = Task<Float, Error> {
            try await gpu.crossEntropyMean(logits: logits, targets: targets)
        }
        defer { ceTask.cancel() }
        // dLogits (softmax - onehot) / (B*L)
        var dLogits = CrossEntropyLoss.gradLogits(logits: logits, targets: targets)
        // Scale upstream gradient
        if scale != 1.0 {
            for i in 0..<dLogits.count { dLogits.data[i] *= scale }
        }
        // Out projection grads
        let (w0, b0) = decoder.getOutProjParams()
        var gl = GraphLinear(inFeatures: config.dim,
                             outFeatures: config.vocabSize,
                             bias: b0 != nil,
                             seed: 0)
        gl.weight = w0
        gl.bias = b0
        _ = try await gl.forwardAsync(Tensor.zeros([1, config.dim]), on: gpu)
        let (dW, dB) = try await gl.gradientsGPUAsync(X: flat, dY: dLogits, on: gpu)
        let dXflat = try await gl.inputGradientsGPUAsync(dY: dLogits, on: gpu)
        // Overflow detection
        func hasNaNOrInf(_ t: Tensor) -> Bool { for v in t.data { if !v.isFinite { return true } } ; return false }
        var overflow = hasNaNOrInf(dW) || hasNaNOrInf(dB)
        var params: [Tensor] = [w0]
        var grads: [Tensor] = [dW]
        if let b = b0, dB.count > 0 { params.append(b); grads.append(dB) }
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
            // Append
            params.append(p.w1); grads.append(gradsLast.dW1)
            if let b1 = p.b1, let db1 = gradsLast.dB1 { params.append(b1); grads.append(db1) }
            params.append(p.w2); grads.append(gradsLast.dW2)
            if let b2 = p.b2, let db2 = gradsLast.dB2 { params.append(b2); grads.append(db2) }
            params.append(p.gamma); grads.append(gradsLast.dGamma)
            params.append(p.beta); grads.append(gradsLast.dBeta)
            // Overflow include last-block grads
            if hasNaNOrInf(gradsLast.dW1) { overflow = true }
            if let db1 = gradsLast.dB1, hasNaNOrInf(db1) { overflow = true }
            if hasNaNOrInf(gradsLast.dW2) { overflow = true }
            if let db2 = gradsLast.dB2, hasNaNOrInf(db2) { overflow = true }
            if hasNaNOrInf(gradsLast.dGamma) || hasNaNOrInf(gradsLast.dBeta) { overflow = true }
        }
        if overflow {
            await gpu.syncBatch(label: "decoderTrainer.stepScaled")
            let ce = try await ceTask.value
            return (ce, true)
        }
        // Unscale grads before step
        let inv = (scale == 0) ? 1.0 : (1.0 / scale)
        if inv != 1.0 {
            for i in 0..<grads.count { for j in 0..<grads[i].count { grads[i].data[j] *= inv } }
        }
        // Global grad clip across outProj and optional last-block
        if clipNorm > 0 {
            var gl = grads
            _ = GradClip.clipGlobalL2Norm(tensors: &gl, maxNorm: clipNorm, eps: 1e-6)
            grads = gl
        }
        var pack = params
        opt.step(params: &pack, grads: grads)
        // Write back
        var cursor = 0
        let newW = pack[cursor]; cursor += 1
        var newB: Tensor? = nil
        if b0 != nil { newB = pack[cursor]; cursor += 1 }
        decoder.setOutProjParams(weight: newW, bias: newB)
        decoder.invalidateOutProjCache()
        if unfreezeLastTCN {
            let p = decoder.getLastBlockParams()
            let newW1 = pack[cursor]; cursor += 1
            var newB1: Tensor? = nil
            if p.b1 != nil { newB1 = pack[cursor]; cursor += 1 }
            let newW2 = pack[cursor]; cursor += 1
            var newB2: Tensor? = nil
            if p.b2 != nil { newB2 = pack[cursor]; cursor += 1 }
            let newGamma = pack[cursor]; cursor += 1
            let newBeta = pack[cursor]; cursor += 1
            decoder.setLastBlockParams(w1: newW1, b1: newB1, w2: newW2, b2: newB2, gamma: newGamma, beta: newBeta)
            decoder.invalidateLastBlockCaches()
        }
        await gpu.syncBatch(label: "decoderTrainer.stepScaled")
        let ce = try await ceTask.value
        return (ce, false)
    }
}
