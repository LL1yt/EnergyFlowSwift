import Foundation
import EFCore
import PyTorchSwift

public final class DecoderTrainer {
    public let config: TextDecoderConfig
    public var decoder: TextDecoder
    public var opt: AdamW

    public init(config: TextDecoderConfig,
                lr: Float = 5e-3,
                weightDecay: Float = 0.0,
                seed: UInt64 = 0xDCD010) {
        self.config = config
        self.decoder = TextDecoder(config: config, seed: seed)
        self.opt = AdamW(lr: lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: weightDecay)
    }

    // One training step (projection-only): returns CE loss
    public func step(ids: [[Int]], zTeacher: Tensor, targets: [[Int]], unfreezeLastTCN: Bool = false) throws -> Float {
        // Forward
        let (flat, logits, cache) = decoder.forwardForTraining(ids: ids, z: zTeacher)
        let ce = CrossEntropyLoss.meanLogits(logits: logits, targets: targets)
        // Gradients on outProj
        let dLogits = CrossEntropyLoss.gradLogits(logits: logits, targets: targets) // [B*L, V]
        let (w0, b0) = decoder.getOutProjParams()
        var gl = GraphLinear(inFeatures: config.dim, outFeatures: config.vocabSize, bias: b0 != nil, seed: 0)
        gl.weight = w0
        gl.bias = b0
        _ = try? gl.forward(Tensor.zeros([1, config.dim]))
        let (dW, dB) = try gl.gradientsGPU(X: flat, dY: dLogits)
        let dXflat = try gl.inputGradientsGPU(dY: dLogits) // upstream for last block
        // Build params/grads pack
        var params: [Tensor] = [w0]
        var grads: [Tensor] = [dW]
        if let b = b0, dB.count > 0 { params.append(b); grads.append(dB) }
        // Optionally unfreeze last TCN block
        if unfreezeLastTCN {
            let B = ids.count, L = config.maxLength, D = config.dim
            let dOut = dXflat.reshaped([B, L, D])
            let p = decoder.getLastBlockParams()
            let gradsLast = try decoderLastTCNBackward(cache: cache, dOut: dOut, kernelSize: config.kernelSize, dilation: (config.dilationSchedule.last ?? 1), params: LastTCNParams(w1: p.w1, b1: p.b1, w2: p.w2, b2: p.b2, gamma: p.gamma, beta: p.beta))
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
        return ce
    }
}
