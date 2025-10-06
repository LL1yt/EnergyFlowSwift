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
    public func step(ids: [[Int]], zTeacher: Tensor, targets: [[Int]]) throws -> Float {
        // Forward
        let (flat, logits) = decoder.forwardForTraining(ids: ids, z: zTeacher)
        let ce = CrossEntropyLoss.meanLogits(logits: logits, targets: targets)
        // Gradients on outProj only
        let dLogits = CrossEntropyLoss.gradLogits(logits: logits, targets: targets) // [B*L, V]
        let (w0, b0) = decoder.getOutProjParams()
        var gl = GraphLinear(inFeatures: config.dim, outFeatures: config.vocabSize, bias: b0 != nil, seed: 0)
        gl.weight = w0
        gl.bias = b0
        _ = try? gl.forward(Tensor.zeros([1, config.dim])) // populate GPU caches
        let (dW, dB) = try gl.gradientsGPU(X: flat, dY: dLogits)
        // Apply AdamW on outProj only
        var params: [Tensor] = [w0]
        var grads: [Tensor] = [dW]
        if let b = b0, dB.count > 0 {
            params.append(b)
            grads.append(dB)
        }
        var pack = params
        opt.step(params: &pack, grads: grads)
        let newW = pack[0]
        let newB = (b0 != nil) ? pack[1] : nil
        decoder.setOutProjParams(weight: newW, bias: newB)
        decoder.invalidateOutProjCache()
        return ce
    }
}
