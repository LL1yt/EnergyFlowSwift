import Foundation
import EFCore

public final class CombinedTrainer {
    public let enc: TextToCubeEncoder
    public let decTrainer: DecoderTrainer
    public var optEncProj: AdamW
    public var alphaCos: Float
    public var betaMSE: Float

    public init(encConfig: TextToCubeEncoderConfig,
                decConfig: TextDecoderConfig,
                lrEncProj: Float = 5e-3,
                weightDecayEnc: Float = 0.0,
                alphaCos: Float = 1.0,
                betaMSE: Float = 1.0) {
        self.enc = TextToCubeEncoder(modelConfig: encConfig)
        self.decTrainer = DecoderTrainer(config: decConfig)
        self.optEncProj = AdamW(lr: lrEncProj, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: weightDecayEnc)
        self.alphaCos = alphaCos
        self.betaMSE = betaMSE
        // Initial sync of last block params encoder -> decoder
        let p = enc.getLastBlockParams()
        decTrainer.decoder.setLastBlockParams(w1: p.w1, b1: p.b1, w2: p.w2, b2: p.b2, gamma: p.gamma, beta: p.beta)
        decTrainer.decoder.invalidateLastBlockCaches()
    }

    // Mode A: tokens -> z_s, KD vs z_t; projection-only + optional last-block update
    public func stepA(inputIDs: [[Int]], attentionMask: [[Int]], zTeacher: Tensor, unfreezeLastTCN: Bool = true) throws -> (mse: Float, cos: Float) {
        let modelCfg = enc.modelConfig
        // Forward with cache
        let res = enc.forwardForTrainingWithLastBlockCache(inputIDs: inputIDs, attentionMask: attentionMask)
        let out = res.out // [B, D]
        // KD losses
        let mseRow = Losses.mseRowwise(out, zTeacher)
        let cosRow = Losses.cosineSimilarityRowwise(out, zTeacher)
        // Build dY for combined loss alpha*(1-cos) + beta*MSE
        var dY = dY_MSEMean(y: out, target: zTeacher)
        let dYcos = dY_CosineMeanLoss(y: out, target: zTeacher)
        let B = dY.shape[0]; let D = dY.shape[1]
        for i in 0..<(B*D) { dY.data[i] = betaMSE * dY.data[i] + alphaCos * dYcos.data[i] }
        // Projection grads + upstream to encoder
        let (dWproj, dBproj) = try enc.projectionGradientsGPU(X: res.pooled, dY: dY)
        let dXin = try enc.projectionInputGradientsGPU(dY: dY)
        let dEnc = enc.maskedMeanBackward(dPooled: dXin, mask: res.maskFixed, seqLen: modelCfg.maxLength)
        // Last block grads
        var paramsList: [Tensor] = []
        var gradsList: [Tensor] = []
        // out-proj first
        let proj = enc.getProjParams()
        paramsList.append(proj.weight); gradsList.append(dWproj)
        if let bp = proj.bias { paramsList.append(bp); gradsList.append(dBproj) }
        if unfreezeLastTCN {
            let p = enc.getLastBlockParams()
            let grads = try lastTCNBackward(cache: res.cache, mask: res.maskFixed, dOut: dEnc, modelCfg: modelCfg, params: LastTCNParams(w1: p.w1, b1: p.b1, w2: p.w2, b2: p.b2, gamma: p.gamma, beta: p.beta))
            paramsList.append(p.w1); gradsList.append(grads.dW1)
            if let b1 = p.b1, let db1 = grads.dB1 { paramsList.append(b1); gradsList.append(db1) }
            paramsList.append(p.w2); gradsList.append(grads.dW2)
            if let b2 = p.b2, let db2 = grads.dB2 { paramsList.append(b2); gradsList.append(db2) }
            paramsList.append(p.gamma); gradsList.append(grads.dGamma)
            paramsList.append(p.beta); gradsList.append(grads.dBeta)
        }
        var pack = paramsList
        optEncProj.step(params: &pack, grads: gradsList)
        // Write back
        var cursor = 0
        let newWproj = pack[cursor]; cursor += 1
        var newBproj: Tensor? = nil
        if enc.getProjParams().bias != nil { newBproj = pack[cursor]; cursor += 1 }
        enc.setProjParams(weight: newWproj, bias: newBproj)
        enc.invalidateProjectionCache()
        if unfreezeLastTCN {
            let p0 = enc.getLastBlockParams()
            let newW1 = pack[cursor]; cursor += 1
            var newB1: Tensor? = nil
            if p0.b1 != nil { newB1 = pack[cursor]; cursor += 1 }
            let newW2 = pack[cursor]; cursor += 1
            var newB2: Tensor? = nil
            if p0.b2 != nil { newB2 = pack[cursor]; cursor += 1 }
            let newGamma = pack[cursor]; cursor += 1
            let newBeta = pack[cursor]; cursor += 1
            enc.setLastBlockParams(w1: newW1, b1: newB1, w2: newW2, b2: newB2, gamma: newGamma, beta: newBeta)
            enc.invalidateLastBlockCaches()
            // Sync last block enc -> dec
            let pSync = enc.getLastBlockParams()
            decTrainer.decoder.setLastBlockParams(w1: pSync.w1, b1: pSync.b1, w2: pSync.w2, b2: pSync.b2, gamma: pSync.gamma, beta: pSync.beta)
            decTrainer.decoder.invalidateLastBlockCaches()
        }
        return (mseRow.mean, cosRow.mean)
    }

    // Mode B: z_t + teacher forcing CE; uses DecoderTrainer and then syncs last block to encoder
    public func stepB(ids: [[Int]], targets: [[Int]], zTeacher: Tensor, unfreezeLastTCN: Bool = true) throws -> Float {
        let ce = try decTrainer.step(ids: ids, zTeacher: zTeacher, targets: targets, unfreezeLastTCN: unfreezeLastTCN)
        if unfreezeLastTCN {
            let p = decTrainer.decoder.getLastBlockParams()
            enc.setLastBlockParams(w1: p.w1, b1: p.b1, w2: p.w2, b2: p.b2, gamma: p.gamma, beta: p.beta)
            enc.invalidateLastBlockCaches()
        }
        return ce
    }
}
