import Foundation
import EFCore

public final class CombinedTrainer {
    public let enc: TextToCubeEncoder
    public let decTrainer: DecoderTrainer
    public var optEncProj: AdamW
    public var alphaCos: Float
    public var betaMSE: Float
    public let gpu: GPUActor
    // Scheduler & clip
    private let baseLREnc: Float
    private let minLREnc: Float
    private let warmupSteps: Int
    private let cosineDecaySteps: Int
    private let clipNorm: Float
    private var stepAIndex: Int = 0

    public init(encConfig: TextToCubeEncoderConfig,
                decConfig: TextDecoderConfig,
                lrEncProj: Float = 5e-3,
                weightDecayEnc: Float = 0.0,
                alphaCos: Float = 1.0,
                betaMSE: Float = 1.0,
                warmupSteps: Int = 0,
                cosineDecaySteps: Int = 0,
                minLREncProj: Float = 0.0,
                clipNorm: Float = 0.0,
                gpu: GPUActor = GPU.shared) {
        self.enc = TextToCubeEncoder(modelConfig: encConfig)
        self.decTrainer = DecoderTrainer(config: decConfig, gpu: gpu)
        self.optEncProj = AdamW(lr: lrEncProj, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: weightDecayEnc)
        self.alphaCos = alphaCos
        self.betaMSE = betaMSE
        self.gpu = gpu
        self.baseLREnc = lrEncProj
        self.minLREnc = minLREncProj
        self.warmupSteps = warmupSteps
        self.cosineDecaySteps = cosineDecaySteps
        self.clipNorm = clipNorm
        // Enforce dim sync: encoder/decoder trunks must match teacher embeddingDim
        let pe = enc.getLastBlockParams()
        let pd = decTrainer.decoder.getLastBlockParams()
        let match = pe.w1.shape == pd.w1.shape && pe.w2.shape == pd.w2.shape && pe.gamma.shape == pd.gamma.shape && pe.beta.shape == pd.beta.shape
        precondition(match, "Encoder/Decoder last TCN shapes mismatch. Ensure enc.hiddenDim == dec.dim == teacher embeddingDim. Got enc: w1=\(pe.w1.prettyShape) w2=\(pe.w2.prettyShape) gamma=\(pe.gamma.prettyShape); dec: w1=\(pd.w1.prettyShape) w2=\(pd.w2.prettyShape) gamma=\(pd.gamma.prettyShape)")
        decTrainer.decoder.setLastBlockParams(w1: pe.w1, b1: pe.b1, w2: pe.w2, b2: pe.b2, gamma: pe.gamma, beta: pe.beta)
        decTrainer.decoder.invalidateLastBlockCaches()
        Logger.shared.info1("Init sync enc->dec last TCN params (dims verified)", category: Logger.Category.training)
    }

    // Mode A: tokens -> z_s, KD vs z_t; projection-only + optional last-block update
    public func stepA(inputIDs: [[Int]],
                      attentionMask: [[Int]],
                      zTeacher: Tensor,
                      unfreezeLastTCN: Bool = true) async throws -> (mse: Float, cos: Float) {
        let modelCfg = enc.modelConfig
        await gpu.beginBatch()
        // Forward with cache
        let res = try await enc.forwardForTrainingWithLastBlockCache(inputIDs: inputIDs,
                                                                     attentionMask: attentionMask,
                                                                     on: gpu)
        let out = res.out // [B, D]
        // KD losses
        let metricsTask = Task<(Float, Float), Error> {
            try await gpu.kdMetricsMean(student: out, teacher: zTeacher)
        }
        defer { metricsTask.cancel() }
        // Build dY for combined loss alpha*(1-cos) + beta*MSE
        var dY = dY_MSEMean(y: out, target: zTeacher)
        let dYcos = dY_CosineMeanLoss(y: out, target: zTeacher)
        let B = dY.shape[0]; let D = dY.shape[1]
        for i in 0..<(B*D) { dY.data[i] = betaMSE * dY.data[i] + alphaCos * dYcos.data[i] }
        // Projection grads + upstream to encoder
        let (dWproj, dBproj) = try await enc.projectionGradientsGPU(X: res.pooled,
                                                                    dY: dY,
                                                                    on: gpu)
        let dXin = try await enc.projectionInputGradientsGPU(dY: dY, on: gpu)
        let dEnc = try await enc.maskedMeanBackward(dPooled: dXin,
                                                    mask: res.maskFixed,
                                                    seqLen: modelCfg.maxLength,
                                                    on: gpu)
        // Last block grads
        var paramsList: [Tensor] = []
        var gradsList: [Tensor] = []
        // out-proj first
        let proj = enc.getProjParams()
        paramsList.append(proj.weight); gradsList.append(dWproj)
        if let bp = proj.bias { paramsList.append(bp); gradsList.append(dBproj) }
        if unfreezeLastTCN {
            let p = enc.getLastBlockParams()
            let grads = try await lastTCNBackward(cache: res.cache,
                                                  mask: res.maskFixed,
                                                  dOut: dEnc,
                                                  modelCfg: modelCfg,
                                                  params: LastTCNParams(w1: p.w1, b1: p.b1, w2: p.w2, b2: p.b2, gamma: p.gamma, beta: p.beta),
                                                  on: gpu)
            paramsList.append(p.w1); gradsList.append(grads.dW1)
            if let b1 = p.b1, let db1 = grads.dB1 { paramsList.append(b1); gradsList.append(db1) }
            paramsList.append(p.w2); gradsList.append(grads.dW2)
            if let b2 = p.b2, let db2 = grads.dB2 { paramsList.append(b2); gradsList.append(db2) }
            paramsList.append(p.gamma); gradsList.append(grads.dGamma)
            paramsList.append(p.beta); gradsList.append(grads.dBeta)
        }
        // LR schedule for encoder projection/last block
        let lrNow = LRSchedulers.warmupCosine(baseLR: baseLREnc, minLR: minLREnc, warmupSteps: warmupSteps, decaySteps: cosineDecaySteps, step: stepAIndex)
        if optEncProj.lr != lrNow { optEncProj.lr = lrNow }
        // Global clip across collected grads (projection and optional last block)
        if clipNorm > 0 {
            var gl = gradsList
            _ = GradClip.clipGlobalL2Norm(tensors: &gl, maxNorm: clipNorm, eps: 1e-6)
            gradsList = gl
        }
        var pack = paramsList
        optEncProj.step(params: &pack, grads: gradsList)
        stepAIndex += 1
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
        // Friendly progress log at mid-verbosity (every 100 A-steps)
        await gpu.syncBatch(label: "trainer.stepA")
        let (mseMean, cosMean) = try await metricsTask.value
        if stepAIndex % 100 == 0 {
            Logger.shared.info1(String(format: "A-step %d: B=%d lr=%.4g clip=%.2f mse=%.6f cos=%.6f unfreeze=%@",
                                       stepAIndex, zTeacher.shape[0], lrNow, clipNorm,
                                       mseMean, cosMean, String(describing: unfreezeLastTCN)),
                                category: Logger.Category.training)
        }
        return (mseMean, cosMean)
    }

    // Mode B: z_t + teacher forcing CE; uses DecoderTrainer and then syncs last block to encoder
    public func stepB(ids: [[Int]],
                      targets: [[Int]],
                      zTeacher: Tensor,
                      unfreezeLastTCN: Bool = true) async throws -> Float {
        await gpu.beginBatch()
        let (ce, _) = try await decTrainer.stepScaled(ids: ids,
                                                      zTeacher: zTeacher,
                                                      targets: targets,
                                                      unfreezeLastTCN: unfreezeLastTCN,
                                                      scale: 1.0,
                                                      clipNorm: clipNorm)
        if unfreezeLastTCN {
            let pd = decTrainer.decoder.getLastBlockParams()
            let pe = enc.getLastBlockParams()
            let match = pd.w1.shape == pe.w1.shape && pd.w2.shape == pe.w2.shape && pd.gamma.shape == pe.gamma.shape && pd.beta.shape == pe.beta.shape
            precondition(match, "dec->enc last TCN sync mismatch (unexpected). Ensure enc.hiddenDim == dec.dim == teacher embeddingDim.")
            enc.setLastBlockParams(w1: pd.w1, b1: pd.b1, w2: pd.w2, b2: pd.b2, gamma: pd.gamma, beta: pd.beta)
            enc.invalidateLastBlockCaches()
        }
        // Occasional friendly log (every ~300 A-steps worth of B calls is arbitrary; here just per call throttle with small prob)
        if stepAIndex % 150 == 0 { // re-use A counter as rough global step
            Logger.shared.info1(String(format: "B-call at A-step %d: B=%d ce=%.6f unfreeze=%@",
                                       stepAIndex, ids.count, ce, String(describing: unfreezeLastTCN)),
                                category: Logger.Category.training)
        }
        await gpu.syncBatch(label: "trainer.stepB")
        return ce
    }
}
