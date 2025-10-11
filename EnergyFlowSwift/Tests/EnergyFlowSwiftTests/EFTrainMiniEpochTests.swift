import XCTest
import class Foundation.FileManager
@testable import EFCore
@testable import EnergyFlow


final class EFTrainMiniEpochTests: XCTestCase {
    func test_mini_epoch_decreases_loss() async throws {
        // Tiny model config
        let cfg = TextToCubeEncoderConfig(
            hiddenDim: 32,
            maxLength: 16,
            outputDim: 16,
            useTanhOutput: false,
            tcnBlocks: 1,
            kernelSize: 3,
            dilationSchedule: [1],
            ffDim: 32,
            useGPUProjection: true,
            baseSeed: 7
        )
        let enc = TextToCubeEncoder(modelConfig: cfg, vocabSize: 256)
        // Build tiny synthetic token-mode dataset in-memory: B=4, L varies (<=maxLen)
        let B = 4, L = 12, D = cfg.outputDim
        let ids: [[Int]] = (0..<B).map { _ in (0..<L).map { _ in Int.random(in: 1..<200) } }
        let mask: [[Int]] = Array(repeating: Array(repeating: 1, count: L), count: B)
        let tgts: [[Float]] = (0..<B).map { _ in (0..<D).map { _ in Float.random(in: -0.1...0.1) } }
        // Forward with cache, compute initial loss
        let res = try await enc.forwardForTrainingWithLastBlockCache(inputIDs: ids,
                                                                     attentionMask: mask)
        let out0 = res.out
        let t0 = Tensor(shape: [B, D], data: tgts.flatMap { $0 })
        let mse0 = Losses.mseRowwise(out0, t0)
        let cos0 = Losses.cosineSimilarityRowwise(out0, t0)
        let alpha: Float = 1.0, beta: Float = 1.0
        let loss0 = Double((1 - cos0.mean) * alpha + mse0.mean * beta)
        // Build dY and projection grads
        var dY = dY_MSEMean(y: out0, target: t0)
        let dYcos = dY_CosineMeanLoss(y: out0, target: t0)
        for i in 0..<dY.count { dY.data[i] = beta * dY.data[i] + alpha * dYcos.data[i] }
        let (dWproj, dBproj) = try await enc.projectionGradientsGPU(X: res.pooled,
                                                                    dY: dY)
        let dXin = try await enc.projectionInputGradientsGPU(dY: dY)
        let dEnc = try await enc.maskedMeanBackward(dPooled: dXin,
                                                    mask: res.maskFixed,
                                                    seqLen: cfg.maxLength)
        // Last block grads via helper (with GPU conv1 GEMM)
        let params = enc.getLastBlockParams()
        let grads = try lastTCNBackward(cache: res.cache, mask: res.maskFixed, dOut: dEnc, modelCfg: cfg,
                                        params: LastTCNParams(w1: params.w1, b1: params.b1, w2: params.w2, b2: params.b2, gamma: params.gamma, beta: params.beta))
        // Optimizer step using helper
        let opt = AdamW(lr: 3e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0)
        optimizerStepProjectionAndLastBlock(enc: enc, opt: opt, inputs: OptimStepInputs(projGradW: dWproj, projGradB: dBproj, lastGrads: grads, lrNow: opt.lr, scale: 1.0, clipNorm: 0.0))
        // Forward again and check loss decreased
        let res2 = try await enc.forwardForTrainingWithLastBlockCache(inputIDs: ids,
                                                                      attentionMask: mask)
        let out1 = res2.out
        let mse1 = Losses.mseRowwise(out1, t0)
        let cos1 = Losses.cosineSimilarityRowwise(out1, t0)
        let loss1 = Double((1 - cos1.mean) * alpha + mse1.mean * beta)
        XCTAssertLessThan(loss1, loss0, "loss should decrease after mini epoch")
    }
}
