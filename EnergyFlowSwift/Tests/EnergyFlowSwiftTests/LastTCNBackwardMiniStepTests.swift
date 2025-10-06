import XCTest
@testable import EnergyFlow
@testable import EFCore

final class LastTCNBackwardMiniStepTests: XCTestCase {
    func test_one_step_loss_decreases_with_last_block_unfreeze() throws {
        // Tiny config
        let modelCfg = TextToCubeEncoderConfig(
            hiddenDim: 32,
            maxLength: 16,
            outputDim: 16,
            useTanhOutput: false,
            tcnBlocks: 1,
            kernelSize: 1,
            dilationSchedule: [1],
            ffDim: 32,
            useGPUProjection: true,
            baseSeed: 123
        )
        let enc = TextToCubeEncoder(modelConfig: modelCfg, vocabSize: 200)
        // Tiny token batch
        let B = 2, L = 8, D = modelCfg.outputDim
        var ids: [[Int]] = Array(repeating: Array(repeating: 1, count: L), count: B)
        var mask: [[Int]] = Array(repeating: Array(repeating: 1, count: L), count: B)
        // Create random-ish targets
        var targets: [[Float]] = []
        for _ in 0..<B {
            var row: [Float] = []
            for _ in 0..<D { row.append(Float.random(in: -0.1...0.1)) }
            targets.append(row)
        }
        // Forward with cache
        let res = enc.forwardForTrainingWithLastBlockCache(inputIDs: ids, attentionMask: mask)
        let out0 = res.out
        // Pre-loss
        let t0 = Tensor(shape: [B, D], data: targets.flatMap { $0 })
        let mse0 = Losses.mseRowwise(out0, t0)
        let cos0 = Losses.cosineSimilarityRowwise(out0, t0)
        let alpha: Float = 1.0, beta: Float = 1.0
        let loss0 = Double((1 - cos0.mean) * alpha + mse0.mean * beta)
        // Build dY
        var dY = dY_MSEMean(y: out0, target: t0)
        let dYcos = dY_CosineMeanLoss(y: out0, target: t0)
        for i in 0..<dY.count { dY.data[i] = beta * dY.data[i] + alpha * dYcos.data[i] }
        // Projection grads + dXin
        let (pooled, _) = (res.pooled, res.out)
        let (dWproj, dBproj) = try enc.projectionGradientsGPU(X: pooled, dY: dY)
        let dXin = try enc.projectionInputGradientsGPU(dY: dY)
        // Back through masked mean
        let dEnc = enc.maskedMeanBackward(dPooled: dXin, mask: mask, seqLen: modelCfg.maxLength)
        // Last block grads
        let params = enc.getLastBlockParams()
        let grads = try lastTCNBackward(cache: res.cache, mask: res.maskFixed, dOut: dEnc, modelCfg: modelCfg, params: LastTCNParams(w1: params.w1, b1: params.b1, w2: params.w2, b2: params.b2, gamma: params.gamma, beta: params.beta))
        // Optimizer step on projection + last block
        var paramsList: [Tensor] = [enc.getProjParams().weight]
        var gradsList: [Tensor] = [dWproj]
        if let bP = enc.getProjParams().bias { paramsList.append(bP); gradsList.append(dBproj) }
        paramsList.append(contentsOf: [params.w1])
        gradsList.append(grads.dW1)
        if let b1 = params.b1, let db1 = grads.dB1 { paramsList.append(b1); gradsList.append(db1) }
        paramsList.append(contentsOf: [params.w2])
        gradsList.append(grads.dW2)
        if let b2 = params.b2, let db2 = grads.dB2 { paramsList.append(b2); gradsList.append(db2) }
        paramsList.append(contentsOf: [params.gamma, params.beta])
        gradsList.append(contentsOf: [grads.dGamma, grads.dBeta])
        var plCopy = paramsList
        let opt = AdamW(lr: 5e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.0)
        opt.step(params: &plCopy, grads: gradsList)
        // Write back params
        var idx = 0
        let newWproj = plCopy[idx]; idx += 1
        var newBproj: Tensor? = nil
        if enc.getProjParams().bias != nil { newBproj = plCopy[idx]; idx += 1 }
        let newW1 = plCopy[idx]; idx += 1
        var newB1: Tensor? = nil
        if params.b1 != nil { newB1 = plCopy[idx]; idx += 1 }
        let newW2 = plCopy[idx]; idx += 1
        var newB2: Tensor? = nil
        if params.b2 != nil { newB2 = plCopy[idx]; idx += 1 }
        let newGamma = plCopy[idx]; idx += 1
        let newBeta = plCopy[idx]; idx += 1
        enc.setProjParams(weight: newWproj, bias: newBproj)
        enc.invalidateProjectionCache()
        enc.setLastBlockParams(w1: newW1, b1: newB1, w2: newW2, b2: newB2, gamma: newGamma, beta: newBeta)
        enc.invalidateLastBlockCaches()
        // Forward again
        let res2 = enc.forwardForTrainingWithLastBlockCache(inputIDs: ids, attentionMask: mask)
        let out1 = res2.out
        let mse1 = Losses.mseRowwise(out1, t0)
        let cos1 = Losses.cosineSimilarityRowwise(out1, t0)
        let loss1 = Double((1 - cos1.mean) * alpha + mse1.mean * beta)
        XCTAssertLessThan(loss1, loss0, "loss should decrease after one step")
    }
}
