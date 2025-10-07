import XCTest
@testable import EnergyFlow
@testable import EFCore
@testable import PyTorchSwift

final class EFDecoderModeBMiniEpochTests: XCTestCase {
    func test_modeB_projection_only_training_decreases_CE() throws {
        // Config
        let cfg = TextDecoderConfig(vocabSize: 64, dim: 32, hidden: 64, nBlocks: 2, kernelSize: 3, dilationSchedule: [1,2], maxLength: 8)
        let dec = TextDecoder(config: cfg, seed: 2025)
        let B = 6, L = cfg.maxLength, V = cfg.vocabSize, D = cfg.dim
        // Synthetic batch: ids [B,L], targets = ids shifted by one (wrap), z_teacher random
        var ids: [[Int]] = []
        var targets: [[Int]] = []
        for _ in 0..<B {
            var row: [Int] = []
            for _ in 0..<L { row.append(Int.random(in: 0..<V)) }
            ids.append(row)
            // shift left by 1 for next-token targets
            var tgt = Array(row.dropFirst()); tgt.append(row.first!)
            targets.append(tgt)
        }
        var zData: [Float] = []
        for _ in 0..<(B*D) { zData.append(Float.random(in: -0.5...0.5)) }
        let zt = Tensor(shape: [B, D], data: zData)
        // Forward before update
        let (flat0, logits0, _) = dec.forwardForTraining(ids: ids, z: zt)
        let ce0 = CrossEntropyLoss.meanLogits(logits: logits0, targets: targets)
        // Compute dLogits and projection gradients (outProj only)
        let dLogitsFlat = CrossEntropyLoss.gradLogits(logits: logits0, targets: targets) // [B*L, V]
        // Get params and step optimizer
        let (w0, b0) = dec.getOutProjParams()
        var params: [Tensor] = [w0]
        if let b = b0 { params.append(b) }
        let opt = AdamW(lr: 5e-2, beta1: 0.0, beta2: 0.0, eps: 1e-8, weightDecay: 0.0)
        // Gradients via GraphLinear
        var gl = GraphLinear(inFeatures: cfg.dim, outFeatures: cfg.vocabSize, bias: b0 != nil, seed: 0)
        gl.weight = w0
        gl.bias = b0
        _ = try? gl.forward(Tensor.zeros([1, cfg.dim]))
        let (dW, dB) = try gl.gradientsGPU(X: flat0, dY: dLogitsFlat)
        // Apply step
        var pack = params
        var grads: [Tensor] = [dW]
        if let db = dB as Tensor? { grads.append(db) }
        opt.step(params: &pack, grads: grads)
        // Write back and invalidate cache
        let newW = pack[0]
        let newB = (b0 != nil) ? pack[1] : nil
        dec.setOutProjParams(weight: newW, bias: newB)
        dec.invalidateOutProjCache()
        // Forward after update
        let logits1 = dec.forward(ids: ids, z: zt)
        let ce1 = CrossEntropyLoss.meanLogits(logits: logits1, targets: targets)
        XCTAssertLessThan(ce1, ce0, "CE should decrease after projection-only update")
    }
}
