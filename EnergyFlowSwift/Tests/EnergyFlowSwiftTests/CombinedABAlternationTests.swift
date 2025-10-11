import XCTest
@testable import EnergyFlow
@testable import EFCore

final class CombinedABAlternationTests: XCTestCase {
    func test_modeA_then_modeB_runs_and_improves_metrics() async throws {
        // Encoder config
        let encCfg = TextToCubeEncoderConfig(hiddenDim: 32,
                                             maxLength: 8,
                                             outputDim: 32,
                                             useTanhOutput: false,
                                             tcnBlocks: 2,
                                             kernelSize: 3,
                                             dilationSchedule: [1,2],
                                             ffDim: 64,
                                             useGPUProjection: true,
                                             baseSeed: 123)
        // Decoder config (dim must equal outputDim)
        let decCfg = TextDecoderConfig(vocabSize: 100,
                                       dim: 32,
                                       hidden: 64,
                                       nBlocks: 2,
                                       kernelSize: 3,
                                       dilationSchedule: [1,2],
                                       maxLength: 8)
        let trainer = CombinedTrainer(encConfig: encCfg, decConfig: decCfg, lrEncProj: 5e-2, alphaCos: 1.0, betaMSE: 1.0)
        let B = 4, L = encCfg.maxLength, V = decCfg.vocabSize, D = encCfg.outputDim
        // Synthetic token batch and teacher embeddings/text targets
        var ids: [[Int]] = []
        var mask: [[Int]] = []
        var targets: [[Int]] = []
        for _ in 0..<B {
            var row: [Int] = []
            for _ in 0..<L { row.append(Int.random(in: 0..<V)) }
            ids.append(row)
            mask.append(Array(repeating: 1, count: L))
            var t = Array(row.dropFirst()); t.append(row.first!)
            targets.append(t)
        }
        var zData: [Float] = []
        for _ in 0..<(B*D) { zData.append(Float.random(in: -0.2...0.2)) }
        let zt = Tensor(shape: [B, D], data: zData)
        // Metrics before
        let encOut0 = try await trainer.enc.forwardForTraining(inputIDs: ids,
                                                               attentionMask: mask)
        let mse0 = Losses.mseRowwise(encOut0.out, zt).mean
        let cos0 = Losses.cosineSimilarityRowwise(encOut0.out, zt).mean
        let decLogits0 = try await trainer.decTrainer.decoder.forward(ids: ids, z: zt, on: trainer.gpu)
        let ce0 = CrossEntropyLoss.meanLogits(logits: decLogits0, targets: targets)
        // One A step (projection-only + last block)
        _ = try await trainer.stepA(inputIDs: ids,
                                    attentionMask: mask,
                                    zTeacher: zt,
                                    unfreezeLastTCN: true)
        // Re-evaluate encoder immediately after Mode A
        let encOutA = try await trainer.enc.forwardForTraining(inputIDs: ids,
                                                               attentionMask: mask)
        let mseA1 = Losses.mseRowwise(encOutA.out, zt).mean
        let cosA1 = Losses.cosineSimilarityRowwise(encOutA.out, zt).mean
        XCTAssertLessThan(mseA1, mse0, "MSE should decrease after Mode A step")
        XCTAssertGreaterThan(cosA1, cos0, "Cosine should increase after Mode A step")
        // One B step (projection-only + last block)
        let ceB = try await trainer.stepB(ids: ids,
                                          targets: targets,
                                          zTeacher: zt,
                                          unfreezeLastTCN: true)
        // Assertions: CE decreased after Mode B
        XCTAssertLessThan(ceB, ce0, "CE should decrease after Mode B step")
    }
}
