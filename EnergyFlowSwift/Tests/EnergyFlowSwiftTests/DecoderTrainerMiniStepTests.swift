import XCTest
@testable import EnergyFlow
@testable import EFCore
@testable import PyTorchSwift

final class DecoderTrainerMiniStepTests: XCTestCase {
    func test_projection_only_step_decreases_CE() throws {
        let cfg = TextDecoderConfig(vocabSize: 80, dim: 32, hidden: 64, nBlocks: 2, kernelSize: 3, dilationSchedule: [1,2], maxLength: 8)
        let trainer = DecoderTrainer(config: cfg, lr: 5e-2, weightDecay: 0.0, seed: 42)
        let B = 6, L = cfg.maxLength, V = cfg.vocabSize, D = cfg.dim
        // Synthetic ids and next-token targets
        var ids: [[Int]] = []
        var tgts: [[Int]] = []
        for _ in 0..<B {
            var row: [Int] = []
            for _ in 0..<L { row.append(Int.random(in: 0..<V)) }
            ids.append(row)
            var t = Array(row.dropFirst()); t.append(row.first!)
            tgts.append(t)
        }
        var zData: [Float] = []
        for _ in 0..<(B*D) { zData.append(Float.random(in: -0.5...0.5)) }
        let zt = Tensor(shape: [B, D], data: zData)
        let ce0 = try trainer.step(ids: ids, zTeacher: zt, targets: tgts)
        let ce1 = try trainer.step(ids: ids, zTeacher: zt, targets: tgts)
        XCTAssertLessThan(ce1, ce0, "CE should decrease after two projection-only steps")
    }
}
