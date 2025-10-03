import XCTest
@testable import EnergyFlow
@testable import EFCore

final class CosineGradTests: XCTestCase {
    func lossOneMinusMeanCos(_ y: Tensor, _ t: Tensor) -> Float {
        let cos = Losses.cosineSimilarityRowwise(y, t)
        return 1.0 - cos.mean
    }

    func testCosineGradReducesLossOnSmallStep() throws {
        // Random small batch
        let B = 3, D = 7
        let y0 = Tensor.randomUniform([B, D], min: -0.5, max: 0.5, seed: 101)
        let t = Tensor.randomUniform([B, D], min: -0.5, max: 0.5, seed: 202)
        var y = y0
        let L0 = lossOneMinusMeanCos(y, t)
        // Gradient step with small lr
        let dy = dY_CosineMeanLoss(y: y, target: t)
        let lr: Float = 1e-1
        for i in 0..<y.count { y.data[i] -= lr * dy.data[i] }
        let L1 = lossOneMinusMeanCos(y, t)
        XCTAssertLessThan(L1, L0, String(format: "L1=%.6f should be < L0=%.6f", L1, L0))
    }
}
