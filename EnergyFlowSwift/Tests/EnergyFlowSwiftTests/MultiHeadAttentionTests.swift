import XCTest
@testable import PyTorchSwift
@testable import EFCore

final class MultiHeadAttentionTests: XCTestCase {
    func testShapesAcrossHeads() {
        let b = 2, l = 5, h = 768
        let mask = Array(repeating: Array(repeating: 1, count: l), count: b)
        // Simple deterministic input
        var data = (0..<(b*l*h)).map { _ in Float(0.01) }
        let x = Tensor(shape: [b,l,h], data: data)

        for heads in [1, 2, 4, 8] {
            let layer = TransformerEncoderLayer(hidden: h, ffDim: 512, numHeads: heads, seed: 42)
            let out = layer.forward(x, mask: mask)
            XCTAssertEqual(out.shape, [b,l,h], "Shape mismatch for heads=\(heads)")
        }
    }

    func testDeterminismSameSeed() {
        let b = 1, l = 3, h = 768
        let mask = Array(repeating: Array(repeating: 1, count: l), count: b)
        var data: [Float] = []
        data.reserveCapacity(b*l*h)
        for i in 0..<(b*l*h) { data.append(Float(i) * 0.001) }
        let x = Tensor(shape: [b,l,h], data: data)

        let layerA = TransformerEncoderLayer(hidden: h, ffDim: 256, numHeads: 8, seed: 123)
        let layerB = TransformerEncoderLayer(hidden: h, ffDim: 256, numHeads: 8, seed: 123)
        let outA = layerA.forward(x, mask: mask)
        let outB = layerB.forward(x, mask: mask)
        XCTAssertEqual(outA.shape, outB.shape)
        for i in 0..<outA.data.count {
            XCTAssertEqual(outA.data[i], outB.data[i], accuracy: 1e-6, "Mismatch at index \(i)")
        }
    }

    func testDifferentSeedsProduceDifferentOutputs() {
        let b = 1, l = 3, h = 768
        let mask = Array(repeating: Array(repeating: 1, count: l), count: b)
        let x = Tensor.randomUniform([b,l,h], min: -0.01, max: 0.01, seed: 7)
        let layerA = TransformerEncoderLayer(hidden: h, ffDim: 256, numHeads: 8, seed: 123)
        let layerB = TransformerEncoderLayer(hidden: h, ffDim: 256, numHeads: 8, seed: 124)
        let outA = layerA.forward(x, mask: mask)
        let outB = layerB.forward(x, mask: mask)
        // Count exact or near-exact equality should be small
        var sameCount = 0
        for i in 0..<outA.data.count {
            if abs(outA.data[i] - outB.data[i]) < 1e-7 { sameCount += 1 }
        }
        XCTAssertLessThan(sameCount, outA.data.count / 10, "Outputs too similar across different seeds")
    }
}
