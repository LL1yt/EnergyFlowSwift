import XCTest
@testable import EnergyFlow
@testable import EFCore
@testable import PyTorchSwift

final class TextDecoderShapeTests: XCTestCase {
    func testDecoderForwardShapes() async throws {
        let cfg = TextDecoderConfig(vocabSize: 123, dim: 32, hidden: 64, nBlocks: 2, kernelSize: 3, dilationSchedule: [1,2], maxLength: 8)
        let dec = TextDecoder(config: cfg, seed: 777)
        let B = 3, L = cfg.maxLength, V = cfg.vocabSize
        // Fake ids [B,L]
        let ids: [[Int]] = (0..<B).map { _ in (0..<L).map { _ in Int.random(in: 0..<V) } }
        // Fake z [B,dim]
        var zData: [Float] = []
        for _ in 0..<(B*cfg.dim) { zData.append(Float.random(in: -0.5...0.5)) }
        let z = Tensor(shape: [B, cfg.dim], data: zData)
        let logits = try await dec.forward(ids: ids, z: z)
        XCTAssertEqual(logits.shape, [B, L, V])
    }
}
