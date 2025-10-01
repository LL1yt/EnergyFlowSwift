import XCTest
@testable import EnergyFlow
@testable import EFCore

final class TCNEncoderSmokeTests: XCTestCase {
    func testTCNPathShapes() throws {
        // Small config
        let cfg = TextToCubeEncoderConfig(
            hiddenDim: 64,
            maxLength: 16,
            maxPosition: 32,
            outputDim: 32,
            useTanhOutput: false,
            useTCN: true,
            tcnBlocks: 2,
            kernelSize: 3,
            dilationSchedule: [1,2],
            numLayers: 1,
            numHeads: 1,
            ffDim: 64,
            dropout: 0.0,
            useGPUProjection: false,
            baseSeed: 7
        )
        let enc = TextToCubeEncoder(modelConfig: cfg, vocabSize: 100)
        let texts = ["hello world", "swift metal"]
        let y = enc.encode(texts)
        XCTAssertEqual(y.shape, [2, 32])
    }
}