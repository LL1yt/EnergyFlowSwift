import XCTest
@testable import EnergyFlow
@testable import EFCore

final class TextToCubeEncoderTCNEncodeTokensTests: XCTestCase {
    func testEncodeTokensTCNReturnsExpectedShape() throws {
        // Config: small TCN with GPU projection
        let cfg = TextToCubeEncoderConfig(
            hiddenDim: 64,
            maxLength: 8,
            outputDim: 32,
            useTanhOutput: false,
            tcnBlocks: 2,
            kernelSize: 3,
            dilationSchedule: [1,2],
            ffDim: 64,
            useGPUProjection: true,
            baseSeed: 123
        )
        let enc = TextToCubeEncoder(modelConfig: cfg, vocabSize: 100)
        // Pre-tokenized toy batch (ragged), will be padded to maxLength inside
        let inputIDs: [[Int]] = [
            [5, 6, 7, 8],
            [1, 2],
            [3, 4, 5]
        ]
        let attentionMask: [[Int]] = inputIDs.map { row in
            var m = Array(repeating: 0, count: row.count)
            for i in 0..<row.count { m[i] = 1 }
            return m
        }
        let y = enc.encodeTokens(inputIDs: inputIDs, attentionMask: attentionMask)
        XCTAssertEqual(y.shape, [3, 32])
    }
}