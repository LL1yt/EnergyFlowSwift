import XCTest
@testable import EnergyFlow

final class TextBridgeSmokeTests: XCTestCase {
    func testEncodeShapeAndFinite() {
        // Given
        let cfg = TextToCubeEncoderConfig( // defaults: hiddenDim=768, outputDim=768
            hiddenDim: 768,
            maxLength: 8,
            maxPosition: 64,
            outputDim: 768,
            useTanhOutput: false,
            numLayers: 1,    // keep tiny for smoke
            numHeads: 8,
            ffDim: 512,
            dropout: 0.0,
            baseSeed: 42
        )
        let enc = TextToCubeEncoder(energyConfig: createDebugConfig(), modelConfig: cfg, vocabSize: 1000)

        let texts = ["hello world", "swift transformer smoke test"]

        // When
        let out = enc.encode(texts)

        // Then
        XCTAssertEqual(out.shape.count, 2)
        XCTAssertEqual(out.shape[0], texts.count)
        XCTAssertEqual(out.shape[1], cfg.outputDim)
        // Ensure all values are finite
        for v in out.data {
            XCTAssert(v.isFinite, "Found non-finite value in output")
        }
    }
}
