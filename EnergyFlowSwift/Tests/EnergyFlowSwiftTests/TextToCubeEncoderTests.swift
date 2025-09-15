import XCTest
@testable import EnergyFlow

final class TextToCubeEncoderTests: XCTestCase {
    func testOutputShapeAndRange() {
        let config = createDebugConfig() // 20x20x10 -> surfaceDim = 400
        let encoder = TextToCubeEncoder(config: config)
        let texts = ["hello world", "swift mps metal"]
        let out = encoder.encode(texts, maxLength: 16)
        XCTAssertEqual(out.shape, [2, config.surfaceDim])
        // values should be in [-1, 1] after tanh
        var minVal: Float = Float.greatestFiniteMagnitude
        var maxVal: Float = -Float.greatestFiniteMagnitude
        for v in out.data { minVal = min(minVal, v); maxVal = max(maxVal, v) }
        XCTAssertGreaterThanOrEqual(minVal, -1.0001)
        XCTAssertLessThanOrEqual(maxVal, 1.0001)
    }
}
