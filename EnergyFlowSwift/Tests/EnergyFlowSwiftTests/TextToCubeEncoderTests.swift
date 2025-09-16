import XCTest
@testable import EnergyFlow

final class TextToCubeEncoderTests: XCTestCase {
    func testOutputShapeAndRange() {
let energy = createDebugConfig() // 20x20x10
let model = TextToCubeEncoderConfig(hiddenDim: 256, maxLength: 16, maxPosition: 64, outputDim: 768, useTanhOutput: false)
let encoder = TextToCubeEncoder(energyConfig: energy, modelConfig: model)
let texts = ["hello world", "swift mps metal"]
let out = encoder.encode(texts)
XCTAssertEqual(out.shape, [2, model.outputDim])
        // values should be in [-1, 1] after tanh
        var minVal: Float = Float.greatestFiniteMagnitude
        var maxVal: Float = -Float.greatestFiniteMagnitude
        for v in out.data { minVal = min(minVal, v); maxVal = max(maxVal, v) }
        XCTAssertGreaterThanOrEqual(minVal, -1.0001)
        XCTAssertLessThanOrEqual(maxVal, 1.0001)
    }
}
