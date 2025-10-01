import XCTest
@testable import EFCore
@testable import PyTorchSwift

final class GraphLinearFP16Tests: XCTestCase {
    func testGraphLinearFP16CloseToCPU() throws {
        // Gate by env to avoid crashes on unsupported SDK/toolchain combos
        let env = ProcessInfo.processInfo.environment
        guard env["EF_ENABLE_FP16_TEST"] == "1" else {
            throw XCTSkip("FP16 test disabled. Set EF_ENABLE_FP16_TEST=1 to run.")
        }
        let B = 3, In = 5, Out = 7
        let seed: UInt64 = 123
        let x = Tensor.randomUniform([B, In], min: -1, max: 1, seed: seed)

        var cpu = Linear(inFeatures: In, outFeatures: Out, bias: true, seed: seed &+ 1)
        var gpu = GraphLinear(inFeatures: In, outFeatures: Out, bias: true, seed: seed &+ 1)
        gpu.weight = cpu.weight
        gpu.bias = cpu.bias

        let yCPU = cpu.forward(x)
        let yGPU = try gpu.forward(x)

        XCTAssertEqual(yCPU.shape, yGPU.shape)
        // FP16 introduces rounding; keep tolerance looser
        let tol: Float = 3e-4
        for i in 0..<yCPU.data.count {
            XCTAssertEqual(yCPU.data[i], yGPU.data[i], accuracy: tol, "Mismatch at index \(i)")
        }
    }
}