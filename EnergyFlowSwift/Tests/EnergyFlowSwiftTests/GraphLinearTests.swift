import XCTest
@testable import EFCore
@testable import PyTorchSwift

final class GraphLinearTests: XCTestCase {
    func testGraphLinearMatchesCPU() throws {
        // Simple matrix multiply + bias on GPU vs CPU reference
        let B = 3, In = 5, Out = 7
        let seed: UInt64 = 123
        let x = Tensor.randomUniform([B, In], min: -1, max: 1, seed: seed)

        var cpu = Linear(inFeatures: In, outFeatures: Out, bias: true, seed: seed &+ 1)
        var gpu = GraphLinear(inFeatures: In, outFeatures: Out, bias: true, seed: seed &+ 1)
        // Share the same parameters to ensure equal outputs
        gpu.weight = cpu.weight
        gpu.bias = cpu.bias

        let yCPU = cpu.forward(x)
        let yGPU = try gpu.forward(x)

        XCTAssertEqual(yCPU.shape, yGPU.shape)
        let tol: Float = 1e-5
        for i in 0..<yCPU.data.count {
            XCTAssertEqual(yCPU.data[i], yGPU.data[i], accuracy: tol, "Mismatch at index \(i)")
        }
    }
}