import XCTest
@testable import EFCore
@testable import EnergyFlow

final class GraphLinearGPUGradParityTests: XCTestCase {
    func testGPUvsCPUGradParity() throws {
        let B = 16, In = 32, Out = 24
        // Random X and dY
        let X = Tensor.randomUniform([B, In], min: -0.5, max: 0.5, seed: 2024)
        let dY = Tensor.randomUniform([B, Out], min: -0.5, max: 0.5, seed: 42)
        let gl = GraphLinear(inFeatures: In, outFeatures: Out, bias: true, seed: 7)
        // CPU grads
        let (dWcpu, dBcpu) = gradsGraphLinear(X: X, dY: dY, outFeatures: Out, inFeatures: In)
        // GPU grads
        let (dWgpu, dBgpu) = try gl.gradientsGPU(X: X, dY: dY)
        // Compare with relaxed tolerance for FP16
        func maxAbsDiff(_ a: Tensor, _ b: Tensor) -> Float {
            precondition(a.shape == b.shape)
            var m: Float = 0
            for i in 0..<a.count { m = max(m, abs(a.data[i] - b.data[i])) }
            return m
        }
        let tol: Float = 3e-2
        XCTAssertLessThan(maxAbsDiff(dWcpu, dWgpu), tol)
        XCTAssertLessThan(maxAbsDiff(dBcpu, dBgpu), tol)
    }
}
