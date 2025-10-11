import XCTest
@testable import EFCore
@testable import PyTorchSwift

final class LNExecCacheTests: XCTestCase {
    func testLayerNormForwardParityGPUActor() async throws {
        let N = 4, D = 8
        let x = Tensor.randomUniform([N, D], min: -1, max: 1, seed: 11)
        var ln = LayerNorm(dim: D)
        for i in 0..<D { ln.gamma.data[i] = 0.8 + 0.05 * Float(i) }
        for i in 0..<D { ln.beta.data[i] = (i % 3 == 0) ? 0.02 : -0.01 }
        let cpu = ln.forward(x)
        let gpu = try await GPU.shared.layerNormForward(x: x, gamma: ln.gamma, beta: ln.beta, eps: ln.eps)
        XCTAssertEqual(cpu.shape, gpu.shape)
        for i in 0..<(N*D) {
            XCTAssertLessThan(abs(cpu.data[i] - gpu.data[i]), 1e-3)
        }
    }
}
