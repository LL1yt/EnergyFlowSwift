import XCTest
@testable import EFCore
@testable import PyTorchSwift

final class LNGeLUGemmCacheTests: XCTestCase {
    // Compare CPU (LayerNorm.forward -> GELU -> CPU matmul) vs MPSGraph cache
    func testForwardParitySmall() throws {
        let N = 4, D = 8, Out = 5
        // Random inputs
        let x = Tensor.randomUniform([N, D], min: -1, max: 1, seed: 1)
        let ln = LayerNorm(dim: D)
        var gamma = ln.gamma
        var beta = ln.beta
        // Perturb gamma/beta slightly to avoid trivial identity
        for i in 0..<D { gamma.data[i] = 0.9 + 0.2 * Float(i % 3) / 3.0 }
        for i in 0..<D { beta.data[i] = (i % 2 == 0) ? 0.01 : -0.02 }
        let W = Tensor.randomUniform([Out, D], min: -0.5, max: 0.5, seed: 2)
        let b = Tensor.randomUniform([Out], min: -0.1, max: 0.1, seed: 3)
        // CPU path: LN -> GELU -> matmul + bias
        func cpuForward(_ x: Tensor, gamma: Tensor, beta: Tensor, W: Tensor, b: Tensor) -> Tensor {
            precondition(x.shape == [N, D] && gamma.shape == [D] && beta.shape == [D])
            precondition(W.shape == [Out, D] && b.shape == [Out])
            // LN per row
            var y = Tensor.zeros([N, D])
            for n in 0..<N {
                let base = n * D
                var mean: Float = 0
                for j in 0..<D { mean += x.data[base + j] }
                mean /= Float(D)
                var varAcc: Float = 0
                for j in 0..<D { let d = x.data[base + j] - mean; varAcc += d*d }
                let invStd: Float = 1.0 / Float(sqrt(Double(varAcc / Float(D) + 1e-5)))
                for j in 0..<D {
                    let norm = (x.data[base + j] - mean) * invStd
                    y.data[base + j] = norm * gamma.data[j] + beta.data[j]
                }
            }
            // GELU approx
            let c: Float = 0.7978845608028654
            var yG = Tensor.zeros([N, D])
            for i in 0..<(N*D) {
                let v = y.data[i]
                let u = c * (v + 0.044715 * v * v * v)
                let t = Float(Darwin.tanh(Double(u)))
                yG.data[i] = 0.5 * v * (1 + t)
            }
            // Matmul yG [N,D] with WT [D,Out]
            var out = Tensor.zeros([N, Out])
            for n in 0..<N {
                for o in 0..<Out {
                    var acc: Float = 0
                    for d in 0..<D {
                        acc += yG.data[n*D + d] * W.data[o*D + d]
                    }
                    out.data[n*Out + o] = acc + b.data[o]
                }
            }
            return out
        }
        let cpuY = cpuForward(x, gamma: gamma, beta: beta, W: W, b: b)
        let gpuY = try LNGeLUGemmCache.shared.runForward(x: x, gamma: gamma, beta: beta, W: W, bias: b, eps: 1e-5)
        XCTAssertEqual(cpuY.shape, gpuY.shape)
        for i in 0..<(N*Out) {
            XCTAssertLessThan(abs(cpuY.data[i] - gpuY.data[i]), 1e-4)
        }
        // Second call should reuse executable and arrays silently
        var x2 = x
        for i in 0..<(N*D) { x2.data[i] += 0.01 }
        _ = try LNGeLUGemmCache.shared.runForward(x: x2, gamma: gamma, beta: beta, W: W, bias: b, eps: 1e-5)
    }
}