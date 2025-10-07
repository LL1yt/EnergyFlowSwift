import XCTest
@testable import EFCore
@testable import PyTorchSwift

final class GPUKernelsAndStrideTests: XCTestCase {
    // CPU refs
    private func cpuMaskedMean(_ x: Tensor, mask: [[Int]]) -> Tensor {
        let B = x.shape[0], L = x.shape[1], H = x.shape[2]
        var out = Tensor.zeros([B, H])
        for b in 0..<B {
            var denom: Float = 0
            for t in 0..<L { denom += Float(mask[b][t]) }
            denom = max(denom, 1e-9)
            for t in 0..<L {
                let m = Float(mask[b][t])
                if m == 0 { continue }
                let base = (b * L + t) * H
                for h in 0..<H { out.data[b*H + h] += x.data[base + h] * m }
            }
            for h in 0..<H { out.data[b*H + h] /= denom }
        }
        return out
    }
    private func cpuGELUBackward(x: Tensor, dy: Tensor) -> Tensor {
        let c: Float = 0.7978845608028654
        var dx = Tensor.zeros(x.shape)
        for i in 0..<x.count {
            let v = x.data[i]
            let v2 = v*v
            let v3 = v2*v
            let u = c * (v + 0.044715 * v3)
            let th = Float(Darwin.tanh(Double(u)))
            let sech2 = 1 - th*th
            let du = c * (1 + 3*0.044715*v2)
            let dgelu = 0.5 * (1 + th) + 0.5 * v * sech2 * du
            dx.data[i] = dy.data[i] * dgelu
        }
        return dx
    }
    private func cpuLayerNormBackward(x: Tensor, dy: Tensor, gamma: Tensor, eps: Float = 1e-5) -> (Tensor, Tensor, Tensor) {
        let N = x.shape[0], D = x.shape[1]
        var dx = Tensor.zeros([N, D])
        var dG = Tensor.zeros([D])
        var dB = Tensor.zeros([D])
        for n in 0..<N {
            let base = n * D
            var mean: Float = 0
            for j in 0..<D { mean += x.data[base + j] }
            mean /= Float(D)
            var varAcc: Float = 0
            for j in 0..<D { let d = x.data[base + j] - mean; varAcc += d*d }
            let invStd = 1.0 / Float(sqrt(Double(varAcc / Float(D) + eps)))
            var sumG: Float = 0
            var sumGX: Float = 0
            var xhat = [Float](repeating: 0, count: D)
            for j in 0..<D {
                xhat[j] = (x.data[base + j] - mean) * invStd
                let gy = dy.data[base + j] * gamma.data[j]
                sumG += gy
                sumGX += gy * xhat[j]
                dG.data[j] += dy.data[base + j] * xhat[j]
                dB.data[j] += dy.data[base + j]
            }
            for j in 0..<D {
                let gy = dy.data[base + j] * gamma.data[j]
                let term = Float(D) * gy - sumG - xhat[j] * sumGX
                dx.data[base + j] = (invStd / Float(D)) * term
            }
        }
        return (dx, dG, dB)
    }

    func testMaskedMeanGPUParity() throws {
        let B = 3, L = 5, H = 4
        var x = Tensor.zeros([B, L, H])
        for i in 0..<x.count { x.data[i] = Float(i % 7) * 0.1 }
        var mask = Array(repeating: Array(repeating: 0, count: L), count: B)
        for b in 0..<B { for t in 0..<L { mask[b][t] = (t % 2 == 0) ? 1 : 0 } }
        let cpu = cpuMaskedMean(x, mask: mask)
        let gpu = ElementwiseGPU.maskedMean(x: x, mask: mask)
        XCTAssertEqual(cpu.shape, gpu.shape)
        for i in 0..<cpu.count { XCTAssertLessThan(abs(cpu.data[i] - gpu.data[i]), 1e-6) }
    }

    func testGraphLinearDXAugStrideAlignment() throws {
        let inF = 32, outF = 5, B = 32
        var gl = GraphLinear(inFeatures: inF, outFeatures: outF, bias: true, seed: 1)
        let dY = Tensor.randomUniform([B, outF], min: -0.1, max: 0.1, seed: 2)
        _ = try gl.inputGradientsGPU(dY: dY) // should not crash and no invalid device load
    }

    func testGraphConv1DStrideAlignment() {
        let B = 2, L = 11, Cin = 7, Cout = 5, K = 3
        var x = Tensor.randomUniform([B, L, Cin], min: -0.5, max: 0.5, seed: 3)
        let conv = GraphConv1D(inChannels: Cin, outChannels: Cout, kernelSize: K, dilation: 1, bias: true, seed: 4)
        let y = conv.forward(x)
        XCTAssertEqual(y.shape, [B, L, Cout])
        for v in y.data { XCTAssert(v.isFinite) }
    }

    func testGELUBackwardParity() {
        let N = 6, D = 5
        let x = Tensor.randomUniform([N, D], min: -1, max: 1, seed: 5)
        let dy = Tensor.randomUniform([N, D], min: -1, max: 1, seed: 6)
        let dxCPU = cpuGELUBackward(x: x, dy: dy)
        let dxGPU = GELUGPU.backward(x: x, dY: dy)
for i in 0..<(N*D) { XCTAssertLessThan(abs(dxCPU.data[i] - dxGPU.data[i]), 5e-4) }
    }

    func testLayerNormBackwardParity() {
        let N = 8, D = 7
        let x = Tensor.randomUniform([N, D], min: -1, max: 1, seed: 7)
        let g = Tensor.randomUniform([N, D], min: -1, max: 1, seed: 8)
        let gamma = Tensor.randomUniform([D], min: 0.5, max: 1.5, seed: 9)
        let (dxC, dGC, dBC) = cpuLayerNormBackward(x: x, dy: g, gamma: gamma)
        let (dxG, dGG, dBG) = LayerNormGPU.backward(x: x, g: g, gamma: gamma, eps: 1e-5)
        for i in 0..<(N*D) { XCTAssertLessThan(abs(dxC.data[i] - dxG.data[i]), 1e-3) }
        for j in 0..<D { XCTAssertLessThan(abs(dGC.data[j] - dGG.data[j]), 1e-3) }
        for j in 0..<D { XCTAssertLessThan(abs(dBC.data[j] - dBG.data[j]), 1e-3) }
    }

    func testMaskedMeanBackwardGPUParity() {
        let B = 3, L = 6, H = 4
        var dy = Tensor.randomUniform([B, H], min: -1, max: 1, seed: 10)
        var mask = Array(repeating: Array(repeating: 0, count: L), count: B)
        for b in 0..<B { for t in 0..<L { mask[b][t] = (t % 3 == 0) ? 1 : 0 } }
        // CPU ref
        var dxCPU = Tensor.zeros([B, L, H])
        for b in 0..<B {
            var denom: Float = 0
            for t in 0..<L { denom += Float(mask[b][t]) }
            denom = max(denom, 1e-9)
            for t in 0..<L {
                let m = Float(mask[b][t])
                if m == 0 { continue }
                let outBase = (b * L + t) * H
                let inBase = b * H
                for h in 0..<H { dxCPU.data[outBase + h] = dy.data[inBase + h] * (m / denom) }
            }
        }
        let dxGPU = ElementwiseGPU.maskedMeanBackward(dPooled: dy, mask: mask, seqLen: L)
        XCTAssertEqual(dxCPU.shape, dxGPU.shape)
        for i in 0..<dxCPU.count { XCTAssertLessThan(abs(dxCPU.data[i] - dxGPU.data[i]), 1e-6) }
    }
}
