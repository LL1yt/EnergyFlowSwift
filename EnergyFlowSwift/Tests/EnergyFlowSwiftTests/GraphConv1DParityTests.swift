import XCTest
@testable import EFCore
@testable import PyTorchSwift

final class GraphConv1DParityTests: XCTestCase {
    func maxAbsDiff(_ a: Tensor, _ b: Tensor) -> Float {
        precondition(a.shape == b.shape)
        var m: Float = 0
        for i in 0..<a.count { m = max(m, abs(a.data[i] - b.data[i])) }
        return m
    }

    func testGPUvsCPUParitySmallShapes() throws {
        let B = 2, L = 16, Cin = 8, Cout = 12
        let kernelSizes = [3, 5]
        let dilations = [1, 2]
        for k in kernelSizes {
            for d in dilations {
                // Inputs
                let x = Tensor.randomUniform([B, L, Cin], min: -0.5, max: 0.5, seed: 12345)
                // CPU reference conv
                let cpu = Conv1D(inChannels: Cin, outChannels: Cout, kernelSize: k, dilation: d, bias: true, seed: 777)
                let yCPU = cpu.forward(x)
                // GPU conv: copy weights/bias from CPU to ensure identical params
                let gpu = GraphConv1D(inChannels: Cin, outChannels: Cout, kernelSize: k, dilation: d, bias: true, seed: 999)
                gpu.weight = cpu.weight
                gpu.bias = cpu.bias
                let yGPU = gpu.forward(x)
                let diff = maxAbsDiff(yCPU, yGPU)
                // FP16 path has quantization; allow relaxed tolerance
                XCTAssertLessThan(diff, 3e-2, "maxAbsDiff=\(diff) for k=\(k) d=\(d)")
            }
        }
    }
}
