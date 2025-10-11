import XCTest
@testable import EFCore
@testable import EnergyFlow

final class GraphLinearTrainingStepTests: XCTestCase {
    func testSingleStepReducesMSEOnToyData() async throws {
        // Toy data: X [B, In], target T [B, Out]
        let B = 4, In = 5, Out = 3
        // Build a fixed X and a hidden true W/b to generate T
        let X = Tensor.randomUniform([B, In], min: -0.5, max: 0.5, seed: 123)
        let trueW = Tensor.randomUniform([Out, In], min: -0.1, max: 0.1, seed: 888)
        let trueB = Tensor.randomUniform([Out], min: -0.01, max: 0.01, seed: 999)
        // T = X @ trueW^T + trueB
        var T = Tensor.zeros([B, Out])
        for b in 0..<B {
            for o in 0..<Out {
                var acc: Float = 0
                for i in 0..<In { acc += X.data[b*In + i] * trueW.data[o*In + i] }
                T.data[b*Out + o] = acc + trueB.data[o]
            }
        }
        // Model: GraphLinear with random weights
        var gl = GraphLinear(inFeatures: In, outFeatures: Out, bias: true, seed: 42)
        // Forward
        let Y0 = try await gl.forwardAsync(X)
        let mse0 = Losses.mseRowwise(Y0, T).mean
        // Backward (MSE-only) and one AdamW step
        let dY = dY_MSEMean(y: Y0, target: T)
        let (dW, dB) = gradsGraphLinear(X: X, dY: dY, outFeatures: Out, inFeatures: In)
        let opt = AdamW(lr: 1e-2, beta1: 0.0, beta2: 0.0, eps: 1e-8, weightDecay: 0.0) // moderate LR for stable decrease
        var params: [Tensor] = [gl.weight, gl.bias ?? Tensor.zeros([Out])]
        let grads: [Tensor] = [dW, dB]
        opt.step(params: &params, grads: grads)
        gl.weight = params[0]
        if gl.bias != nil { gl.bias = params[1] }
        gl.invalidateCache()
        // Forward again
        let Y1 = try await gl.forwardAsync(X)
        let mse1 = Losses.mseRowwise(Y1, T).mean
        XCTAssertLessThan(mse1, mse0, String(format: "mse1=%.6f should be < mse0=%.6f", mse1, mse0))
    }
}
