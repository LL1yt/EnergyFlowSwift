import XCTest
@testable import EFCore
@testable import PyTorchSwift

final class GraphLinearBenchTests: XCTestCase {
    func testGraphLinearMicroBench() throws {
        let env = ProcessInfo.processInfo.environment
        guard env["EF_ENABLE_BENCH"] == "1" else {
            throw XCTSkip("Micro-bench disabled. Set EF_ENABLE_BENCH=1 to run.")
        }
        // Sizes (override via env if needed)
        let B = Int(env["EF_BENCH_B"] ?? "64") ?? 64
        let In = Int(env["EF_BENCH_IN"] ?? "256") ?? 256
        let Out = Int(env["EF_BENCH_OUT"] ?? "256") ?? 256
        let iters = Int(env["EF_BENCH_ITERS"] ?? "100") ?? 100
        let warmup = Int(env["EF_BENCH_WARMUP"] ?? "10") ?? 10
        let seed: UInt64 = 123

        let logger = Logger.shared
        logger.info("Bench sizes: B=\(B) In=\(In) Out=\(Out) warmup=\(warmup) iters=\(iters)", category: Logger.Category.training)

        // Data
        let x = Tensor.randomUniform([B, In], min: -1, max: 1, seed: seed)
        var cpu = Linear(inFeatures: In, outFeatures: Out, bias: true, seed: seed &+ 1)
        var gl = GraphLinear(inFeatures: In, outFeatures: Out, bias: true, seed: seed &+ 1)
        gl.weight = cpu.weight; gl.bias = cpu.bias

        func time(_ label: String, _ body: () throws -> Void) rethrows -> Double {
            let t0 = CFAbsoluteTimeGetCurrent()
            try body()
            let t1 = CFAbsoluteTimeGetCurrent()
            return (t1 - t0)
        }

        // Warmup
        for _ in 0..<warmup { _ = cpu.forward(x) }
        for _ in 0..<warmup { _ = try gl.forward(x) }

        // Measure CPU
        let cpuT = try time("cpu") {
            for _ in 0..<iters { _ = cpu.forward(x) }
        }
        // Measure GPU fp32
        let gpuT = try time("gpu16") {
            for _ in 0..<iters { _ = try gl.forward(x) }
        }

        let perCPU = cpuT / Double(iters)
        let perG = gpuT / Double(iters)
        logger.info(String(format: "AVG per iter: CPU=%.6f s  GPU16=%.6f s", perCPU, perG), category: Logger.Category.training)
        logger.info(String(format: "Speedup: GPU16 x%.2f", perCPU/perG), category: Logger.Category.training)

        // Sanity: forms equal
        let yCPU = cpu.forward(x)
        let yG = try gl.forward(x)
        XCTAssertEqual(yCPU.shape, yG.shape)
    }
}