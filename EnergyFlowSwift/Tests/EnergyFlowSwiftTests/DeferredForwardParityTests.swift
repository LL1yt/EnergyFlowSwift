import XCTest
@testable import EnergyFlow
@testable import EFCore

final class DeferredForwardParityTests: XCTestCase {
    func testEncoderDeferredForwardParity() async throws {
        // Enable verbose logging to stdout for this test run
        Logger.shared.enabled = true
        Logger.shared.mirrorToStdout = true
        Logger.shared.level = .debug
        Logger.shared.info("Encoder deferred forward parity test: start", category: Logger.Category.training)
        let encCfg = TextToCubeEncoderConfig(hiddenDim: 32,
                                             maxLength: 8,
                                             outputDim: 32,
                                             tcnBlocks: 2,
                                             kernelSize: 3,
                                             dilationSchedule: [1,2],
                                             ffDim: 64)
        let enc = TextToCubeEncoder(modelConfig: encCfg)
        let gpu = GPU.shared
        // Build tiny batch of tokens
        let B = 3, L = encCfg.maxLength
        var ids: [[Int]] = []
        var mask: [[Int]] = []
        for _ in 0..<B {
            var row: [Int] = []
            for _ in 0..<L { row.append(Int.random(in: 0..<50)) }
            ids.append(row)
            mask.append(Array(repeating: 1, count: L))
        }
        Logger.shared.info("Encoder: running immediate forward", category: Logger.Category.training)
        // Immediate forward
        let imm = try await enc.forwardForTrainingWithLastBlockCache(inputIDs: ids, attentionMask: mask, on: gpu)
        Logger.shared.info("Encoder: running deferred forward", category: Logger.Category.training)
        // Deferred forward (no batching to avoid deadlock with immediate GPU ops inside)
        let def = try await enc.forwardForTrainingWithLastBlockCacheDeferred(inputIDs: ids, attentionMask: mask, on: gpu)
        await gpu.syncBatch(label: "encoder.deferred.forward")
        Logger.shared.info("Encoder: deferred forward synced, awaiting readbacks", category: Logger.Category.training)
        let pooledDef = try await def.pooledRB.value()
        let outDef = try await def.outRB.value()
        Logger.shared.info("Encoder: readbacks resolved; comparing parity", category: Logger.Category.training)
        // Compare shapes and basic parity
        XCTAssertEqual(imm.pooled.shape, pooledDef.shape)
        XCTAssertEqual(imm.out.shape, outDef.shape)
        // Numeric tolerance (loose check)
        func meanAbsDiff(_ a: Tensor, _ b: Tensor) -> Float {
            precondition(a.count == b.count)
            var acc: Float = 0
            for i in 0..<a.count { acc += abs(a.data[i] - b.data[i]) }
            return acc / Float(max(1, a.count))
        }
        XCTAssertLessThan(meanAbsDiff(imm.pooled, pooledDef), 1e-4)
        XCTAssertLessThan(meanAbsDiff(imm.out, outDef), 1e-4)
    }

    func testDecoderDeferredForwardParity() async throws {
        Logger.shared.info("Decoder deferred forward parity test: start", category: Logger.Category.training)
        let decCfg = TextDecoderConfig(vocabSize: 100,
                                       dim: 32,
                                       hidden: 64,
                                       nBlocks: 2,
                                       kernelSize: 3,
                                       dilationSchedule: [1,2],
                                       maxLength: 8)
        let dec = TextDecoder(config: decCfg)
        let gpu = GPU.shared
        // Build tiny batch of tokens and z
        let B = 3, L = decCfg.maxLength, D = decCfg.dim
        var ids: [[Int]] = []
        for _ in 0..<B {
            var row: [Int] = []
            for _ in 0..<L { row.append(Int.random(in: 0..<decCfg.vocabSize)) }
            ids.append(row)
        }
        var zData: [Float] = []
        zData.reserveCapacity(B*D)
        for _ in 0..<(B*D) { zData.append(Float.random(in: -0.1...0.1)) }
        let zt = Tensor(shape: [B, D], data: zData)
        Logger.shared.info("Decoder: running immediate forward", category: Logger.Category.training)
        // Immediate forward
        let imm = try await dec.forwardForTraining(ids: ids, z: zt, on: gpu)
        Logger.shared.info("Decoder: running deferred forward", category: Logger.Category.training)
        // Deferred forward (no batching to avoid deadlock with immediate GPU ops inside)
        let def = try await dec.forwardForTrainingDeferred(ids: ids, z: zt, on: gpu)
        await gpu.syncBatch(label: "decoder.deferred.forward")
        Logger.shared.info("Decoder: deferred forward synced, awaiting readbacks", category: Logger.Category.training)
        let flatDef = try await def.flatRB.value()
        let logitsDef = try await def.logitsRB.value()
        Logger.shared.info("Decoder: readbacks resolved; comparing parity", category: Logger.Category.training)
        // Compare shapes and parity
        XCTAssertEqual(imm.flatFeatures.shape, flatDef.shape)
        XCTAssertEqual(imm.logits.shape, logitsDef.shape)
        // Numeric tolerance
        func meanAbsDiff(_ a: Tensor, _ b: Tensor) -> Float {
            precondition(a.count == b.count)
            var acc: Float = 0
            for i in 0..<a.count { acc += abs(a.data[i] - b.data[i]) }
            return acc / Float(max(1, a.count))
        }
        XCTAssertLessThan(meanAbsDiff(imm.flatFeatures, flatDef), 1e-4)
        XCTAssertLessThan(meanAbsDiff(imm.logits, logitsDef), 1e-4)
    }
}
