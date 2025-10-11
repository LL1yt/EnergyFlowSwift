import XCTest
@testable import EnergyFlow
@testable import EFCore

final class DeferredForwardParityTests: XCTestCase {
    func testEncoderDeferredForwardParity() async throws {
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
        // Immediate forward
        let imm = try await enc.forwardForTrainingWithLastBlockCache(inputIDs: ids, attentionMask: mask, on: gpu)
        // Deferred forward
        await gpu.beginBatch()
        let def = try await enc.forwardForTrainingWithLastBlockCacheDeferred(inputIDs: ids, attentionMask: mask, on: gpu)
        await gpu.syncBatch(label: "encoder.deferred.forward")
        let pooledDef = try await def.pooledRB.value()
        let outDef = try await def.outRB.value()
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
        // Immediate forward
        let imm = try await dec.forwardForTraining(ids: ids, z: zt, on: gpu)
        // Deferred forward
        await gpu.beginBatch()
        let def = try await dec.forwardForTrainingDeferred(ids: ids, z: zt, on: gpu)
        await gpu.syncBatch(label: "decoder.deferred.forward")
        let flatDef = try await def.flatRB.value()
        let logitsDef = try await def.logitsRB.value()
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
