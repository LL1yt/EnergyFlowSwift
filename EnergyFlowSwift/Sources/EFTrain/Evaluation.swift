import Foundation
import EFCore
import EnergyFlow

// Evaluate encoder on a sample set with batching and micro-batching.
public func evaluate(enc: TextToCubeEncoder,
                     samples: [JSONLSample],
                     batchSize: Int,
                     microBatch: Int,
                     maxLen: Int,
                     on gpu: GPUActor = GPU.shared) async throws -> (Double, Double) {
    var ptr = 0
    var totalMSE: Double = 0
    var totalCos: Double = 0
    var seen = 0
    while ptr < samples.count {
        let end = min(ptr + batchSize, samples.count)
        let slice = Array(samples[ptr..<end])
        ptr = end
        let hasTokens = slice.allSatisfy { $0.inputIDs != nil && $0.attentionMask != nil }
        if hasTokens {
            var ids: [[Int]] = []
            var mask: [[Int]] = []
            var tgts: [[Float]] = []
            for s in slice { ids.append(s.inputIDs!); mask.append(s.attentionMask!); tgts.append(s.target) }
            let B = ids.count
            let micro = microBatch > 0 ? microBatch : B
            var offset = 0
            let numChunks = (B + micro - 1) / micro
            for _ in 0..<numChunks {
                let take = min(micro, B - offset)
                let idsChunk = Array(ids[offset..<(offset+take)])
                let maskChunk = Array(mask[offset..<(offset+take)])
                let tgtChunk = Array(tgts[offset..<(offset+take)])
                let out = try await enc.encodeTokens(inputIDs: idsChunk,
                                                     attentionMask: maskChunk,
                                                     on: gpu)
                let b = out.shape[0]; let d = out.shape[1]
                var pred = Array(repeating: Array(repeating: 0.0 as Float, count: d), count: b)
                for bi in 0..<b { for di in 0..<d { pred[bi][di] = out.data[bi*d + di] } }
                let m = batchMSE(pred, tgtChunk)
                let c = batchCosine(pred, tgtChunk)
                totalMSE += Double(m) * Double(b)
                totalCos += Double(c) * Double(b)
                seen += b
                offset += take
            }
        } else {
            let texts = slice.map { $0.text ?? "" }
            let tgts = slice.map { $0.target }
            let B = texts.count
            let micro = microBatch > 0 ? microBatch : B
            var offset = 0
            let numChunks = (B + micro - 1) / micro
            for _ in 0..<numChunks {
                let take = min(micro, B - offset)
                let txtChunk = Array(texts[offset..<(offset+take)])
                let tgtChunk = Array(tgts[offset..<(offset+take)])
                let out = try await enc.encode(txtChunk, on: gpu)
                let b = out.shape[0]; let d = out.shape[1]
                var pred = Array(repeating: Array(repeating: 0.0 as Float, count: d), count: b)
                for bi in 0..<b { for di in 0..<d { pred[bi][di] = out.data[bi*d + di] } }
                let m = batchMSE(pred, tgtChunk)
                let c = batchCosine(pred, tgtChunk)
                totalMSE += Double(m) * Double(b)
                totalCos += Double(c) * Double(b)
                seen += b
                offset += take
            }
        }
    }
    let avgMSE = seen > 0 ? totalMSE / Double(seen) : 0
    let avgCos = seen > 0 ? totalCos / Double(seen) : 0
    return (avgMSE, avgCos)
}

public func batchMSE(_ y: [[Float]], _ t: [[Float]]) -> Float {
    precondition(y.count == t.count)
    var acc: Float = 0
    for i in 0..<y.count { acc += mse(y[i], t[i]) }
    return acc / Float(max(y.count, 1))
}

public func mse(_ y: [Float], _ t: [Float]) -> Float {
    precondition(y.count == t.count)
    var acc: Float = 0
    for i in 0..<y.count { let d = y[i] - t[i]; acc += d*d }
    return acc / Float(y.count)
}

public func cosine(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    var dot: Float = 0
    var na: Float = 0
    var nb: Float = 0
    for i in 0..<a.count { dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
    let denom = sqrt(max(na, 1e-12)) * sqrt(max(nb, 1e-12))
    return denom > 0 ? dot / denom : 0
}

public func batchCosine(_ y: [[Float]], _ t: [[Float]]) -> Float {
    precondition(y.count == t.count)
    var acc: Float = 0
    for i in 0..<y.count { acc += cosine(y[i], t[i]) }
    return acc / Float(max(y.count, 1))
}
