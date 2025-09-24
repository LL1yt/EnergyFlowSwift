import Foundation
import EFCore

// Minimal single-head self-attention for [B, L, H]
// - Projections: Q, K, V, O via Linear
// - Mask: [[Int]] with 1 for real tokens, 0 for padding (masking keys)
// - Scaled dot-product with numerically stable softmax
public struct SingleHeadSelfAttention {
    public let hidden: Int
    public var qProj: Linear
    public var kProj: Linear
    public var vProj: Linear
    public var oProj: Linear

    public init(hidden: Int, seed: UInt64) {
        self.hidden = hidden
        self.qProj = Linear(inFeatures: hidden, outFeatures: hidden, seed: Seed.derive(seed, label: "q"))
        self.kProj = Linear(inFeatures: hidden, outFeatures: hidden, seed: Seed.derive(seed, label: "k"))
        self.vProj = Linear(inFeatures: hidden, outFeatures: hidden, seed: Seed.derive(seed, label: "v"))
        self.oProj = Linear(inFeatures: hidden, outFeatures: hidden, seed: Seed.derive(seed, label: "o"))
    }

    // x: [B, L, H], mask: [B][L] -> y: [B, L, H]
    public func forward(_ x: Tensor, mask: [[Int]]) -> Tensor {
        precondition(x.shape.count == 3 && x.shape[2] == hidden, "SingleHeadSelfAttention expects [B,L,H]")
        let b = x.shape[0]
        let l = x.shape[1]
        let h = x.shape[2]
        precondition(mask.count == b && mask.allSatisfy { $0.count == l }, "mask shape mismatch")

        Logger.shared.debug("SingleHeadSelfAttention.forward b=\(b) l=\(l) h=\(h)", category: Logger.Category.textBridge)

        // Flatten to [B*L, H] for linear projections
        let flat = x.reshaped([b * l, h])
        let q = qProj.forward(flat).reshaped([b, l, h])
        let k = kProj.forward(flat).reshaped([b, l, h])
        let v = vProj.forward(flat).reshaped([b, l, h])

        var out = Tensor.zeros([b, l, h])
        let scale: Float = 1.0 / sqrtf(Float(h))

        // For each batch and each query position compute attention over keys
        for bi in 0..<b {
            for qi in 0..<l {
                if mask[bi][qi] == 0 {
                    // Zero out outputs for padded query positions (strict invariant)
                    continue
                }
                // Compute scores[q, k]
                var maxScore: Float = -Float.greatestFiniteMagnitude
                var scores = [Float](repeating: 0, count: l)
                for ki in 0..<l {
                    // Mask out padded keys
                    if mask[bi][ki] == 0 {
                        scores[ki] = -1e9
                        continue
                    }
                    var dot: Float = 0
                    let qBase = (bi * l + qi) * h
                    let kBase = (bi * l + ki) * h
                    for d in 0..<h { dot += q.data[qBase + d] * k.data[kBase + d] }
                    let s = dot * scale
                    scores[ki] = s
                    if s > maxScore { maxScore = s }
                }
                // Stable softmax over scores
                var sumExp: Float = 0
                for ki in 0..<l {
                    let v = scores[ki] - maxScore
                    scores[ki] = expf(v)
                    // If key was masked, it remains exp(-1e9) ~ 0
                    sumExp += scores[ki]
                }
                let invSum = sumExp > 0 ? (1.0 / sumExp) : 0
                // Weighted sum over V
                let outBase = (bi * l + qi) * h
                for d in 0..<h {
                    var acc: Float = 0
                    for ki in 0..<l {
                        if scores[ki] == 0 { continue }
                        let vBase = (bi * l + ki) * h
                        acc += scores[ki] * v.data[vBase + d]
                    }
                    out.data[outBase + d] = acc * invSum
                }
            }
        }

        // Output projection O over [B*L, H]
        let outFlat = out.reshaped([b * l, h])
        let proj = oProj.forward(outFlat).reshaped([b, l, h])
        return proj
    }
}

// Multi-head self-attention with concat + output projection
public struct MultiHeadSelfAttention {
    public let hidden: Int
    public let numHeads: Int
    public let headDim: Int
    public var qProj: Linear
    public var kProj: Linear
    public var vProj: Linear
    public var oProj: Linear

    public init(hidden: Int, numHeads: Int, seed: UInt64) {
        precondition(numHeads >= 1, "numHeads must be >= 1")
        precondition(hidden % numHeads == 0, "hidden must be divisible by numHeads")
        self.hidden = hidden
        self.numHeads = numHeads
        self.headDim = hidden / numHeads
        self.qProj = Linear(inFeatures: hidden, outFeatures: hidden, seed: Seed.derive(seed, label: "q"))
        self.kProj = Linear(inFeatures: hidden, outFeatures: hidden, seed: Seed.derive(seed, label: "k"))
        self.vProj = Linear(inFeatures: hidden, outFeatures: hidden, seed: Seed.derive(seed, label: "v"))
        self.oProj = Linear(inFeatures: hidden, outFeatures: hidden, seed: Seed.derive(seed, label: "o"))
    }

    // x: [B, L, H], mask: [B][L] -> y: [B, L, H]
    public func forward(_ x: Tensor, mask: [[Int]]) -> Tensor {
        precondition(x.shape.count == 3 && x.shape[2] == hidden, "MultiHeadSelfAttention expects [B,L,H]")
        let b = x.shape[0]
        let l = x.shape[1]
        let h = x.shape[2]
        precondition(mask.count == b && mask.allSatisfy { $0.count == l }, "mask shape mismatch")

        Logger.shared.debug("MultiHeadSelfAttention.forward b=\(b) l=\(l) h=\(h) heads=\(numHeads)", category: Logger.Category.textBridge)

        // Projections
        let flat = x.reshaped([b * l, h])
        let q = qProj.forward(flat).reshaped([b, l, h])
        let k = kProj.forward(flat).reshaped([b, l, h])
        let v = vProj.forward(flat).reshaped([b, l, h])

        var combined = Tensor.zeros([b, l, h])

        for bi in 0..<b {
            for qi in 0..<l {
                if mask[bi][qi] == 0 {
                    // keep combined zeros for padded queries
                    continue
                }
                for head in 0..<numHeads {
                    let offset = head * headDim
                    let scale: Float = 1.0 / sqrtf(Float(headDim))
                    var maxScore: Float = -Float.greatestFiniteMagnitude
                    var scores = [Float](repeating: 0, count: l)
                    // scores for this head
                    for ki in 0..<l {
                        if mask[bi][ki] == 0 {
                            scores[ki] = -1e9
                            continue
                        }
                        var dot: Float = 0
                        let qBase = (bi * l + qi) * h + offset
                        let kBase = (bi * l + ki) * h + offset
                        for d in 0..<headDim {
                            dot += q.data[qBase + d] * k.data[kBase + d]
                        }
                        let s = dot * scale
                        scores[ki] = s
                        if s > maxScore { maxScore = s }
                    }
                    // softmax
                    var sumExp: Float = 0
                    for ki in 0..<l {
                        let v = scores[ki] - maxScore
                        scores[ki] = expf(v)
                        sumExp += scores[ki]
                    }
                    let invSum = sumExp > 0 ? (1.0 / sumExp) : 0
                    // weighted sum over V for this head
                    let outBase = (bi * l + qi) * h + offset
                    for d in 0..<headDim {
                        var acc: Float = 0
                        for ki in 0..<l {
                            if scores[ki] == 0 { continue }
                            let vBase = (bi * l + ki) * h + offset
                            acc += scores[ki] * v.data[vBase + d]
                        }
                        combined.data[outBase + d] = acc * invSum
                    }
                }
            }
        }

        // Output projection on concatenated heads [B*L, H]
        let outFlat = combined.reshaped([b * l, h])
        let proj = oProj.forward(outFlat).reshaped([b, l, h])
        return proj
    }
}
