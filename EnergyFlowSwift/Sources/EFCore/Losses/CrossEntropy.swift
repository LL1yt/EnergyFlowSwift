import Foundation

// Cross entropy utils for logits [B,L,V], targets [B,L] (token IDs)
// Returns mean CE over all positions.
public enum CrossEntropyLoss {
    public static func meanLogits(logits: Tensor, targets: [[Int]]) -> Float {
        precondition(logits.shape.count == 3, "logits must be [B,L,V]")
        let B = logits.shape[0]
        let L = logits.shape[1]
        let V = logits.shape[2]
        precondition(targets.count == B && targets.allSatisfy { $0.count == L }, "targets shape mismatch")
        var total: Float = 0
        var count: Int = 0
        // Stable softmax via log-sum-exp
        for b in 0..<B {
            for t in 0..<L {
                let base = (b * L + t) * V
                var maxLogit: Float = -Float.greatestFiniteMagnitude
                for v in 0..<V { let lv = logits.data[base + v]; if lv > maxLogit { maxLogit = lv } }
                var sumExp: Float = 0
                for v in 0..<V { sumExp += expf(logits.data[base + v] - maxLogit) }
                let logZ = maxLogit + logf(sumExp)
                let y = targets[b][t]
                precondition(y >= 0 && y < V, "target id out of range")
                let ll = logits.data[base + y] - logZ
                total += -ll
                count += 1
            }
        }
        return total / Float(max(count, 1))
    }
}
