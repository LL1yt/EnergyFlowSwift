import Foundation

// MARK: - Losses and Metrics (CPU reference)
// Shapes:
// - Vectors: [B, D]
// - Returns per-example arrays (count B) and mean values

public enum Losses {
    public struct CosineResult {
        public let perExample: [Float]
        public let mean: Float
    }
    public struct MSEResult {
        public let perExample: [Float]
        public let mean: Float
    }

    // Row-wise cosine similarity between two [B, D] tensors
    public static func cosineSimilarityRowwise(_ a: Tensor, _ b: Tensor) -> CosineResult {
        precondition(a.shape.count == 2 && b.shape.count == 2 && a.shape == b.shape, "cosine requires same 2D shape [B,D]")
        let B = a.shape[0]
        let D = a.shape[1]
        var out = [Float](repeating: 0, count: B)
        var mean: Float = 0
        for bi in 0..<B {
            var dot: Float = 0
            var na: Float = 0
            var nb: Float = 0
            let base = bi * D
            for d in 0..<D {
                let va = a.data[base + d]
                let vb = b.data[base + d]
                dot += va * vb
                na += va * va
                nb += vb * vb
            }
            let denom = sqrt(max(na, 1e-12)) * sqrt(max(nb, 1e-12))
            let c = denom > 0 ? (dot / denom) : 0
            out[bi] = c
            mean += c
        }
        mean /= Float(max(B, 1))
        return CosineResult(perExample: out, mean: mean)
    }

    // Row-wise MSE between [B, D]
    public static func mseRowwise(_ a: Tensor, _ b: Tensor) -> MSEResult {
        precondition(a.shape.count == 2 && b.shape.count == 2 && a.shape == b.shape, "mse requires same 2D shape [B,D]")
        let B = a.shape[0]
        let D = a.shape[1]
        var out = [Float](repeating: 0, count: B)
        var mean: Float = 0
        for bi in 0..<B {
            let base = bi * D
            var acc: Float = 0
            for d in 0..<D {
                let diff = a.data[base + d] - b.data[base + d]
                acc += diff * diff
            }
            let v = acc / Float(D)
            out[bi] = v
            mean += v
        }
        mean /= Float(max(B, 1))
        return MSEResult(perExample: out, mean: mean)
    }
}