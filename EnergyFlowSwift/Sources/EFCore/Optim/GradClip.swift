import Foundation

public enum GradClip {
    // Scales all tensors in-place so that the global L2 norm <= maxNorm.
    // Returns the scale applied (1.0 if no clipping or maxNorm<=0).
    public static func clipGlobalL2Norm(tensors: inout [Tensor], maxNorm: Float, eps: Float = 1e-6) -> Float {
        if maxNorm <= 0 { return 1.0 }
        var sumsq: Float = 0
        for t in tensors {
            for v in t.data { sumsq += v * v }
        }
        let norm = sqrt(sumsq)
        let denom = max(norm, eps)
        let scale = min(1.0, maxNorm / denom)
        if scale < 1.0 {
            for i in 0..<tensors.count {
                for j in 0..<tensors[i].count { tensors[i].data[j] *= scale }
            }
        }
        return scale
    }
}