import Foundation

public enum LRSchedulers {
    // Linear warmup for first warmupSteps, then cosine decay to minLR over decaySteps.
    // If decaySteps <= 0, LR stays at baseLR after warmup.
    public static func warmupCosine(baseLR: Float, minLR: Float, warmupSteps: Int, decaySteps: Int, step: Int) -> Float {
        if warmupSteps > 0 && step < warmupSteps {
            let p = Float(step + 1) / Float(max(warmupSteps, 1))
            return max(minLR, baseLR * p)
        }
        if decaySteps <= 0 { return baseLR }
        let t = max(0, step - warmupSteps)
        let tt = min(t, decaySteps)
        let cosArg = Double.pi * Double(tt) / Double(max(decaySteps, 1))
        let f = 0.5 * (1.0 + Float(Darwin.cos(cosArg)))
        return minLR + (baseLR - minLR) * f
    }
}