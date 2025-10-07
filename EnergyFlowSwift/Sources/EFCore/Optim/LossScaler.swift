import Foundation

// Simple dynamic loss scaler for mixed precision training.
// Usage:
// - Multiply upstream gradients (e.g., dY) by currentScale before GPU backward
// - Unscale grads for optimizer by multiplying with 1/currentScale (feed via inputs.scale factor)
// - On overflow (NaN/Inf detected in grads), call onOverflow() and skip the step
// - On successful step, call onGoodStep() to allow gradual scale growth
public struct LossScaler {
    public private(set) var currentScale: Float
    public let growthFactor: Float
    public let backoffFactor: Float
    public let growthInterval: Int
    private var goodSteps: Int
    public let minScale: Float
    public let maxScale: Float

    public init(initialScale: Float = 1024.0,
                growthFactor: Float = 2.0,
                backoffFactor: Float = 0.5,
                growthInterval: Int = 2000,
                minScale: Float = 1.0,
                maxScale: Float = 65536.0) {
        self.currentScale = initialScale
        self.growthFactor = growthFactor
        self.backoffFactor = backoffFactor
        self.growthInterval = max(1, growthInterval)
        self.goodSteps = 0
        self.minScale = minScale
        self.maxScale = maxScale
    }

    public mutating func onOverflow() {
        currentScale = max(minScale, currentScale * backoffFactor)
        goodSteps = 0
    }

    public mutating func onGoodStep() {
        goodSteps += 1
        if goodSteps >= growthInterval {
            currentScale = min(maxScale, currentScale * growthFactor)
            goodSteps = 0
        }
    }

    @inline(__always) public func invScale() -> Float { return 1.0 / max(currentScale, 1.0) }
}