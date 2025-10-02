import Foundation

// MARK: - AdamW Optimizer (CPU-side, tensor lists)
// This is a skeleton to enable Swift-side training hookup later.
// It updates inout Tensors given external gradients.

public final class AdamW {
    public var lr: Float
    public var beta1: Float
    public var beta2: Float
    public var eps: Float
    public var weightDecay: Float

    private var t: Int = 0
    // m,v states per parameter index (flattened arrays of size param.count)
    private var mStates: [[Float]] = []
    private var vStates: [[Float]] = []

    public init(lr: Float = 3e-4, beta1: Float = 0.9, beta2: Float = 0.999, eps: Float = 1e-8, weightDecay: Float = 0.01) {
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weightDecay = weightDecay
    }

    private func ensureStates(paramCount: Int) {
        if mStates.count == paramCount { return }
        mStates = mStates.count == paramCount ? mStates : Array(repeating: [], count: paramCount)
        vStates = vStates.count == paramCount ? vStates : Array(repeating: [], count: paramCount)
    }

    // Step over parameter list. Each param has matching grad shape.
    // NOTE: Caller is responsible for providing grads. Autodiff hookup is separate.
    public func step(params: inout [Tensor], grads: [Tensor]) {
        precondition(params.count == grads.count, "AdamW.step: params and grads mismatch")
        ensureStates(paramCount: params.count)
        t += 1
        let bt1t = pow(beta1, Float(t))
        let bt2t = pow(beta2, Float(t))
        let corr1 = 1 - bt1t
        let corr2 = 1 - bt2t
        for i in 0..<params.count {
            var p = params[i]
            let g = grads[i]
            precondition(p.shape == g.shape, "AdamW.step: shape mismatch at index \(i)")
            if mStates[i].count != p.count { mStates[i] = [Float](repeating: 0, count: p.count) }
            if vStates[i].count != p.count { vStates[i] = [Float](repeating: 0, count: p.count) }
            for j in 0..<p.count {
                var grad = g.data[j]
                // weight decay (AdamW decoupled)
                grad += weightDecay * p.data[j]
                // m and v
                mStates[i][j] = beta1 * mStates[i][j] + (1 - beta1) * grad
                vStates[i][j] = beta2 * vStates[i][j] + (1 - beta2) * (grad * grad)
                let mHat = mStates[i][j] / max(corr1, 1e-12)
                let vHat = vStates[i][j] / max(corr2, 1e-12)
                p.data[j] -= lr * (mHat / (sqrt(vHat) + eps))
            }
            params[i] = p
        }
    }
}