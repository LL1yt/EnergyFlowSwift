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
        if mStates.count != paramCount {
            mStates = Array(repeating: [], count: paramCount)
            vStates = Array(repeating: [], count: paramCount)
        }
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

    // MARK: - State serialization (optimizer m/v, step t)
    // Binary format: "EFOP1" magic, u32 t, u32 paramCount, then per-param: u32 count, count*f32 m, count*f32 v
    public func exportState(paramCounts: [Int]) -> Data {
        var data = Data()
        // magic
        data.append(contentsOf: [0x45, 0x46, 0x4F, 0x50, 0x31]) // "EFOP1"
        func putU32(_ v: UInt32) { var le = v.littleEndian; withUnsafeBytes(of: &le) { data.append(contentsOf: $0) } }
        func putF32(_ v: Float) { var le = v.bitPattern.littleEndian; withUnsafeBytes(of: &le) { data.append(contentsOf: $0) } }
        putU32(UInt32(max(t, 0)))
        putU32(UInt32(paramCounts.count))
        // ensure states sized
        ensureStates(paramCount: paramCounts.count)
        for i in 0..<paramCounts.count {
            let cnt = max(0, paramCounts[i])
            putU32(UInt32(cnt))
            if mStates[i].count != cnt { mStates[i] = [Float](repeating: 0, count: cnt) }
            if vStates[i].count != cnt { vStates[i] = [Float](repeating: 0, count: cnt) }
            for j in 0..<cnt { putF32(mStates[i][j]) }
            for j in 0..<cnt { putF32(vStates[i][j]) }
        }
        return data
    }

    @discardableResult
    public func importState(_ data: Data, expectedParamCounts: [Int]) -> Bool {
        var idx = 0
        func getU8() -> UInt8 { defer { idx += 1 }; return data[idx] }
        guard idx + 5 <= data.count else { return false }
        // magic EFOP1
        if getU8() != 0x45 || getU8() != 0x46 || getU8() != 0x4F || getU8() != 0x50 || getU8() != 0x31 { return false }
        func getU32() -> UInt32 { defer { idx += 4 }; return data[idx..<(idx+4)].withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian }
        func getF32() -> Float { defer { idx += 4 }; let u = data[idx..<(idx+4)].withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian; return Float(bitPattern: u) }
        let tVal = Int(getU32())
        let pCount = Int(getU32())
        guard pCount == expectedParamCounts.count else { return false }
        ensureStates(paramCount: pCount)
        for i in 0..<pCount {
            let cnt = Int(getU32())
            guard cnt == expectedParamCounts[i] else { return false }
            if idx + 4*cnt*2 > data.count { return false }
            var m = [Float](repeating: 0, count: cnt)
            var v = [Float](repeating: 0, count: cnt)
            for j in 0..<cnt { m[j] = getF32() }
            for j in 0..<cnt { v[j] = getF32() }
            mStates[i] = m
            vStates[i] = v
        }
        self.t = tVal
        return true
    }

    @discardableResult
    public func saveState(path: String, paramCounts: [Int]) -> Bool {
        let url = URL(fileURLWithPath: path)
        do {
            let dir = url.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let data = exportState(paramCounts: paramCounts)
            try data.write(to: url)
            return true
        } catch {
            return false
        }
    }

    @discardableResult
    public func loadState(path: String, expectedParamCounts: [Int]) -> Bool {
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url) else { return false }
        return importState(data, expectedParamCounts: expectedParamCounts)
    }
}
