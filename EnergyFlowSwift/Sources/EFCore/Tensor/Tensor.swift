import Foundation

public enum DType: Sendable {
    case float32
}

public enum Device: Sendable {
    case cpu
}

// Tensor is a pure value type (arrays of Int/Float), safe to send across concurrency domains.
public struct Tensor: Sendable {
    public var shape: [Int]
    public var data: [Float]

    public init(shape: [Int], data: [Float]) {
        precondition(shape.reduce(1, *) == data.count, "Data count does not match shape product")
        self.shape = shape
        self.data = data
    }

    public init(zeros shape: [Int]) {
        self.shape = shape
        self.data = Array(repeating: 0, count: shape.reduce(1, *))
    }

    public static func zeros(_ shape: [Int]) -> Tensor {
        Tensor(zeros: shape)
    }

    public static func randomUniform(_ shape: [Int], min: Float = -0.02, max: Float = 0.02, seed: UInt64 = 42) -> Tensor {
        var rng = SeededGenerator(seed: seed)
        let count = shape.reduce(1, *)
        var arr = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let r = Float.random(in: 0..<1, using: &rng)
            arr[i] = min + (max - min) * r
        }
        return Tensor(shape: shape, data: arr)
    }

    public var count: Int { data.count }

    public func withStrides() -> ([Int], [Int]) {
        let s = shape
        var strides = [Int](repeating: 0, count: s.count)
        var acc = 1
        for i in stride(from: s.count - 1, through: 0, by: -1) {
            strides[i] = acc
            acc *= s[i]
        }
        return (s, strides)
    }

    public func flattenIndex(_ indices: [Int]) -> Int {
        precondition(indices.count == shape.count, "Rank mismatch")
        let (_, strides) = withStrides()
        var idx = 0
        for i in 0..<indices.count {
            idx += indices[i] * strides[i]
        }
        return idx
    }

    public subscript(_ indices: Int...) -> Float {
        get { data[flattenIndex(indices)] }
        set { data[flattenIndex(indices)] = newValue }
    }

    public func applied(_ f: (Float) -> Float) -> Tensor {
        var out = data
        for i in 0..<out.count { out[i] = f(out[i]) }
        return Tensor(shape: shape, data: out)
    }
}

// Simple deterministic RNG for reproducibility
public struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    public init(seed: UInt64) { self.state = seed &* 0x9E3779B97F4A7C15 }
    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
