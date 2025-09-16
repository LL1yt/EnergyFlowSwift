import Foundation

// MARK: - Tensor shape helpers (beginner-friendly)
// - prettyShape: returns a human-readable shape string, e.g. "[2, 256]"
// - reshaped: returns a new Tensor with the same data but a different shape (same element count)
// - assertShape: runtime check that compares a Tensor's shape to an expected pattern
//   Use -1 in expected to accept any size at that dimension (wildcard).

public extension Tensor {
    // Human-readable shape string for logging
    var prettyShape: String {
        "[" + shape.map { String($0) }.joined(separator: ", ") + "]"
    }

    // Return a new tensor with the same data but a different shape
    func reshaped(_ newShape: [Int], file: StaticString = #file, line: UInt = #line) -> Tensor {
        let newCount = newShape.reduce(1, *)
        precondition(newCount == self.count, "reshaped: element count mismatch. old: \(self.count), new: \(newCount). oldShape=\(self.prettyShape), newShape=\(newShape)", file: file, line: line)
        return Tensor(shape: newShape, data: self.data)
    }
}

// Assert that a tensor's shape matches the expected pattern.
// Use -1 as a wildcard for any dimension.
@discardableResult
public func assertShape(_ t: Tensor, _ expected: [Int], file: StaticString = #file, line: UInt = #line) -> Bool {
    guard t.shape.count == expected.count else {
        fatalError("assertShape: rank mismatch. got=\(t.prettyShape), expected=\(expected).", file: file, line: line)
    }
    for (i, (got, exp)) in zip(t.shape, expected).enumerated() {
        if exp == -1 { continue }
        if got != exp {
            fatalError("assertShape: dim[\(i)] mismatch. got=\(got), expected=\(exp). full=\(t.prettyShape) vs \(expected)", file: file, line: line)
        }
    }
    return true
}
