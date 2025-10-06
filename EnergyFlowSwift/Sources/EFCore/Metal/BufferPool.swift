import Foundation
import Metal

// Simple global buffer pool to reuse MTLBuffers by label.
// Notes:
// - Not thread-safe (nonisolated(unsafe)); safe for single-threaded training loops.
// - If a requested length exceeds the cached buffer, we allocate a new, larger buffer and replace it.
// - Buffers are created with storageModeShared for simplicity.
public enum BufferPool {
    nonisolated(unsafe) private static var buffers: [String: MTLBuffer] = [:]

    private static func roundUp(_ n: Int, toMultipleOf m: Int) -> Int {
        let r = (n + m - 1) / m
        return r * m
    }

    public static func buffer(device: MTLDevice, length: Int, label: String) -> MTLBuffer {
        let need = max(1, roundUp(length, toMultipleOf: 4_096)) // 4KB granularity
        if let b = buffers[label] {
            if b.length >= need { return b }
        }
        let buf = device.makeBuffer(length: need, options: .storageModeShared)!
        buf.label = label
        buffers[label] = buf
        return buf
    }

    // Convenience to clear (not used in hot path)
    public static func clear() {
        buffers.removeAll()
    }
}
