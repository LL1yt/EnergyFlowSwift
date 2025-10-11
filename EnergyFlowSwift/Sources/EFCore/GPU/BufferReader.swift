import Metal
import Foundation

/// Helper protocol for reading data from GPU buffers after command completion
protocol BufferReadable: Sendable {
    associatedtype Output: Sendable
    func read() -> Output
}

/// Generic buffer reader that captures buffer pointer and reads data after GPU work completes
struct BufferReader<T: Sendable>: BufferReadable, Sendable {
    let bufferPtr: UInt
    let reader: @Sendable () -> T
    
    init(buffer: MTLBuffer, reader: @escaping @Sendable () -> T) {
        self.bufferPtr = UInt(bitPattern: buffer.contents())
        self.reader = reader
    }
    
    func read() -> T {
        return reader()
    }
}

// MARK: - Common Buffer Readers

/// Reads Float array from buffer
struct FloatBufferReader: BufferReadable, Sendable {
    let bufferPtr: UInt
    let count: Int
    let byteCount: Int
    let shape: [Int]
    
    init(buffer: MTLBuffer, count: Int, shape: [Int]) {
        self.bufferPtr = UInt(bitPattern: buffer.contents())
        self.count = count
        self.byteCount = count * MemoryLayout<Float>.size
        self.shape = shape
    }
    
    func read() -> Tensor {
        let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
        var output = [Float](repeating: 0, count: count)
        memcpy(&output, ptr, byteCount)
        return Tensor(shape: shape, data: output)
    }
}

/// Reads Float16 array from buffer and converts to Float
struct Float16BufferReader: BufferReadable, Sendable {
    let bufferPtr: UInt
    let count: Int
    let byteCount: Int
    let shape: [Int]
    
    init(buffer: MTLBuffer, count: Int, shape: [Int]) {
        self.bufferPtr = UInt(bitPattern: buffer.contents())
        self.count = count
        self.byteCount = count * MemoryLayout<Float16>.size
        self.shape = shape
    }
    
    func read() -> Tensor {
        let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
        var halfData = [Float16](repeating: 0, count: count)
        memcpy(&halfData, ptr, byteCount)
        var output = [Float](repeating: 0, count: count)
        for i in 0..<count {
            output[i] = Float(halfData[i])
        }
        return Tensor(shape: shape, data: output)
    }
}

/// Reads Float16 from strided buffer layout (for matrix operations)
struct StridedFloat16BufferReader: BufferReadable, Sendable {
    let bufferPtr: UInt
    let rows: Int
    let cols: Int
    let elemSize: Int
    let rowBytes: Int
    let shape: [Int]
    
    init(buffer: MTLBuffer, rows: Int, cols: Int, rowBytes: Int, shape: [Int]) {
        self.bufferPtr = UInt(bitPattern: buffer.contents())
        self.rows = rows
        self.cols = cols
        self.elemSize = MemoryLayout<Float16>.size
        self.rowBytes = rowBytes
        self.shape = shape
    }
    
    func read() -> Tensor {
        let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
        var halfData = [Float16](repeating: 0, count: rows * cols)
        let rowSize = cols * elemSize
        halfData.withUnsafeMutableBytes { raw in
            if let base = raw.baseAddress {
                for row in 0..<rows {
                    let dst = base.advanced(by: row * rowSize)
                    let src = ptr.advanced(by: row * rowBytes)
                    memcpy(dst, src, rowSize)
                }
            }
        }
        var output = [Float](repeating: 0, count: rows * cols)
        for i in 0..<output.count {
            output[i] = Float(halfData[i])
        }
        return Tensor(shape: shape, data: output)
    }
}

/// Helper to read tuple results (for metrics)
struct TupleFloatReader: BufferReadable, Sendable {
    let bufferPtr: UInt
    let compute: @Sendable (UnsafeMutablePointer<Float>) -> (Float, Float)
    
    init(buffer: MTLBuffer, compute: @escaping @Sendable (UnsafeMutablePointer<Float>) -> (Float, Float)) {
        self.bufferPtr = UInt(bitPattern: buffer.contents())
        self.compute = compute
    }
    
    func read() -> (Float, Float) {
        let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
            .bindMemory(to: Float.self, capacity: 2)
        return compute(ptr)
    }
}

/// Helper for single Float result
struct SingleFloatReader: BufferReadable, Sendable {
    let bufferPtr: UInt
    let compute: @Sendable (UnsafeMutablePointer<Float>) -> Float
    
    init(buffer: MTLBuffer, compute: @escaping @Sendable (UnsafeMutablePointer<Float>) -> Float) {
        self.bufferPtr = UInt(bitPattern: buffer.contents())
        self.compute = compute
    }
    
    func read() -> Float {
        let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
            .bindMemory(to: Float.self, capacity: 2)
        return compute(ptr)
    }
}

/// Helper for multiple buffer results (like LayerNorm backward)
struct MultiBufferReader: BufferReadable, Sendable {
    let readers: [@Sendable () -> Tensor]
    
    func read() -> [Tensor] {
        return readers.map { $0() }
    }
}

// Extension to make buffer reading more convenient
extension GPUActor {
    /// Helper to schedule command buffer with a buffer reader
    func scheduleCommandBufferWithReader<R: BufferReadable>(
        label: String,
        commandBuffer: MTLCommandBuffer,
        deferUntilSync: Bool,
        reader: R
    ) -> GPUReadback<R.Output> where R.Output: Sendable {
        return scheduleCommandBuffer(
            label: label,
            commandBuffer: commandBuffer,
            deferUntilSync: deferUntilSync
        ) { [reader] in
            return reader.read()
        }
    }
    
    /// Helper for awaitCommandBuffer with a buffer reader
    func awaitCommandBufferWithReader<R: BufferReadable>(
        label: String,
        commandBuffer: MTLCommandBuffer,
        reader: R
    ) async throws -> R.Output where R.Output: Sendable {
        return try await awaitCommandBuffer(
            label: label,
            commandBuffer: commandBuffer
        ) { [reader] in
            return reader.read()
        }
    }
}
