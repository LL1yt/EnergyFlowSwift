import Foundation
import Dispatch
import Metal
import MetalPerformanceShaders

public enum GPUActorError: Error {
    case deviceUnavailable
    case commandQueueUnavailable
    case pipelineFunctionMissing(String)
    case commandBufferUnavailable(String)
}

public actor GPUActor {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    var elementwisePipelines: ElementwisePipelines?
    var geluPipelines: GELUPipelines?
    var layerNormPipelines: LayerNormPipelines?
    var embeddingPipelines: EmbeddingPipelines?
    var im2ColPipelines: Im2ColPipelines?
    var convPackPipelines: ConvPackPipelines?
    var conv1DCaches: [UUID: Conv1DCacheEntry] = [:]
    var linearCaches: [UUID: LinearCacheEntry] = [:]
    var buffers: [String: MTLBuffer] = [:]

    public init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("GPUActor: No Metal device available.")
        }
        guard let queue = device.makeCommandQueue() else {
            fatalError("GPUActor: Failed to create Metal command queue.")
        }
        self.device = device
        self.commandQueue = queue
    }

    func roundUp(_ value: Int, toMultipleOf alignment: Int) -> Int {
        let q = (value + alignment - 1) / alignment
        return max(1, q * alignment)
    }

    func alignedRowBytes(columns: Int, elemSize: Int) -> Int {
        let raw = columns * elemSize
        return ((raw + 15) / 16) * 16
    }

    public func buffer(length: Int, label: String) -> MTLBuffer {
        let required = roundUp(length, toMultipleOf: 4_096)
        if let existing = buffers[label], existing.length >= required {
            return existing
        }
        guard let buffer = device.makeBuffer(length: required, options: .storageModeShared) else {
            fatalError("GPUActor: Failed to allocate buffer '\(label)' length=\(required)")
        }
        buffer.label = label
        buffers[label] = buffer
        return buffer
    }
}

public enum GPU {
    public static let shared = GPUActor()

    public static func blocking<T>(
        label: String? = nil,
        _ operation: @escaping @Sendable (GPUActor) async throws -> T
    ) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        let box = _LockedResultBox<T>()
        Task {
            let result: Result<T, Error>
            do {
                let value = try await operation(shared)
                result = .success(value)
            } catch {
                result = .failure(error)
            }
            box.set(result)
            semaphore.signal()
        }
        semaphore.wait()
        guard let final = box.get() else {
            fatalError("GPU.blocking\(label.map { "(\($0))" } ?? ""): missing result")
        }
        return try final.get()
    }
}

// Simple box to shuttle a Result out of an async task.
private final class _LockedResultBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<T, Error>?

    func set(_ v: Result<T, Error>) {
        lock.lock()
        value = v
        lock.unlock()
    }

    func get() -> Result<T, Error>? {
        lock.lock()
        let v = value
        lock.unlock()
        return v
    }
}
