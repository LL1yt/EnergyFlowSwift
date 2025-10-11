import Foundation
import Dispatch
import Metal
import MetalPerformanceShaders

public enum GPUActorError: Error {
    case deviceUnavailable
    case commandQueueUnavailable
    case pipelineFunctionMissing(String)
    case commandBufferUnavailable(String)
    case commandBufferFailed(label: String, underlying: Error?)
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
    private struct PendingHostReadback {
        let label: String
        let resume: () -> Void
    }

    private var pendingCommandBuffers: [MTLCommandBuffer] = []
    private var pendingHostReadbacks: [PendingHostReadback] = []
    private var activeBatchDepth: Int = 0

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

    // MARK: - Batch helpers (Phase 6b scaffolding)

    public func beginBatch() {
        activeBatchDepth &+= 1
    }

    public func syncBatch(label: String? = nil) async {
        if activeBatchDepth > 0 {
            activeBatchDepth &-= 1
        }
        guard activeBatchDepth == 0 else { return }
        let pending = pendingCommandBuffers
        pendingCommandBuffers.removeAll(keepingCapacity: true)
        for buffer in pending {
            buffer.waitUntilCompleted()
        }
        drainHostReadbacks()
    }

    public func isBatching() -> Bool {
        return activeBatchDepth > 0
    }

    // MARK: - Metrics helpers

    public func kdMetricsMean(student: Tensor, teacher: Tensor) -> (mse: Float, cos: Float) {
        let mse = Losses.mseRowwise(student, teacher).mean
        let cos = Losses.cosineSimilarityRowwise(student, teacher).mean
        return (mse, cos)
    }

    public func crossEntropyMean(logits: Tensor, targets: [[Int]]) -> Float {
        return CrossEntropyLoss.meanLogits(logits: logits, targets: targets)
    }

    // MARK: - Command buffer utilities

    func awaitCommandBuffer<T>(label: String,
                               commandBuffer: MTLCommandBuffer,
                               produce: @escaping () throws -> T) async throws -> T {
        return try await withCheckedThrowingContinuation { continuation in
            commandBuffer.addCompletedHandler { [weak self] cb in
                guard let self else { return }
                Task {
                    await self.handleCommandBufferCompletion(label: label,
                                                             commandBuffer: cb,
                                                             produce: produce,
                                                             continuation: continuation)
                }
            }
            if activeBatchDepth > 0 {
                pendingCommandBuffers.append(commandBuffer)
            }
            commandBuffer.commit()
        }
    }

    private func enqueueHostReadback(label: String, resume: @escaping () -> Void) {
        pendingHostReadbacks.append(PendingHostReadback(label: label, resume: resume))
        if activeBatchDepth == 0 {
            drainHostReadbacks()
        }
    }

    private func drainHostReadbacks() {
        guard !pendingHostReadbacks.isEmpty else { return }
        let tasks = pendingHostReadbacks
        pendingHostReadbacks.removeAll(keepingCapacity: true)
        for task in tasks {
            task.resume()
        }
    }

    private func handleCommandBufferCompletion<T>(label: String,
                                                  commandBuffer: MTLCommandBuffer,
                                                  produce: @escaping () throws -> T,
                                                  continuation: CheckedContinuation<T, Error>) async {
        let resume: () -> Void = {
            if let error = commandBuffer.error {
                continuation.resume(throwing: GPUActorError.commandBufferFailed(label: label, underlying: error))
                return
            }
            do {
                let value = try produce()
                continuation.resume(returning: value)
            } catch {
                continuation.resume(throwing: error)
            }
        }
        enqueueHostReadback(label: label, resume: resume)
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
