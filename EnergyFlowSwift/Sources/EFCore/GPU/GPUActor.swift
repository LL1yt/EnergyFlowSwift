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
    var metricsPipelines: MetricsPipelines?
    var conv1DCaches: [UUID: Conv1DCacheEntry] = [:]
    var linearCaches: [UUID: LinearCacheEntry] = [:]
    var buffers: [String: MTLBuffer] = [:]
    private final class CommandBufferToken: @unchecked Sendable {
        let label: String
        let buffer: MTLCommandBuffer

        init(label: String, buffer: MTLCommandBuffer) {
            self.label = label
            self.buffer = buffer
        }
    }

    private struct PendingHostReadback {
        let label: String
        let execute: @Sendable () -> Void
    }

    private var pendingCommandBuffers: [CommandBufferToken] = []
    private var pendingHostReadbacks: [PendingHostReadback] = []
    private var activeBatchDepth: Int = 0
    private var batchEpoch: UInt64 = 0

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
        for token in pending {
            _ = await token.buffer.completed()
        }
        drainHostReadbacks()
        batchEpoch &+= 1
    }

    public func isBatching() -> Bool {
        return activeBatchDepth > 0
    }

    // MARK: - Command buffer utilities

    func awaitCommandBuffer<T: Sendable>(label: String,
                               commandBuffer: MTLCommandBuffer,
                               produce: @escaping @Sendable () throws -> T) async throws -> T {
        let readback = scheduleCommandBuffer(label: label,
                                             commandBuffer: commandBuffer,
                                             deferUntilSync: false,
                                             produce: produce)
        return try await readback.value()
    }

    func scheduleCommandBuffer<T: Sendable>(label: String,
                                  commandBuffer: MTLCommandBuffer,
                                  deferUntilSync: Bool,
                                  produce: @escaping @Sendable () throws -> T) -> GPUReadback<T> {
        let state = GPUReadbackState<T>()
        let token = CommandBufferToken(label: label, buffer: commandBuffer)
        let epochSnapshot = batchEpoch
        let requiresSync = deferUntilSync && activeBatchDepth > 0
        
        // Capture the error before the handler to make it Sendable-safe
        commandBuffer.addCompletedHandler { [weak self] cb in
            let error = cb.error
            guard let self else { return }
            Task { [weak self] in
                guard let self else { return }
                await self.enqueueHostReadback(label: label) { [state, produce] in
                    if let error = error {
                        state.resolve(.failure(GPUActorError.commandBufferFailed(label: label, underlying: error)))
                        return
                    }
                    do {
                        let value = try produce()
                        state.resolve(.success(value))
                    } catch {
                        state.resolve(.failure(error))
                    }
                }
            }
        }
        if activeBatchDepth > 0 {
            pendingCommandBuffers.append(token)
        }
        commandBuffer.commit()
        return GPUReadback(state: state,
                           actor: self,
                           epoch: epochSnapshot,
                           label: label,
                           requiresSync: requiresSync)
    }

    private func enqueueHostReadback(label: String, execute: @escaping @Sendable () -> Void) {
        pendingHostReadbacks.append(PendingHostReadback(label: label, execute: execute))
        if activeBatchDepth == 0 {
            drainHostReadbacks()
        }
    }

    private func drainHostReadbacks() {
        guard !pendingHostReadbacks.isEmpty else { return }
        let tasks = pendingHostReadbacks
        pendingHostReadbacks.removeAll(keepingCapacity: true)
        for task in tasks {
            task.execute()
        }
    }

    fileprivate func ensureBatchSynced(for epoch: UInt64, label: String) async {
        if activeBatchDepth > 0 {
            fatalError("GPUActor: readback \(label) awaited while batch still active. Call syncBatch() before accessing results.")
        }
        if batchEpoch == epoch {
            fatalError("GPUActor: readback \(label) awaited before syncBatch drained queued work.")
        }
    }

}

private final class GPUReadbackState<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<T, Error>?
    private var continuations: [CheckedContinuation<T, Error>] = []

    func awaitValue() async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            if let result = result {
                lock.unlock()
                continuation.resume(with: result)
            } else {
                continuations.append(continuation)
                lock.unlock()
            }
        }
    }

    func resolve(_ result: Result<T, Error>) {
        lock.lock()
        if self.result != nil {
            lock.unlock()
            return
        }
        self.result = result
        let continuations = self.continuations
        self.continuations.removeAll()
        lock.unlock()
        for continuation in continuations {
            continuation.resume(with: result)
        }
    }
}

public struct GPUReadback<T: Sendable>: Sendable {
    private let state: GPUReadbackState<T>
    private let actor: GPUActor?
    private let epoch: UInt64
    private let label: String
    private let requiresSync: Bool

    fileprivate init(state: GPUReadbackState<T>,
                     actor: GPUActor,
                     epoch: UInt64,
                     label: String,
                     requiresSync: Bool) {
        self.state = state
        self.actor = actor
        self.epoch = epoch
        self.label = label
        self.requiresSync = requiresSync
    }

    public init(resolved value: T) {
        let state = GPUReadbackState<T>()
        state.resolve(.success(value))
        self.state = state
        self.actor = nil
        self.epoch = 0
        self.label = ""
        self.requiresSync = false
    }

    public func value() async throws -> T {
        if requiresSync, let actor = actor {
            await actor.ensureBatchSynced(for: epoch, label: label)
        }
        return try await state.awaitValue()
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
