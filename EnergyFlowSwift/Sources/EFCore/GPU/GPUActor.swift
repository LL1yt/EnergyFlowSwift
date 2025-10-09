import Foundation
import Dispatch
import Metal

public enum GPUActorError: Error {
    case deviceUnavailable
    case commandQueueUnavailable
    case pipelineFunctionMissing(String)
    case commandBufferUnavailable(String)
}

public actor GPUActor {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private var elementwisePipelines: ElementwisePipelines?
    private var buffers: [String: MTLBuffer] = [:]

    public init() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("GPUActor: No Metal device available.")
        }
        guard let queue = device.makeCommandQueue() else {
            fatalError("GPUActor: Failed to create command queue.")
        }
        self.device = device
        self.commandQueue = queue
    }

    private func roundUp(_ value: Int, toMultipleOf alignment: Int) -> Int {
        let q = (value + alignment - 1) / alignment
        return max(1, q * alignment)
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

    public func residualAdd(y: Tensor, x: Tensor) throws -> Tensor {
        precondition(y.shape == x.shape, "GPUActor.residualAdd shape mismatch")
        let count = y.count
        if count == 0 { return y }
        let pipelines = try ensureElementwisePipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.residualAdd: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Elementwise.residualAdd"
        let byteCount = count * MemoryLayout<Float>.size
        let yBuffer = buffer(length: byteCount, label: "GPUActor.Elementwise.residualAdd.y")
        let xBuffer = buffer(length: byteCount, label: "GPUActor.Elementwise.residualAdd.x")
        x.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(xBuffer.contents(), base, byteCount)
            }
        }
        y.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(yBuffer.contents(), base, byteCount)
            }
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.residualAdd: encoder creation failed")
        }
        encoder.label = "GPUActor.Elementwise.residualAdd.encoder"
        encoder.setComputePipelineState(pipelines.residualAdd)
        encoder.setBuffer(yBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        var n = Int32(count)
        encoder.setBytes(&n, length: MemoryLayout<Int32>.size, index: 2)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var output = [Float](repeating: 0, count: count)
        memcpy(&output, yBuffer.contents(), byteCount)
        return Tensor(shape: y.shape, data: output)
    }

    private func ensureElementwisePipelines() throws -> ElementwisePipelines {
        if let pipelines = elementwisePipelines {
            return pipelines
        }
        let newPipelines = try ElementwisePipelines(device: device)
        elementwisePipelines = newPipelines
        return newPipelines
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
            let r: Result<T, Error>
            do {
                let value = try await operation(shared)
                r = .success(value)
            } catch {
                r = .failure(error)
            }
            box.set(r)
            semaphore.signal()
        }
        semaphore.wait()
        guard let final = box.get() else {
            fatalError("GPU.blocking\(label.map { "(\($0))" } ?? ""): missing result")
        }
        return try final.get()
    }
}

// Simple locked box to pass a single Result across a Task boundary without races.
// Using @unchecked Sendable because NSLock is not statically Sendable but is safe by construction here.
private final class _LockedResultBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<T, Error>? = nil
    func set(_ v: Result<T, Error>) {
        lock.lock(); defer { lock.unlock() }
        value = v
    }
    func get() -> Result<T, Error>? {
        lock.lock(); defer { lock.unlock() }
        return value
    }
}

private struct ElementwisePipelines {
    let library: MTLLibrary
    let residualAdd: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: ElementwiseMetalLibrary.source, options: nil)
        guard let residualAddFunction = library.makeFunction(name: "residual_add_f32") else {
            throw GPUActorError.pipelineFunctionMissing("residual_add_f32")
        }
        let residualAddState = try device.makeComputePipelineState(function: residualAddFunction)
        self.library = library
        self.residualAdd = residualAddState
    }
}
