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

    public func addBroadcast2DInto3D(y: Tensor, addBD: Tensor, sequenceLength L: Int) throws -> Tensor {
        precondition(y.shape.count == 3, "addBroadcast2DInto3D expects y [B,L,D]")
        let B = y.shape[0], seq = y.shape[1], D = y.shape[2]
        precondition(seq == L, "sequenceLength mismatch: provided \(L) vs tensor \(seq)")
        precondition(addBD.shape == [B, D], "addBroadcast2DInto3D expects add [B,D]")
        let count = B * L * D
        if count == 0 { return y }
        let pipelines = try ensureElementwisePipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.addBroadcast: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Elementwise.addBroadcast2DInto3D"
        let elem = MemoryLayout<Float>.size
        let yBytes = count * elem
        let addBytes = B * D * elem
        let yBuffer = buffer(length: yBytes, label: "GPUActor.Elementwise.addBroadcast.y")
        let addBuffer = buffer(length: addBytes, label: "GPUActor.Elementwise.addBroadcast.add")
        y.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(yBuffer.contents(), base, yBytes)
            }
        }
        addBD.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(addBuffer.contents(), base, addBytes)
            }
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.addBroadcast: encoder creation failed")
        }
        encoder.label = "GPUActor.Elementwise.addBroadcast.encoder"
        encoder.setComputePipelineState(pipelines.addBroadcast2DInto3D)
        encoder.setBuffer(yBuffer, offset: 0, index: 0)
        encoder.setBuffer(addBuffer, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vD = Int32(D)
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 4)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var out = [Float](repeating: 0, count: count)
        memcpy(&out, yBuffer.contents(), yBytes)
        return Tensor(shape: y.shape, data: out)
    }

    public func maskZero(y: Tensor, mask: [[Int]]) throws -> Tensor {
        precondition(y.shape.count == 3, "maskZero expects y [B,L,D]")
        let B = y.shape[0], L = y.shape[1], D = y.shape[2]
        let maskFlat = flattenMask(mask, expectedBatch: B, expectedLength: L)
        let count = B * L * D
        if count == 0 { return y }
        let pipelines = try ensureElementwisePipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskZero: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Elementwise.maskZero"
        let elem = MemoryLayout<Float>.size
        let yBytes = count * elem
        let maskBytes = maskFlat.count * MemoryLayout<Int32>.size
        let yBuffer = buffer(length: yBytes, label: "GPUActor.Elementwise.maskZero.y")
        let maskBuffer = buffer(length: maskBytes, label: "GPUActor.Elementwise.maskZero.mask")
        y.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(yBuffer.contents(), base, yBytes)
            }
        }
        maskFlat.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(maskBuffer.contents(), base, maskBytes)
            }
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskZero: encoder creation failed")
        }
        encoder.label = "GPUActor.Elementwise.maskZero.encoder"
        encoder.setComputePipelineState(pipelines.maskZero)
        encoder.setBuffer(yBuffer, offset: 0, index: 0)
        encoder.setBuffer(maskBuffer, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vD = Int32(D)
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 4)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var out = [Float](repeating: 0, count: count)
        memcpy(&out, yBuffer.contents(), yBytes)
        return Tensor(shape: y.shape, data: out)
    }

    public func maskedMean(x: Tensor, mask: [[Int]]) throws -> Tensor {
        precondition(x.shape.count == 3, "maskedMean expects x [B,L,H]")
        let B = x.shape[0], L = x.shape[1], H = x.shape[2]
        let maskFlat = flattenMask(mask, expectedBatch: B, expectedLength: L)
        let pipelines = try ensureElementwisePipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskedMean: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Elementwise.maskedMean"
        let elem = MemoryLayout<Float>.size
        let xBytes = B * L * H * elem
        let yBytes = B * H * elem
        let maskBytes = maskFlat.count * MemoryLayout<Int32>.size
        let xBuffer = buffer(length: xBytes, label: "GPUActor.Elementwise.maskedMean.x")
        let yBuffer = buffer(length: yBytes, label: "GPUActor.Elementwise.maskedMean.y")
        let maskBuffer = buffer(length: maskBytes, label: "GPUActor.Elementwise.maskedMean.mask")
        x.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(xBuffer.contents(), base, xBytes)
            }
        }
        memset(yBuffer.contents(), 0, yBytes)
        maskFlat.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(maskBuffer.contents(), base, maskBytes)
            }
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskedMean: encoder creation failed")
        }
        encoder.label = "GPUActor.Elementwise.maskedMean.encoder"
        encoder.setComputePipelineState(pipelines.maskedMean)
        encoder.setBuffer(xBuffer, offset: 0, index: 0)
        encoder.setBuffer(maskBuffer, offset: 0, index: 1)
        encoder.setBuffer(yBuffer, offset: 0, index: 2)
        var vB = Int32(B), vL = Int32(L), vH = Int32(H), vEps: Float = 1e-9
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes(&vH, length: MemoryLayout<Int32>.size, index: 5)
        encoder.setBytes(&vEps, length: MemoryLayout<Float>.size, index: 6)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (B * H + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var out = [Float](repeating: 0, count: B * H)
        memcpy(&out, yBuffer.contents(), yBytes)
        return Tensor(shape: [B, H], data: out)
    }

    public func maskedMeanBackward(dPooled: Tensor, mask: [[Int]], seqLen: Int) throws -> Tensor {
        precondition(dPooled.shape.count == 2, "maskedMeanBackward expects [B,H]")
        let B = dPooled.shape[0], H = dPooled.shape[1]
        let maskFlat = flattenMask(mask, expectedBatch: B, expectedLength: seqLen)
        let pipelines = try ensureElementwisePipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskedMeanBackward: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Elementwise.maskedMeanBackward"
        let elem = MemoryLayout<Float>.size
        let dyBytes = B * H * elem
        let dxBytes = B * seqLen * H * elem
        let maskBytes = maskFlat.count * MemoryLayout<Int32>.size
        let dyBuffer = buffer(length: dyBytes, label: "GPUActor.Elementwise.maskedMeanBackward.dy")
        let dxBuffer = buffer(length: dxBytes, label: "GPUActor.Elementwise.maskedMeanBackward.dx")
        let maskBuffer = buffer(length: maskBytes, label: "GPUActor.Elementwise.maskedMeanBackward.mask")
        dPooled.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(dyBuffer.contents(), base, dyBytes)
            }
        }
        memset(dxBuffer.contents(), 0, dxBytes)
        maskFlat.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(maskBuffer.contents(), base, maskBytes)
            }
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskedMeanBackward: encoder creation failed")
        }
        encoder.label = "GPUActor.Elementwise.maskedMeanBackward.encoder"
        encoder.setComputePipelineState(pipelines.maskedMeanBackward)
        encoder.setBuffer(dyBuffer, offset: 0, index: 0)
        encoder.setBuffer(maskBuffer, offset: 0, index: 1)
        encoder.setBuffer(dxBuffer, offset: 0, index: 2)
        var vB = Int32(B), vL = Int32(seqLen), vH = Int32(H), vEps: Float = 1e-9
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes(&vH, length: MemoryLayout<Int32>.size, index: 5)
        encoder.setBytes(&vEps, length: MemoryLayout<Float>.size, index: 6)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (B * seqLen * H + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var out = [Float](repeating: 0, count: B * seqLen * H)
        memcpy(&out, dxBuffer.contents(), dxBytes)
        return Tensor(shape: [B, seqLen, H], data: out)
    }

    private func flattenMask(_ mask: [[Int]], expectedBatch: Int, expectedLength: Int) -> [Int32] {
        precondition(mask.count == expectedBatch, "mask batch mismatch: expected \(expectedBatch) got \(mask.count)")
        var flat = [Int32](repeating: 0, count: expectedBatch * expectedLength)
        for b in 0..<expectedBatch {
            let row = mask[b]
            precondition(row.count == expectedLength, "mask length mismatch at batch \(b): expected \(expectedLength) got \(row.count)")
            for t in 0..<expectedLength {
                flat[b * expectedLength + t] = Int32(row[t])
            }
        }
        return flat
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
    let maskZero: MTLComputePipelineState
    let maskedMean: MTLComputePipelineState
    let maskedMeanBackward: MTLComputePipelineState
    let addBroadcast2DInto3D: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: ElementwiseMetalLibrary.source, options: nil)
        guard let residualAddFunction = library.makeFunction(name: "residual_add_f32") else {
            throw GPUActorError.pipelineFunctionMissing("residual_add_f32")
        }
        guard let maskZeroFunction = library.makeFunction(name: "mask_zero_f32") else {
            throw GPUActorError.pipelineFunctionMissing("mask_zero_f32")
        }
        guard let maskedMeanFunction = library.makeFunction(name: "masked_mean_f32") else {
            throw GPUActorError.pipelineFunctionMissing("masked_mean_f32")
        }
        guard let maskedMeanBackwardFunction = library.makeFunction(name: "masked_mean_bwd_f32") else {
            throw GPUActorError.pipelineFunctionMissing("masked_mean_bwd_f32")
        }
        guard let addBroadcastFunction = library.makeFunction(name: "add_broadcast_2d_into_3d_f32") else {
            throw GPUActorError.pipelineFunctionMissing("add_broadcast_2d_into_3d_f32")
        }
        let residualAddState = try device.makeComputePipelineState(function: residualAddFunction)
        let maskZeroState = try device.makeComputePipelineState(function: maskZeroFunction)
        let maskedMeanState = try device.makeComputePipelineState(function: maskedMeanFunction)
        let maskedMeanBackwardState = try device.makeComputePipelineState(function: maskedMeanBackwardFunction)
        let addBroadcastState = try device.makeComputePipelineState(function: addBroadcastFunction)
        self.library = library
        self.residualAdd = residualAddState
        self.maskZero = maskZeroState
        self.maskedMean = maskedMeanState
        self.maskedMeanBackward = maskedMeanBackwardState
        self.addBroadcast2DInto3D = addBroadcastState
    }
}
