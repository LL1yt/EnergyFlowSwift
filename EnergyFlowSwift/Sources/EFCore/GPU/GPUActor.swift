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

    private var elementwisePipelines: ElementwisePipelines?
    private var geluPipelines: GELUPipelines?
    private var layerNormPipelines: LayerNormPipelines?
    private var embeddingPipelines: EmbeddingPipelines?
    private var linearCaches: [UUID: LinearCacheEntry] = [:]
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

    private func alignedRowBytes(columns: Int, elemSize: Int) -> Int {
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

    private func ensureGELUPipelines() throws -> GELUPipelines {
        if let pipelines = geluPipelines {
            return pipelines
        }
        let newPipelines = try GELUPipelines(device: device)
        geluPipelines = newPipelines
        return newPipelines
    }

    private func ensureLayerNormPipelines() throws -> LayerNormPipelines {
        if let pipelines = layerNormPipelines {
            return pipelines
        }
        let newPipelines = try LayerNormPipelines(device: device)
        layerNormPipelines = newPipelines
        return newPipelines
    }

    private func ensureEmbeddingPipelines() throws -> EmbeddingPipelines {
        if let pipelines = embeddingPipelines {
            return pipelines
        }
        let newPipelines = try EmbeddingPipelines(device: device)
        embeddingPipelines = newPipelines
        return newPipelines
    }

    private func ensureLinearCache(key: UUID,
                                   version: UInt64,
                                   inFeatures: Int,
                                   outFeatures: Int,
                                   weight: Tensor) throws -> LinearCacheEntry {
        if let entry = linearCaches[key],
           entry.version == version,
           entry.inFeatures == inFeatures,
           entry.outFeatures == outFeatures {
            return entry
        }
        let elemHalf = MemoryLayout<Float16>.size
        let rowBytes = alignedRowBytes(columns: inFeatures, elemSize: elemHalf)
        guard let weightBuffer = device.makeBuffer(length: outFeatures * rowBytes, options: .storageModeShared) else {
            throw GPUActorError.commandBufferUnavailable("Linear.ensureCache: weight buffer allocation failed")
        }
        weightBuffer.label = "GPUActor.Linear.weight.\(key)"
        memset(weightBuffer.contents(), 0, weightBuffer.length)
        var weightHalf = [Float16](repeating: 0, count: outFeatures * inFeatures)
        for i in 0..<(outFeatures * inFeatures) {
            weightHalf[i] = Float16(weight.data[i])
        }
        weightHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                for row in 0..<outFeatures {
                    let src = base.advanced(by: row * inFeatures * elemHalf)
                    let dst = weightBuffer.contents().advanced(by: row * rowBytes)
                    memcpy(dst, src, inFeatures * elemHalf)
                }
            }
        }
        let entry = LinearCacheEntry(
            version: version,
            weightBuffer: weightBuffer,
            rowBytes: rowBytes,
            inFeatures: inFeatures,
            outFeatures: outFeatures
        )
        linearCaches[key] = entry
        return entry
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

    public func geluForward(x: Tensor) throws -> Tensor {
        let count = x.count
        if count == 0 { return x }
        let pipelines = try ensureGELUPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("GELU.forward: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.GELU.forward"
        let elemHalf = MemoryLayout<Float16>.size
        let byteCount = count * elemHalf
        let xBuffer = buffer(length: byteCount, label: "GPUActor.GELU.forward.x")
        let yBuffer = buffer(length: byteCount, label: "GPUActor.GELU.forward.y")
        var xHalf = [Float16](repeating: 0, count: count)
        for i in 0..<count { xHalf[i] = Float16(x.data[i]) }
        xHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(xBuffer.contents(), base, byteCount)
            }
        }
        memset(yBuffer.contents(), 0, byteCount)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("GELU.forward: encoder creation failed")
        }
        encoder.label = "GPUActor.GELU.forward.encoder"
        encoder.setComputePipelineState(pipelines.forward)
        encoder.setBuffer(xBuffer, offset: 0, index: 0)
        encoder.setBuffer(yBuffer, offset: 0, index: 1)
        var n = Int32(count)
        encoder.setBytes(&n, length: MemoryLayout<Int32>.size, index: 2)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var yHalf = [Float16](repeating: 0, count: count)
        memcpy(&yHalf, yBuffer.contents(), byteCount)
        var output = [Float](repeating: 0, count: count)
        for i in 0..<count { output[i] = Float(yHalf[i]) }
        return Tensor(shape: x.shape, data: output)
    }

    public func geluBackward(x: Tensor, dY: Tensor) throws -> Tensor {
        precondition(x.shape == dY.shape, "GPUActor.geluBackward shape mismatch")
        let count = x.count
        if count == 0 { return dY }
        let pipelines = try ensureGELUPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("GELU.backward: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.GELU.backward"
        let elemHalf = MemoryLayout<Float16>.size
        let byteCount = count * elemHalf
        let xBuffer = buffer(length: byteCount, label: "GPUActor.GELU.backward.x")
        let dyBuffer = buffer(length: byteCount, label: "GPUActor.GELU.backward.dy")
        let dxBuffer = buffer(length: byteCount, label: "GPUActor.GELU.backward.dx")
        var xHalf = [Float16](repeating: 0, count: count)
        var dyHalf = [Float16](repeating: 0, count: count)
        for i in 0..<count {
            xHalf[i] = Float16(x.data[i])
            dyHalf[i] = Float16(dY.data[i])
        }
        xHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(xBuffer.contents(), base, byteCount)
            }
        }
        dyHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(dyBuffer.contents(), base, byteCount)
            }
        }
        memset(dxBuffer.contents(), 0, byteCount)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("GELU.backward: encoder creation failed")
        }
        encoder.label = "GPUActor.GELU.backward.encoder"
        encoder.setComputePipelineState(pipelines.backward)
        encoder.setBuffer(xBuffer, offset: 0, index: 0)
        encoder.setBuffer(dyBuffer, offset: 0, index: 1)
        encoder.setBuffer(dxBuffer, offset: 0, index: 2)
        var n = Int32(count)
        encoder.setBytes(&n, length: MemoryLayout<Int32>.size, index: 3)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var dxHalf = [Float16](repeating: 0, count: count)
        memcpy(&dxHalf, dxBuffer.contents(), byteCount)
        var output = [Float](repeating: 0, count: count)
        for i in 0..<count { output[i] = Float(dxHalf[i]) }
        return Tensor(shape: x.shape, data: output)
    }

    public func embeddingForward(ids: [[Int]], weight: Tensor) throws -> Tensor {
        let B = ids.count
        let L = ids.first?.count ?? 0
        precondition(ids.allSatisfy { $0.count == L }, "GPUActor.embeddingForward: ragged ids input")
        precondition(weight.shape.count == 2, "GPUActor.embeddingForward: weight must be [V, D]")
        let V = weight.shape[0]
        let D = weight.shape[1]
        if B == 0 || L == 0 { return Tensor.zeros([B, L, D]) }
        let pipelines = try ensureEmbeddingPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Embedding.forward: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Embedding.forward"
        let idsCount = B * L
        let outCount = B * L * D
        let idsBuffer = buffer(length: idsCount * MemoryLayout<Int32>.size, label: "GPUActor.Embedding.ids")
        var idsFlat = [Int32](repeating: 0, count: idsCount)
        for b in 0..<B {
            for t in 0..<L {
                idsFlat[b * L + t] = Int32(ids[b][t])
            }
        }
        idsFlat.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(idsBuffer.contents(), base, idsCount * MemoryLayout<Int32>.size)
            }
        }
        let weightCount = V * D
        let weightBytes = weightCount * MemoryLayout<Float>.size
        let weightBuffer = buffer(length: weightBytes, label: "GPUActor.Embedding.weight")
        weight.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(weightBuffer.contents(), base, weightBytes)
            }
        }
        let outBuffer = buffer(length: outCount * MemoryLayout<Float>.size, label: "GPUActor.Embedding.out")
        memset(outBuffer.contents(), 0, outCount * MemoryLayout<Float>.size)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Embedding.forward: encoder creation failed")
        }
        encoder.label = "GPUActor.Embedding.forward.encoder"
        encoder.setComputePipelineState(pipelines.gather)
        encoder.setBuffer(idsBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outBuffer, offset: 0, index: 2)
        var vB = Int32(B)
        var vL = Int32(L)
        var vV = Int32(V)
        var vD = Int32(D)
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes(&vV, length: MemoryLayout<Int32>.size, index: 5)
        encoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 6)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (outCount + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var output = [Float](repeating: 0, count: outCount)
        memcpy(&output, outBuffer.contents(), outCount * MemoryLayout<Float>.size)
        return Tensor(shape: [B, L, D], data: output)
    }

    public func linearForward(key: UUID,
                              version: UInt64,
                              inFeatures: Int,
                              outFeatures: Int,
                              weight: Tensor,
                              bias: Tensor?,
                              x: Tensor) throws -> Tensor {
        precondition(x.shape.count == 2 && x.shape[1] == inFeatures, "linearForward expects [B, inFeatures]")
        let batch = x.shape[0]
        if batch == 0 { return x }
        let cache = try ensureLinearCache(
            key: key,
            version: version,
            inFeatures: inFeatures,
            outFeatures: outFeatures,
            weight: weight
        )
        let hasBias = bias != nil
        let inputCols = hasBias ? (inFeatures + 1) : inFeatures
        let elemHalf = MemoryLayout<Float16>.size
        let xRowBytes = alignedRowBytes(columns: inputCols, elemSize: elemHalf)
        let yRowBytes = alignedRowBytes(columns: outFeatures, elemSize: elemHalf)
        let xBuffer = buffer(length: batch * xRowBytes, label: "GPUActor.Linear.forward.x.\(key)")
        let yBuffer = buffer(length: batch * yRowBytes, label: "GPUActor.Linear.forward.y.\(key)")
        memset(xBuffer.contents(), 0, batch * xRowBytes)
        memset(yBuffer.contents(), 0, batch * yRowBytes)
        var xHalf = [Float16](repeating: 0, count: batch * inputCols)
        for row in 0..<batch {
            let inBase = row * inFeatures
            let outBase = row * inputCols
            for c in 0..<inFeatures {
                xHalf[outBase + c] = Float16(x.data[inBase + c])
            }
            if hasBias {
                xHalf[outBase + inputCols - 1] = Float16(1.0)
            }
        }
        xHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                for row in 0..<batch {
                    let src = base.advanced(by: row * inputCols * elemHalf)
                    let dst = xBuffer.contents().advanced(by: row * xRowBytes)
                    memcpy(dst, src, inputCols * elemHalf)
                }
            }
        }
        let weightBuffer: MTLBuffer
        let weightRowBytes: Int
        if hasBias, let biasTensor = bias {
            let augmentedRowBytes = alignedRowBytes(columns: inputCols, elemSize: elemHalf)
            let wAugBuffer = buffer(length: outFeatures * augmentedRowBytes, label: "GPUActor.Linear.forward.wAug.\(key)")
            memset(wAugBuffer.contents(), 0, outFeatures * augmentedRowBytes)
            for row in 0..<outFeatures {
                let src = cache.weightBuffer.contents().advanced(by: row * cache.rowBytes)
                let dst = wAugBuffer.contents().advanced(by: row * augmentedRowBytes)
                memcpy(dst, src, inFeatures * elemHalf)
            }
            var biasHalf = [Float16](repeating: 0, count: outFeatures)
            for i in 0..<outFeatures { biasHalf[i] = Float16(biasTensor.data[i]) }
            biasHalf.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    for row in 0..<outFeatures {
                        let dst = wAugBuffer.contents().advanced(by: row * augmentedRowBytes + inFeatures * elemHalf)
                        memcpy(dst, base.advanced(by: row * elemHalf), elemHalf)
                    }
                }
            }
            weightBuffer = wAugBuffer
            weightRowBytes = augmentedRowBytes
        } else {
            weightBuffer = cache.weightBuffer
            weightRowBytes = cache.rowBytes
        }
        let xDesc = MPSMatrixDescriptor(rows: batch, columns: inputCols, rowBytes: xRowBytes, dataType: .float16)
        let wDesc = MPSMatrixDescriptor(rows: outFeatures, columns: inputCols, rowBytes: weightRowBytes, dataType: .float16)
        let yDesc = MPSMatrixDescriptor(rows: batch, columns: outFeatures, rowBytes: yRowBytes, dataType: .float16)
        let xMat = MPSMatrix(buffer: xBuffer, descriptor: xDesc)
        let wMat = MPSMatrix(buffer: weightBuffer, descriptor: wDesc)
        let yMat = MPSMatrix(buffer: yBuffer, descriptor: yDesc)
        let mm = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: true,
            resultRows: batch,
            resultColumns: outFeatures,
            interiorColumns: inputCols,
            alpha: 1.0,
            beta: 0.0
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Linear.forward: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Linear.forward"
        mm.encode(commandBuffer: commandBuffer, leftMatrix: xMat, rightMatrix: wMat, resultMatrix: yMat)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var yHalf = [Float16](repeating: 0, count: batch * outFeatures)
        yHalf.withUnsafeMutableBytes { raw in
            if let base = raw.baseAddress {
                for row in 0..<batch {
                    let dst = base.advanced(by: row * outFeatures * elemHalf)
                    let src = yBuffer.contents().advanced(by: row * yRowBytes)
                    memcpy(dst, src, outFeatures * elemHalf)
                }
            }
        }
        var output = [Float](repeating: 0, count: batch * outFeatures)
        for i in 0..<output.count { output[i] = Float(yHalf[i]) }
        return Tensor(shape: [batch, outFeatures], data: output)
    }

    public func linearGradients(key: UUID,
                                version: UInt64,
                                inFeatures: Int,
                                outFeatures: Int,
                                weight: Tensor,
                                X: Tensor,
                                dY: Tensor,
                                bias: Tensor?) throws -> (Tensor, Tensor) {
        precondition(X.shape.count == 2 && dY.shape.count == 2, "linearGradients expects 2D tensors")
        let batch = X.shape[0]
        precondition(X.shape[1] == inFeatures && dY.shape[0] == batch && dY.shape[1] == outFeatures,
                     "linearGradients shape mismatch")
        if batch == 0 {
            return (Tensor.zeros([outFeatures, inFeatures]), Tensor.zeros([outFeatures]))
        }
        _ = try ensureLinearCache(
            key: key,
            version: version,
            inFeatures: inFeatures,
            outFeatures: outFeatures,
            weight: weight
        )
        let elemHalf = MemoryLayout<Float16>.size
        let rowsL = outFeatures
        let colsL = batch
        let rowsR = batch
        let colsR = inFeatures
        let rowsY = outFeatures
        let colsY = inFeatures
        let lRowBytes = alignedRowBytes(columns: colsL, elemSize: elemHalf)
        let rRowBytes = alignedRowBytes(columns: colsR, elemSize: elemHalf)
        let yRowBytes = alignedRowBytes(columns: colsY, elemSize: elemHalf)
        let lBuffer = buffer(length: rowsL * lRowBytes, label: "GPUActor.Linear.grad.L.\(key)")
        let rBuffer = buffer(length: rowsR * rRowBytes, label: "GPUActor.Linear.grad.R.\(key)")
        let yBuffer = buffer(length: rowsY * yRowBytes, label: "GPUActor.Linear.grad.Y.\(key)")
        memset(lBuffer.contents(), 0, rowsL * lRowBytes)
        memset(rBuffer.contents(), 0, rowsR * rRowBytes)
        memset(yBuffer.contents(), 0, rowsY * yRowBytes)
        var dyHalf = [Float16](repeating: 0, count: rowsL * colsL)
        for bIdx in 0..<batch {
            let srcBase = bIdx * outFeatures
            for outIdx in 0..<outFeatures {
                dyHalf[outIdx * colsL + bIdx] = Float16(dY.data[srcBase + outIdx])
            }
        }
        dyHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                for row in 0..<rowsL {
                    let src = base.advanced(by: row * colsL * elemHalf)
                    let dst = lBuffer.contents().advanced(by: row * lRowBytes)
                    memcpy(dst, src, colsL * elemHalf)
                }
            }
        }
        var xHalf = [Float16](repeating: 0, count: rowsR * colsR)
        for bIdx in 0..<batch {
            let srcBase = bIdx * inFeatures
            let dstBase = bIdx * inFeatures
            for c in 0..<inFeatures {
                xHalf[dstBase + c] = Float16(X.data[srcBase + c])
            }
        }
        xHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                for row in 0..<rowsR {
                    let src = base.advanced(by: row * colsR * elemHalf)
                    let dst = rBuffer.contents().advanced(by: row * rRowBytes)
                    memcpy(dst, src, colsR * elemHalf)
                }
            }
        }
        let lDesc = MPSMatrixDescriptor(rows: rowsL, columns: colsL, rowBytes: lRowBytes, dataType: .float16)
        let rDesc = MPSMatrixDescriptor(rows: rowsR, columns: colsR, rowBytes: rRowBytes, dataType: .float16)
        let yDesc = MPSMatrixDescriptor(rows: rowsY, columns: colsY, rowBytes: yRowBytes, dataType: .float16)
        let lMat = MPSMatrix(buffer: lBuffer, descriptor: lDesc)
        let rMat = MPSMatrix(buffer: rBuffer, descriptor: rDesc)
        let yMat = MPSMatrix(buffer: yBuffer, descriptor: yDesc)
        let mm = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: rowsY,
            resultColumns: colsY,
            interiorColumns: colsL,
            alpha: 1.0,
            beta: 0.0
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Linear.gradients: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Linear.gradients"
        mm.encode(commandBuffer: commandBuffer, leftMatrix: lMat, rightMatrix: rMat, resultMatrix: yMat)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var dWHalf = [Float16](repeating: 0, count: rowsY * colsY)
        dWHalf.withUnsafeMutableBytes { raw in
            if let base = raw.baseAddress {
                for row in 0..<rowsY {
                    let dst = base.advanced(by: row * colsY * elemHalf)
                    let src = yBuffer.contents().advanced(by: row * yRowBytes)
                    memcpy(dst, src, colsY * elemHalf)
                }
            }
        }
        var dWHost = [Float](repeating: 0, count: rowsY * colsY)
        for i in 0..<dWHost.count { dWHost[i] = Float(dWHalf[i]) }
        var dBHost = [Float](repeating: 0, count: outFeatures)
        for bIdx in 0..<batch {
            let base = bIdx * outFeatures
            for o in 0..<outFeatures {
                dBHost[o] += dY.data[base + o]
            }
        }
        let dW = Tensor(shape: [outFeatures, inFeatures], data: dWHost)
        let dB = Tensor(shape: [outFeatures], data: dBHost)
        return (dW, dB)
    }

    public func linearInputGradients(key: UUID,
                                     version: UInt64,
                                     inFeatures: Int,
                                     outFeatures: Int,
                                     weight: Tensor,
                                     bias: Tensor?,
                                     dY: Tensor) throws -> Tensor {
        precondition(dY.shape.count == 2 && dY.shape[1] == outFeatures, "linearInputGradients expects [B, outFeatures]")
        let batch = dY.shape[0]
        if batch == 0 { return Tensor.zeros([0, inFeatures]) }
        let cache = try ensureLinearCache(
            key: key,
            version: version,
            inFeatures: inFeatures,
            outFeatures: outFeatures,
            weight: weight
        )
        let hasBias = bias != nil
        let inAug = hasBias ? (inFeatures + 1) : inFeatures
        let elemHalf = MemoryLayout<Float16>.size
        let dyRowBytes = alignedRowBytes(columns: outFeatures, elemSize: elemHalf)
        let dxRowBytesAug = alignedRowBytes(columns: inAug, elemSize: elemHalf)
        let dyBuffer = buffer(length: batch * dyRowBytes, label: "GPUActor.Linear.inputGrad.dY.\(key)")
        let dxBuffer = buffer(length: batch * dxRowBytesAug, label: "GPUActor.Linear.inputGrad.dx.\(key)")
        memset(dyBuffer.contents(), 0, batch * dyRowBytes)
        memset(dxBuffer.contents(), 0, batch * dxRowBytesAug)
        var dyHalf = [Float16](repeating: 0, count: batch * outFeatures)
        for row in 0..<batch {
            let base = row * outFeatures
            for col in 0..<outFeatures {
                dyHalf[base + col] = Float16(dY.data[base + col])
            }
        }
        dyHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                for row in 0..<batch {
                    let src = base.advanced(by: row * outFeatures * elemHalf)
                    let dst = dyBuffer.contents().advanced(by: row * dyRowBytes)
                    memcpy(dst, src, outFeatures * elemHalf)
                }
            }
        }
        let weightBuffer: MTLBuffer
        let weightRowBytes: Int
        if hasBias, let biasTensor = bias {
            let augmentedRowBytes = alignedRowBytes(columns: inAug, elemSize: elemHalf)
            let wAugBuffer = buffer(length: outFeatures * augmentedRowBytes, label: "GPUActor.Linear.inputGrad.wAug.\(key)")
            memset(wAugBuffer.contents(), 0, outFeatures * augmentedRowBytes)
            for row in 0..<outFeatures {
                let src = cache.weightBuffer.contents().advanced(by: row * cache.rowBytes)
                let dst = wAugBuffer.contents().advanced(by: row * augmentedRowBytes)
                memcpy(dst, src, inFeatures * elemHalf)
            }
            var biasHalf = [Float16](repeating: 0, count: outFeatures)
            for i in 0..<outFeatures { biasHalf[i] = Float16(biasTensor.data[i]) }
            biasHalf.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    for row in 0..<outFeatures {
                        let dst = wAugBuffer.contents().advanced(by: row * augmentedRowBytes + inFeatures * elemHalf)
                        memcpy(dst, base.advanced(by: row * elemHalf), elemHalf)
                    }
                }
            }
            weightBuffer = wAugBuffer
            weightRowBytes = augmentedRowBytes
        } else {
            weightBuffer = cache.weightBuffer
            weightRowBytes = cache.rowBytes
        }
        let dyDesc = MPSMatrixDescriptor(rows: batch, columns: outFeatures, rowBytes: dyRowBytes, dataType: .float16)
        let wDesc = MPSMatrixDescriptor(rows: outFeatures, columns: inAug, rowBytes: weightRowBytes, dataType: .float16)
        let dxDesc = MPSMatrixDescriptor(rows: batch, columns: inAug, rowBytes: dxRowBytesAug, dataType: .float16)
        let dyMat = MPSMatrix(buffer: dyBuffer, descriptor: dyDesc)
        let wMat = MPSMatrix(buffer: weightBuffer, descriptor: wDesc)
        let dxMat = MPSMatrix(buffer: dxBuffer, descriptor: dxDesc)
        let mm = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: batch,
            resultColumns: inAug,
            interiorColumns: outFeatures,
            alpha: 1.0,
            beta: 0.0
        )
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Linear.inputGradients: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Linear.inputGradients"
        mm.encode(commandBuffer: commandBuffer, leftMatrix: dyMat, rightMatrix: wMat, resultMatrix: dxMat)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var dxHalfAug = [Float16](repeating: 0, count: batch * inAug)
        dxHalfAug.withUnsafeMutableBytes { raw in
            if let base = raw.baseAddress {
                for row in 0..<batch {
                    let dst = base.advanced(by: row * inAug * elemHalf)
                    let src = dxBuffer.contents().advanced(by: row * dxRowBytesAug)
                    memcpy(dst, src, inAug * elemHalf)
                }
            }
        }
        var dxHost = [Float](repeating: 0, count: batch * inFeatures)
        for row in 0..<batch {
            let srcBase = row * inAug
            let dstBase = row * inFeatures
            for c in 0..<inFeatures {
                dxHost[dstBase + c] = Float(dxHalfAug[srcBase + c])
            }
        }
        return Tensor(shape: [batch, inFeatures], data: dxHost)
    }

    public func layerNormForward(x: Tensor, gamma: Tensor, beta: Tensor, eps: Float) throws -> Tensor {
        precondition(x.shape.count == 2, "GPUActor.layerNormForward expects [N,D]")
        precondition(gamma.shape == [x.shape[1]] && beta.shape == [x.shape[1]], "LayerNorm gamma/beta mismatch")
        let pipelines = try ensureLayerNormPipelines()
        let N = x.shape[0]
        let D = x.shape[1]
        let count = N * D
        if count == 0 { return x }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("LayerNorm.forward: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.LayerNorm.forward"
        let elemHalf = MemoryLayout<Float16>.size
        let elemFloat = MemoryLayout<Float>.size
        let xBytes = count * elemHalf
        let yBytes = count * elemHalf
        let meanBytes = N * elemFloat
        let invStdBytes = N * elemFloat
        let gammaBytes = D * elemFloat
        let betaBytes = D * elemFloat
        let xBuffer = buffer(length: xBytes, label: "GPUActor.LayerNorm.forward.x")
        let yBuffer = buffer(length: yBytes, label: "GPUActor.LayerNorm.forward.y")
        let meanBuffer = buffer(length: meanBytes, label: "GPUActor.LayerNorm.forward.mean")
        let invStdBuffer = buffer(length: invStdBytes, label: "GPUActor.LayerNorm.forward.invstd")
        let gammaBuffer = buffer(length: gammaBytes, label: "GPUActor.LayerNorm.forward.gamma")
        let betaBuffer = buffer(length: betaBytes, label: "GPUActor.LayerNorm.forward.beta")
        var xHalf = [Float16](repeating: 0, count: count)
        for i in 0..<count { xHalf[i] = Float16(x.data[i]) }
        xHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(xBuffer.contents(), base, xBytes)
            }
        }
        gamma.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(gammaBuffer.contents(), base, gammaBytes)
            }
        }
        beta.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(betaBuffer.contents(), base, betaBytes)
            }
        }
        memset(meanBuffer.contents(), 0, meanBytes)
        memset(invStdBuffer.contents(), 0, invStdBytes)
        memset(yBuffer.contents(), 0, yBytes)
        guard let statsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("LayerNorm.forward: stats encoder creation failed")
        }
        statsEncoder.label = "GPUActor.LayerNorm.forward.stats"
        statsEncoder.setComputePipelineState(pipelines.stats)
        statsEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        statsEncoder.setBuffer(meanBuffer, offset: 0, index: 1)
        statsEncoder.setBuffer(invStdBuffer, offset: 0, index: 2)
        var vN = Int32(N)
        var vD = Int32(D)
        var vEps = eps
        statsEncoder.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 3)
        statsEncoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 4)
        statsEncoder.setBytes(&vEps, length: MemoryLayout<Float>.size, index: 5)
        let statsThreadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let statsThreadgroups = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        statsEncoder.dispatchThreadgroups(statsThreadgroups, threadsPerThreadgroup: statsThreadsPerGroup)
        statsEncoder.endEncoding()
        guard let normEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("LayerNorm.forward: normalize encoder creation failed")
        }
        normEncoder.label = "GPUActor.LayerNorm.forward.norm"
        normEncoder.setComputePipelineState(pipelines.normalize)
        normEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        normEncoder.setBuffer(yBuffer, offset: 0, index: 1)
        normEncoder.setBuffer(gammaBuffer, offset: 0, index: 2)
        normEncoder.setBuffer(betaBuffer, offset: 0, index: 3)
        normEncoder.setBuffer(meanBuffer, offset: 0, index: 4)
        normEncoder.setBuffer(invStdBuffer, offset: 0, index: 5)
        normEncoder.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 6)
        normEncoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 7)
        let normThreadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let normThreadgroups = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        normEncoder.dispatchThreadgroups(normThreadgroups, threadsPerThreadgroup: normThreadsPerGroup)
        normEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var yHalf = [Float16](repeating: 0, count: count)
        memcpy(&yHalf, yBuffer.contents(), yBytes)
        var output = [Float](repeating: 0, count: count)
        for i in 0..<count { output[i] = Float(yHalf[i]) }
        return Tensor(shape: x.shape, data: output)
    }

    public func layerNormBackward(x: Tensor, g: Tensor, gamma: Tensor, eps: Float) throws -> (Tensor, Tensor, Tensor) {
        precondition(x.shape.count == 2 && g.shape == x.shape, "GPUActor.layerNormBackward expects x,g [N,D]")
        precondition(gamma.shape == [x.shape[1]], "GPUActor.layerNormBackward gamma mismatch")
        let pipelines = try ensureLayerNormPipelines()
        let N = x.shape[0]
        let D = x.shape[1]
        let count = N * D
        if count == 0 {
            return (Tensor.zeros(x.shape), Tensor.zeros([D]), Tensor.zeros([D]))
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("LayerNorm.backward: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.LayerNorm.backward"
        let elemHalf = MemoryLayout<Float16>.size
        let elemFloat = MemoryLayout<Float>.size
        let xBytes = count * elemHalf
        let gBytes = count * elemHalf
        let gammaBytes = D * elemFloat
        let meanBytes = N * elemFloat
        let invStdBytes = N * elemFloat
        let sumBytes = N * elemFloat
        let dxBytes = count * elemHalf
        let dGammaBytes = D * elemFloat
        let dBetaBytes = D * elemFloat
        let xBuffer = buffer(length: xBytes, label: "GPUActor.LayerNorm.backward.x")
        let gBuffer = buffer(length: gBytes, label: "GPUActor.LayerNorm.backward.g")
        let gammaBuffer = buffer(length: gammaBytes, label: "GPUActor.LayerNorm.backward.gamma")
        let meanBuffer = buffer(length: meanBytes, label: "GPUActor.LayerNorm.backward.mean")
        let invStdBuffer = buffer(length: invStdBytes, label: "GPUActor.LayerNorm.backward.invstd")
        let sumGBuffer = buffer(length: sumBytes, label: "GPUActor.LayerNorm.backward.sumG")
        let sumGXhatBuffer = buffer(length: sumBytes, label: "GPUActor.LayerNorm.backward.sumGXhat")
        let dxBuffer = buffer(length: dxBytes, label: "GPUActor.LayerNorm.backward.dx")
        let dGammaBuffer = buffer(length: dGammaBytes, label: "GPUActor.LayerNorm.backward.dgamma")
        let dBetaBuffer = buffer(length: dBetaBytes, label: "GPUActor.LayerNorm.backward.dbeta")
        var xHalf = [Float16](repeating: 0, count: count)
        var gHalf = [Float16](repeating: 0, count: count)
        for i in 0..<count {
            xHalf[i] = Float16(x.data[i])
            gHalf[i] = Float16(g.data[i])
        }
        xHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(xBuffer.contents(), base, xBytes)
            }
        }
        gHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(gBuffer.contents(), base, gBytes)
            }
        }
        gamma.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(gammaBuffer.contents(), base, gammaBytes)
            }
        }
        memset(meanBuffer.contents(), 0, meanBytes)
        memset(invStdBuffer.contents(), 0, invStdBytes)
        memset(sumGBuffer.contents(), 0, sumBytes)
        memset(sumGXhatBuffer.contents(), 0, sumBytes)
        memset(dxBuffer.contents(), 0, dxBytes)
        memset(dGammaBuffer.contents(), 0, dGammaBytes)
        memset(dBetaBuffer.contents(), 0, dBetaBytes)
        guard let rowEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("LayerNorm.backward: row encoder creation failed")
        }
        rowEncoder.label = "GPUActor.LayerNorm.backward.row"
        rowEncoder.setComputePipelineState(pipelines.backwardRow)
        rowEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        rowEncoder.setBuffer(gBuffer, offset: 0, index: 1)
        rowEncoder.setBuffer(gammaBuffer, offset: 0, index: 2)
        rowEncoder.setBuffer(meanBuffer, offset: 0, index: 3)
        rowEncoder.setBuffer(invStdBuffer, offset: 0, index: 4)
        rowEncoder.setBuffer(sumGBuffer, offset: 0, index: 5)
        rowEncoder.setBuffer(sumGXhatBuffer, offset: 0, index: 6)
        var vN = Int32(N)
        var vD = Int32(D)
        var vEps = eps
        rowEncoder.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 7)
        rowEncoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 8)
        rowEncoder.setBytes(&vEps, length: MemoryLayout<Float>.size, index: 9)
        let rowThreadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let rowThreadgroups = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        rowEncoder.dispatchThreadgroups(rowThreadgroups, threadsPerThreadgroup: rowThreadsPerGroup)
        rowEncoder.endEncoding()
        guard let dxEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("LayerNorm.backward: dx encoder creation failed")
        }
        dxEncoder.label = "GPUActor.LayerNorm.backward.dx"
        dxEncoder.setComputePipelineState(pipelines.backwardDX)
        dxEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        dxEncoder.setBuffer(gBuffer, offset: 0, index: 1)
        dxEncoder.setBuffer(gammaBuffer, offset: 0, index: 2)
        dxEncoder.setBuffer(meanBuffer, offset: 0, index: 3)
        dxEncoder.setBuffer(invStdBuffer, offset: 0, index: 4)
        dxEncoder.setBuffer(sumGBuffer, offset: 0, index: 5)
        dxEncoder.setBuffer(sumGXhatBuffer, offset: 0, index: 6)
        dxEncoder.setBuffer(dxBuffer, offset: 0, index: 7)
        dxEncoder.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 8)
        dxEncoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 9)
        let dxThreadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let dxThreadgroups = MTLSize(width: (N + 255) / 256, height: 1, depth: 1)
        dxEncoder.dispatchThreadgroups(dxThreadgroups, threadsPerThreadgroup: dxThreadsPerGroup)
        dxEncoder.endEncoding()
        guard let dgbEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("LayerNorm.backward: dgamma/dbeta encoder creation failed")
        }
        dgbEncoder.label = "GPUActor.LayerNorm.backward.dgb"
        dgbEncoder.setComputePipelineState(pipelines.backwardDGamma)
        dgbEncoder.setBuffer(xBuffer, offset: 0, index: 0)
        dgbEncoder.setBuffer(gBuffer, offset: 0, index: 1)
        dgbEncoder.setBuffer(meanBuffer, offset: 0, index: 2)
        dgbEncoder.setBuffer(invStdBuffer, offset: 0, index: 3)
        dgbEncoder.setBuffer(dGammaBuffer, offset: 0, index: 4)
        dgbEncoder.setBuffer(dBetaBuffer, offset: 0, index: 5)
        dgbEncoder.setBytes(&vN, length: MemoryLayout<Int32>.size, index: 6)
        dgbEncoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 7)
        let dgbThreadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let dgbThreadgroups = MTLSize(width: (D + 255) / 256, height: 1, depth: 1)
        dgbEncoder.dispatchThreadgroups(dgbThreadgroups, threadsPerThreadgroup: dgbThreadsPerGroup)
        dgbEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var dxHalf = [Float16](repeating: 0, count: count)
        memcpy(&dxHalf, dxBuffer.contents(), dxBytes)
        var dxHost = [Float](repeating: 0, count: count)
        for i in 0..<count { dxHost[i] = Float(dxHalf[i]) }
        var dGammaHost = [Float](repeating: 0, count: D)
        dGammaHost.withUnsafeMutableBytes { dest in
            if let base = dest.baseAddress {
                memcpy(base, dGammaBuffer.contents(), dGammaBytes)
            }
        }
        var dBetaHost = [Float](repeating: 0, count: D)
        dBetaHost.withUnsafeMutableBytes { dest in
            if let base = dest.baseAddress {
                memcpy(base, dBetaBuffer.contents(), dBetaBytes)
            }
        }
        let dxTensor = Tensor(shape: [N, D], data: dxHost)
        let dGammaTensor = Tensor(shape: [D], data: dGammaHost)
        let dBetaTensor = Tensor(shape: [D], data: dBetaHost)
        return (dxTensor, dGammaTensor, dBetaTensor)
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

private struct LinearCacheEntry {
    let version: UInt64
    let weightBuffer: MTLBuffer
    let rowBytes: Int
    let inFeatures: Int
    let outFeatures: Int
}

private struct GELUPipelines {
    let library: MTLLibrary
    let forward: MTLComputePipelineState
    let backward: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: GELUMetalLibrary.source, options: nil)
        guard let forwardFunction = library.makeFunction(name: "gelu_tanh_fp16") else {
            throw GPUActorError.pipelineFunctionMissing("gelu_tanh_fp16")
        }
        guard let backwardFunction = library.makeFunction(name: "gelu_tanh_bwd_fp16") else {
            throw GPUActorError.pipelineFunctionMissing("gelu_tanh_bwd_fp16")
        }
        let forwardState = try device.makeComputePipelineState(function: forwardFunction)
        let backwardState = try device.makeComputePipelineState(function: backwardFunction)
        self.library = library
        self.forward = forwardState
        self.backward = backwardState
    }
}

private struct LayerNormPipelines {
    let library: MTLLibrary
    let stats: MTLComputePipelineState
    let normalize: MTLComputePipelineState
    let backwardRow: MTLComputePipelineState
    let backwardDX: MTLComputePipelineState
    let backwardDGamma: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: LayerNormMetalLibrary.source, options: nil)
        guard let statsFunction = library.makeFunction(name: "ln_compute_stats_fp16") else {
            throw GPUActorError.pipelineFunctionMissing("ln_compute_stats_fp16")
        }
        guard let normFunction = library.makeFunction(name: "ln_normalize_affine_fp16") else {
            throw GPUActorError.pipelineFunctionMissing("ln_normalize_affine_fp16")
        }
        guard let rowFunction = library.makeFunction(name: "ln_bwd_row_sums_fp16") else {
            throw GPUActorError.pipelineFunctionMissing("ln_bwd_row_sums_fp16")
        }
        guard let dxFunction = library.makeFunction(name: "ln_bwd_dx_fp16") else {
            throw GPUActorError.pipelineFunctionMissing("ln_bwd_dx_fp16")
        }
        guard let dgbFunction = library.makeFunction(name: "ln_bwd_dgamma_dbeta_f32") else {
            throw GPUActorError.pipelineFunctionMissing("ln_bwd_dgamma_dbeta_f32")
        }
        let statsState = try device.makeComputePipelineState(function: statsFunction)
        let normState = try device.makeComputePipelineState(function: normFunction)
        let rowState = try device.makeComputePipelineState(function: rowFunction)
        let dxState = try device.makeComputePipelineState(function: dxFunction)
        let dgbState = try device.makeComputePipelineState(function: dgbFunction)
        self.library = library
        self.stats = statsState
        self.normalize = normState
        self.backwardRow = rowState
        self.backwardDX = dxState
        self.backwardDGamma = dgbState
    }
}

private struct EmbeddingPipelines {
    let library: MTLLibrary
    let gather: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: EmbeddingMetalLibrary.source, options: nil)
        guard let gatherFunction = library.makeFunction(name: "embedding_gather_f32") else {
            throw GPUActorError.pipelineFunctionMissing("embedding_gather_f32")
        }
        let gatherState = try device.makeComputePipelineState(function: gatherFunction)
        self.library = library
        self.gather = gatherState
    }
}
