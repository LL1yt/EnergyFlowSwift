import Metal

extension GPUActor {
    public func geluForward(x: Tensor) async throws -> Tensor {
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
        
        let reader = Float16BufferReader(
            buffer: yBuffer,
            count: count,
            shape: x.shape
        )
        
        return try await awaitCommandBufferWithReader(
            label: "GPUActor.GELU.forward",
            commandBuffer: commandBuffer,
            reader: reader
        )
    }

    public func geluBackward(x: Tensor, dY: Tensor) async throws -> Tensor {
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
        
        let reader = Float16BufferReader(
            buffer: dxBuffer,
            count: count,
            shape: x.shape
        )
        
        return try await awaitCommandBufferWithReader(
            label: "GPUActor.GELU.backward",
            commandBuffer: commandBuffer,
            reader: reader
        )
    }

    public func geluForwardDeferred(x: Tensor, deferUntilSync: Bool = true) async throws -> GPUReadback<Tensor> {
        let count = x.count
        if count == 0 { return GPUReadback(resolved: x) }
        let pipelines = try ensureGELUPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("GELU.forwardDeferred: command buffer creation failed")
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
            throw GPUActorError.commandBufferUnavailable("GELU.forwardDeferred: encoder creation failed")
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
        let reader = Float16BufferReader(buffer: yBuffer, count: count, shape: x.shape)
        return scheduleCommandBufferWithReader(
            label: "GPUActor.GELU.forward",
            commandBuffer: commandBuffer,
            deferUntilSync: deferUntilSync,
            reader: reader
        )
    }

    // New: GELU from a GPU handle, returning a handle to the result (fp16)
    public func geluForwardHandleDeferred(xHandle: GPUTensorHandle,
                                          outputShape: [Int]? = nil,
                                          consumeInput: Bool = false,
                                          deferUntilSync: Bool = true) async throws -> GPUReadback<GPUTensorHandle> {
        let count = xHandle.rows * xHandle.cols
        if count == 0 {
            let emptyHandle = registerHandle(buffer: buffer(length: 1, label: "GELU.forwardHandle.empty"),
                                             shape: outputShape ?? [0, 0],
                                             rows: xHandle.rows,
                                             cols: xHandle.cols,
                                             rowBytes: 0,
                                             elemType: .float16,
                                             label: "GELU.empty")
            return GPUReadback(resolved: emptyHandle)
        }
        let pipelines = try ensureGELUPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("GELU.forwardHandle: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.GELU.forwardHandle"
        let elemHalf = MemoryLayout<Float16>.size
        let elemFloat = MemoryLayout<Float>.size
        // Prepare contiguous fp16 input buffer
        let xInBuf = consumeInput
            ? consumeHandle(xHandle)
            : peekHandle(xHandle)
        guard let xHalfBuf = device.makeBuffer(length: count * elemHalf, options: .storageModeShared) else {
            throw GPUActorError.commandBufferUnavailable("GELU.forwardHandle: temp xHalf allocation failed")
        }
        xHalfBuf.label = "GPUActor.GELU.forwardHandle.xHalf"
        if xHandle.elemType == .float16 {
            // Row-wise copy respecting rowBytes
            for row in 0..<xHandle.rows {
                let src = xInBuf.contents().advanced(by: row * xHandle.rowBytes)
                let dst = xHalfBuf.contents().advanced(by: row * xHandle.cols * elemHalf)
                memcpy(dst, src, xHandle.cols * elemHalf)
            }
        } else {
            // Convert float32 -> float16 row-wise
            for row in 0..<xHandle.rows {
                let src = xInBuf.contents().advanced(by: row * xHandle.rowBytes).bindMemory(to: Float.self, capacity: xHandle.cols)
                var halfRow = [Float16](repeating: 0, count: xHandle.cols)
                for c in 0..<xHandle.cols { halfRow[c] = Float16(src[c]) }
                halfRow.withUnsafeBytes { raw in
                    if let base = raw.baseAddress {
                        let dst = xHalfBuf.contents().advanced(by: row * xHandle.cols * elemHalf)
                        memcpy(dst, base, xHandle.cols * elemHalf)
                    }
                }
            }
        }
        // Output contiguous fp16 buffer
        guard let yBuf = device.makeBuffer(length: count * elemHalf, options: .storageModeShared) else {
            throw GPUActorError.commandBufferUnavailable("GELU.forwardHandle: y buffer allocation failed")
        }
        yBuf.label = "GPUActor.GELU.forwardHandle.y"
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("GELU.forwardHandle: encoder creation failed")
        }
        encoder.label = "GPUActor.GELU.forwardHandle.encoder"
        encoder.setComputePipelineState(pipelines.forward)
        encoder.setBuffer(xHalfBuf, offset: 0, index: 0)
        encoder.setBuffer(yBuf, offset: 0, index: 1)
        var n = Int32(count)
        encoder.setBytes(&n, length: MemoryLayout<Int32>.size, index: 2)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        let handle = registerHandle(buffer: yBuf,
                                    shape: outputShape ?? [xHandle.rows, xHandle.cols],
                                    rows: xHandle.rows,
                                    cols: xHandle.cols,
                                    rowBytes: xHandle.cols * elemHalf,
                                    elemType: .float16,
                                    label: "GELU.forwardHandle.output")
        return scheduleCommandBuffer(label: "GPUActor.GELU.forwardHandle",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
            handle
        }
    }

    private func ensureGELUPipelines() throws -> GELUPipelines {
        if let pipelines = geluPipelines {
            return pipelines
        }
        let pipelines = try GELUPipelines(device: device)
        geluPipelines = pipelines
        return pipelines
    }
}

struct GELUPipelines {
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
        self.library = library
        self.forward = try device.makeComputePipelineState(function: forwardFunction)
        self.backward = try device.makeComputePipelineState(function: backwardFunction)
    }
}
