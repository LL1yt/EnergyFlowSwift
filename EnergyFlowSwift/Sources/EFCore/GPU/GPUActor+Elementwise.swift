import Metal

extension GPUActor {
    public func residualAdd(y: Tensor, x: Tensor) async throws -> Tensor {
        let readback = try await residualAddDeferred(y: y, x: x, deferUntilSync: false)
        return try await readback.value()
    }

    public func residualAddDeferred(y: Tensor,
                                    x: Tensor,
                                    deferUntilSync: Bool = true) async throws -> GPUReadback<Tensor> {
        precondition(y.shape == x.shape, "GPUActor.residualAdd shape mismatch")
        let count = y.count
        if count == 0 { return GPUReadback(resolved: y) }
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
        
        let reader = FloatBufferReader(buffer: yBuffer, count: count, shape: y.shape)
        return scheduleCommandBufferWithReader(
            label: "GPUActor.Elementwise.residualAdd",
            commandBuffer: commandBuffer,
            deferUntilSync: deferUntilSync,
            reader: reader
        )
    }

    public func residualAddHandleDeferred(yHandle: GPUTensorHandle,
                                          xHandle: GPUTensorHandle,
                                          outputShape: [Int]? = nil,
                                          consumeInputs: Bool = false,
                                          deferUntilSync: Bool = true) async throws -> GPUReadback<GPUTensorHandle> {
        precondition(yHandle.rows == xHandle.rows && yHandle.cols == xHandle.cols,
                     "residualAddHandle expects matching handle dimensions")
        let rows = yHandle.rows
        let cols = yHandle.cols
        let count = rows * cols
        let shape = outputShape ?? (yHandle.shape.isEmpty ? xHandle.shape : yHandle.shape)
        let ySource = consumeInputs
            ? consumeHandle(yHandle, expectRows: rows, expectCols: cols)
            : peekHandle(yHandle, expectRows: rows, expectCols: cols)
        let xSource = consumeInputs
            ? consumeHandle(xHandle, expectRows: rows, expectCols: cols)
            : peekHandle(xHandle, expectRows: rows, expectCols: cols)
        if count == 0 {
            let emptyBuffer = buffer(length: 16, label: "GPUActor.Elementwise.residualAddHandle.emptyBuffer")
            memset(emptyBuffer.contents(), 0, 16)
            let handle = registerHandle(buffer: emptyBuffer,
                                        shape: shape,
                                        rows: rows,
                                        cols: cols,
                                        rowBytes: 0,
                                        elemType: .float32,
                                        label: "Elementwise.residualAddHandle.empty")
            return GPUReadback(resolved: handle)
        }
        let yBuffer = try packHandleToFloat32Contiguous(sourceBuffer: ySource,
                                                        rows: rows,
                                                        cols: cols,
                                                        rowBytes: yHandle.rowBytes,
                                                        elemType: yHandle.elemType,
                                                        label: "GPUActor.Elementwise.residualAddHandle.y")
        let xBuffer = try packHandleToFloat32Contiguous(sourceBuffer: xSource,
                                                        rows: rows,
                                                        cols: cols,
                                                        rowBytes: xHandle.rowBytes,
                                                        elemType: xHandle.elemType,
                                                        label: "GPUActor.Elementwise.residualAddHandle.x")
        let pipelines = try ensureElementwisePipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.residualAddHandle: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Elementwise.residualAddHandle"
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.residualAddHandle: encoder creation failed")
        }
        encoder.label = "GPUActor.Elementwise.residualAddHandle.encoder"
        encoder.setComputePipelineState(pipelines.residualAdd)
        encoder.setBuffer(yBuffer, offset: 0, index: 0)
        encoder.setBuffer(xBuffer, offset: 0, index: 1)
        var n = Int32(count)
        encoder.setBytes(&n, length: MemoryLayout<Int32>.size, index: 2)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        let handle = registerHandle(buffer: yBuffer,
                                    shape: shape,
                                    rows: rows,
                                    cols: cols,
                                    rowBytes: cols * MemoryLayout<Float>.size,
                                    elemType: .float32,
                                    label: "Elementwise.residualAddHandle.output")
        return scheduleCommandBuffer(label: "GPUActor.Elementwise.residualAddHandle",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
            handle
        }
    }

    public func addBroadcast2DInto3D(y: Tensor, addBD: Tensor, sequenceLength L: Int) async throws -> Tensor {
        let readback = try await addBroadcast2DInto3DDeferred(y: y,
                                                               addBD: addBD,
                                                               sequenceLength: L,
                                                               deferUntilSync: false)
        return try await readback.value()
    }

    public func addBroadcast2DInto3DDeferred(y: Tensor,
                                             addBD: Tensor,
                                             sequenceLength L: Int,
                                             deferUntilSync: Bool = true) async throws -> GPUReadback<Tensor> {
        precondition(y.shape.count == 3, "addBroadcast2DInto3D expects y [B,L,D]")
        let B = y.shape[0], seq = y.shape[1], D = y.shape[2]
        precondition(seq == L, "sequenceLength mismatch: provided \(L) vs tensor \(seq)")
        precondition(addBD.shape == [B, D], "addBroadcast2DInto3D expects add [B,D]")
        let count = B * L * D
        if count == 0 { return GPUReadback(resolved: y) }
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
        
        let reader = FloatBufferReader(buffer: yBuffer, count: count, shape: y.shape)
        return scheduleCommandBufferWithReader(
            label: "GPUActor.Elementwise.addBroadcast2DInto3D",
            commandBuffer: commandBuffer,
            deferUntilSync: deferUntilSync,
            reader: reader
        )
    }

    public func maskZero(y: Tensor, mask: [[Int]]) async throws -> Tensor {
        let readback = try await maskZeroDeferred(y: y, mask: mask, deferUntilSync: false)
        return try await readback.value()
    }

    public func maskZeroDeferred(y: Tensor,
                                 mask: [[Int]],
                                 deferUntilSync: Bool = true) async throws -> GPUReadback<Tensor> {
        precondition(y.shape.count == 3, "maskZero expects y [B,L,D]")
        let B = y.shape[0], L = y.shape[1], D = y.shape[2]
        let maskFlat = flattenMask(mask, expectedBatch: B, expectedLength: L)
        let count = B * L * D
        if count == 0 { return GPUReadback(resolved: y) }
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
        
        struct ResultReader: Sendable {
            let bufferPtr: UInt
            let count: Int
            let yBytes: Int
            let shape: [Int]
            
            func read() -> Tensor {
                let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
                var out = [Float](repeating: 0, count: count)
                memcpy(&out, ptr, yBytes)
                return Tensor(shape: shape, data: out)
            }
        }
        
        let reader = ResultReader(
            bufferPtr: UInt(bitPattern: yBuffer.contents()),
            count: count,
            yBytes: yBytes,
            shape: y.shape
        )
        
        return scheduleCommandBuffer(label: "GPUActor.Elementwise.maskZero",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) { [reader] in
            return reader.read()
        }
    }

    public func maskZeroHandleDeferred(yHandle: GPUTensorHandle,
                                       mask: [[Int]],
                                       outputShape: [Int]? = nil,
                                       consumeInput: Bool = false,
                                       deferUntilSync: Bool = true) async throws -> GPUReadback<GPUTensorHandle> {
        let B = mask.count
        let L = mask.first?.count ?? 0
        let rowsExpected = B * L
        precondition(yHandle.rows == rowsExpected, "maskZeroHandle expects handle rows \(rowsExpected), got \(yHandle.rows)")
        let cols = yHandle.cols
        let count = yHandle.rows * cols
        let shape = outputShape ?? (yHandle.shape.count == 3 ? yHandle.shape : [B, L, cols])
        let ySource = consumeInput
            ? consumeHandle(yHandle, expectRows: yHandle.rows, expectCols: cols)
            : peekHandle(yHandle, expectRows: yHandle.rows, expectCols: cols)
        if count == 0 {
            let emptyBuffer = buffer(length: 16, label: "GPUActor.Elementwise.maskZeroHandle.emptyBuffer")
            memset(emptyBuffer.contents(), 0, 16)
            let handle = registerHandle(buffer: emptyBuffer,
                                        shape: shape,
                                        rows: yHandle.rows,
                                        cols: cols,
                                        rowBytes: 0,
                                        elemType: .float32,
                                        label: "Elementwise.maskZeroHandle.empty")
            return GPUReadback(resolved: handle)
        }
        let pipelines = try ensureElementwisePipelines()
        let maskFlat = flattenMask(mask, expectedBatch: B, expectedLength: L)
        let yBuffer = try packHandleToFloat32Contiguous(sourceBuffer: ySource,
                                                        rows: yHandle.rows,
                                                        cols: cols,
                                                        rowBytes: yHandle.rowBytes,
                                                        elemType: yHandle.elemType,
                                                        label: "GPUActor.Elementwise.maskZeroHandle.y")
        guard let maskBuffer = device.makeBuffer(length: maskFlat.count * MemoryLayout<Int32>.size,
                                                 options: .storageModeShared) else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskZeroHandle: mask buffer allocation failed")
        }
        maskBuffer.label = "GPUActor.Elementwise.maskZeroHandle.mask"
        maskFlat.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(maskBuffer.contents(), base, maskFlat.count * MemoryLayout<Int32>.size)
            }
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskZeroHandle: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Elementwise.maskZeroHandle"
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskZeroHandle: encoder creation failed")
        }
        encoder.label = "GPUActor.Elementwise.maskZeroHandle.encoder"
        encoder.setComputePipelineState(pipelines.maskZero)
        encoder.setBuffer(yBuffer, offset: 0, index: 0)
        encoder.setBuffer(maskBuffer, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vD = Int32(cols)
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 4)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        let handle = registerHandle(buffer: yBuffer,
                                    shape: shape,
                                    rows: yHandle.rows,
                                    cols: cols,
                                    rowBytes: cols * MemoryLayout<Float>.size,
                                    elemType: .float32,
                                    label: "Elementwise.maskZeroHandle.output")
        return scheduleCommandBuffer(label: "GPUActor.Elementwise.maskZeroHandle",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
            handle
        }
    }

    public func maskedMean(x: Tensor, mask: [[Int]]) async throws -> Tensor {
        let readback = try await maskedMeanDeferred(x: x, mask: mask, deferUntilSync: false)
        return try await readback.value()
    }

    public func maskedMeanDeferred(x: Tensor,
                                   mask: [[Int]],
                                   deferUntilSync: Bool = true) async throws -> GPUReadback<Tensor> {
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
        
        struct ResultReader: Sendable {
            let bufferPtr: UInt
            let B: Int
            let H: Int
            let yBytes: Int
            
            func read() -> Tensor {
                let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
                var output = [Float](repeating: 0, count: B * H)
                memcpy(&output, ptr, yBytes)
                return Tensor(shape: [B, H], data: output)
            }
        }
        
        let reader = ResultReader(
            bufferPtr: UInt(bitPattern: yBuffer.contents()),
            B: B,
            H: H,
            yBytes: yBytes
        )
        
        return scheduleCommandBuffer(label: "GPUActor.Elementwise.maskedMean",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) { [reader] in
            return reader.read()
        }
    }

    // New: return GPU handle to pooled buffer to chain with GPU ops without host copy
    public func maskedMeanHandleDeferred(x: Tensor,
                                         mask: [[Int]],
                                         deferUntilSync: Bool = true) async throws -> GPUReadback<GPUTensorHandle> {
        precondition(x.shape.count == 3, "maskedMeanHandleDeferred expects x [B,L,H]")
        let B = x.shape[0], L = x.shape[1], H = x.shape[2]
        let maskFlat = flattenMask(mask, expectedBatch: B, expectedLength: L)
        let pipelines = try ensureElementwisePipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskedMeanHandle: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Elementwise.maskedMeanHandle"
        let elem = MemoryLayout<Float>.size
        let xBytes = B * L * H * elem
        let yBytes = B * H * elem
        let maskBytes = maskFlat.count * MemoryLayout<Int32>.size
        let xBuffer = buffer(length: xBytes, label: "GPUActor.Elementwise.maskedMeanHandle.x")
        let yBuffer = buffer(length: yBytes, label: "GPUActor.Elementwise.maskedMeanHandle.y")
        let maskBuffer = buffer(length: maskBytes, label: "GPUActor.Elementwise.maskedMeanHandle.mask")
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
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskedMeanHandle: encoder creation failed")
        }
        encoder.label = "GPUActor.Elementwise.maskedMeanHandle.encoder"
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
        // Register handle before committing so closure doesn't capture MTLBuffer
        let handle = registerHandle(buffer: yBuffer,
                                    shape: [B, H],
                                    rows: B,
                                    cols: H,
                                    rowBytes: alignedRowBytes(columns: H, elemSize: elem),
                                    elemType: .float32,
                                    label: "Elementwise.maskedMeanHandle")
        return scheduleCommandBuffer(label: "GPUActor.Elementwise.maskedMeanHandle",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
            handle
        }
    }

    public func maskedMeanHandleFromHandleDeferred(xHandle: GPUTensorHandle,
                                                   mask: [[Int]],
                                                   outputShape: [Int]? = nil,
                                                   consumeInput: Bool = false,
                                                   deferUntilSync: Bool = true) async throws -> GPUReadback<GPUTensorHandle> {
        precondition(xHandle.shape.count == 3, "maskedMeanHandleFromHandleDeferred expects handle shape [B,L,H]")
        let B = xHandle.shape[0]
        let L = xHandle.shape[1]
        let H = xHandle.shape[2]
        precondition(mask.count == B, "mask batch mismatch: expected \(B) got \(mask.count)")
        precondition(mask.first?.count ?? 0 == L, "mask length mismatch: expected \(L)")
        let rowsExpected = B * L
        precondition(xHandle.rows == rowsExpected, "maskedMeanHandleFromHandleDeferred expects rows \(rowsExpected), got \(xHandle.rows)")
        precondition(xHandle.cols == H, "maskedMeanHandleFromHandleDeferred expects cols \(H), got \(xHandle.cols)")
        if B == 0 || L == 0 || H == 0 {
            let emptyBuffer = buffer(length: 16, label: "GPUActor.Elementwise.maskedMeanHandleFromHandle.empty")
            memset(emptyBuffer.contents(), 0, 16)
            if consumeInput {
                _ = consumeHandle(xHandle)
            }
            let handle = registerHandle(buffer: emptyBuffer,
                                        shape: outputShape ?? [B, H],
                                        rows: B,
                                        cols: H,
                                        rowBytes: 0,
                                        elemType: .float32,
                                        label: "Elementwise.maskedMeanHandleFromHandle.empty")
            return GPUReadback(resolved: handle)
        }
        let maskFlat = flattenMask(mask, expectedBatch: B, expectedLength: L)
        let sourceBuffer = consumeInput
            ? consumeHandle(xHandle, expectRows: rowsExpected, expectCols: H)
            : peekHandle(xHandle, expectRows: rowsExpected, expectCols: H)
        let xBuffer = try packHandleToFloat32Contiguous(sourceBuffer: sourceBuffer,
                                                        rows: rowsExpected,
                                                        cols: H,
                                                        rowBytes: xHandle.rowBytes,
                                                        elemType: xHandle.elemType,
                                                        label: "GPUActor.Elementwise.maskedMeanHandleFromHandle.x")
        let pipelines = try ensureElementwisePipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskedMeanHandleFromHandle: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Elementwise.maskedMeanHandleFromHandle"
        let elem = MemoryLayout<Float>.size
        let yRowBytes = H * elem
        let yBuffer = buffer(length: B * yRowBytes, label: "GPUActor.Elementwise.maskedMeanHandleFromHandle.y")
        memset(yBuffer.contents(), 0, B * yRowBytes)
        guard let maskBuffer = device.makeBuffer(length: maskFlat.count * MemoryLayout<Int32>.size,
                                                 options: .storageModeShared) else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskedMeanHandleFromHandle: mask buffer allocation failed")
        }
        maskBuffer.label = "GPUActor.Elementwise.maskedMeanHandleFromHandle.mask"
        maskFlat.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(maskBuffer.contents(), base, maskFlat.count * MemoryLayout<Int32>.size)
            }
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Elementwise.maskedMeanHandleFromHandle: encoder creation failed")
        }
        encoder.label = "GPUActor.Elementwise.maskedMeanHandleFromHandle.encoder"
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
        let handle = registerHandle(buffer: yBuffer,
                                    shape: outputShape ?? [B, H],
                                    rows: B,
                                    cols: H,
                                    rowBytes: yRowBytes,
                                    elemType: .float32,
                                    label: "Elementwise.maskedMeanHandleFromHandle.output")
        return scheduleCommandBuffer(label: "GPUActor.Elementwise.maskedMeanHandleFromHandle",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
            handle
        }
    }

    public func maskedMeanBackward(dPooled: Tensor, mask: [[Int]], seqLen: Int) async throws -> Tensor {
        let readback = try await maskedMeanBackwardDeferred(dPooled: dPooled,
                                                            mask: mask,
                                                            seqLen: seqLen,
                                                            deferUntilSync: false)
        return try await readback.value()
    }

    public func maskedMeanBackwardDeferred(dPooled: Tensor,
                                           mask: [[Int]],
                                           seqLen: Int,
                                           deferUntilSync: Bool = true) async throws -> GPUReadback<Tensor> {
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
        
        struct ResultReader: Sendable {
            let bufferPtr: UInt
            let B: Int
            let seqLen: Int
            let H: Int
            let dxBytes: Int
            
            func read() -> Tensor {
                let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
                var dxHost = [Float](repeating: 0, count: B * seqLen * H)
                memcpy(&dxHost, ptr, dxBytes)
                return Tensor(shape: [B, seqLen, H], data: dxHost)
            }
        }
        
        let reader = ResultReader(
            bufferPtr: UInt(bitPattern: dxBuffer.contents()),
            B: B,
            seqLen: seqLen,
            H: H,
            dxBytes: dxBytes
        )
        
        return scheduleCommandBuffer(label: "GPUActor.Elementwise.maskedMeanBackward",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) { [reader] in
            return reader.read()
        }
    }

    private func packHandleToFloat32Contiguous(sourceBuffer: MTLBuffer,
                                               rows: Int,
                                               cols: Int,
                                               rowBytes: Int,
                                               elemType: GPUElementType,
                                               label: String) throws -> MTLBuffer {
        let elemFloat = MemoryLayout<Float>.size
        guard let dst = device.makeBuffer(length: rows * cols * elemFloat, options: .storageModeShared) else {
            throw GPUActorError.commandBufferUnavailable("\(label): contiguous buffer allocation failed")
        }
        dst.label = label
        let dstPtr = dst.contents().bindMemory(to: Float.self, capacity: rows * cols)
        for row in 0..<rows {
            let dstRow = dstPtr.advanced(by: row * cols)
            if elemType == .float32 {
                let src = sourceBuffer.contents().advanced(by: row * rowBytes).bindMemory(to: Float.self, capacity: cols)
                memcpy(dstRow, src, cols * elemFloat)
            } else {
                let src = sourceBuffer.contents().advanced(by: row * rowBytes).bindMemory(to: Float16.self, capacity: cols)
                for c in 0..<cols {
                    dstRow[c] = Float(src[c])
                }
            }
        }
        return dst
    }

    private func ensureElementwisePipelines() throws -> ElementwisePipelines {
        if let pipelines = elementwisePipelines {
            return pipelines
        }
        let pipelines = try ElementwisePipelines(device: device)
        elementwisePipelines = pipelines
        return pipelines
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

struct ElementwisePipelines {
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
        self.library = library
        self.residualAdd = try device.makeComputePipelineState(function: residualAddFunction)
        self.maskZero = try device.makeComputePipelineState(function: maskZeroFunction)
        self.maskedMean = try device.makeComputePipelineState(function: maskedMeanFunction)
        self.maskedMeanBackward = try device.makeComputePipelineState(function: maskedMeanBackwardFunction)
        self.addBroadcast2DInto3D = try device.makeComputePipelineState(function: addBroadcastFunction)
    }
}
