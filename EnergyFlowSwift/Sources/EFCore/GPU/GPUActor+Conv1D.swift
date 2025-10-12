import MetalPerformanceShaders

extension GPUActor {
    public func conv1DForward(key: UUID,
                              version: UInt64,
                              inChannels: Int,
                              outChannels: Int,
                              kernelSize: Int,
                              dilation: Int,
                              weight: Tensor,
                              bias: Tensor?,
                              x: Tensor) async throws -> Tensor {
        let readback = try await conv1DForwardDeferred(key: key,
                                                       version: version,
                                                       inChannels: inChannels,
                                                       outChannels: outChannels,
                                                       kernelSize: kernelSize,
                                                       dilation: dilation,
                                                       weight: weight,
                                                       bias: bias,
                                                       x: x,
                                                       deferUntilSync: false)
        return try await readback.value()
    }

    public func conv1DForwardDeferred(key: UUID,
                                      version: UInt64,
                                      inChannels: Int,
                                      outChannels: Int,
                                      kernelSize: Int,
                                      dilation: Int,
                                      weight: Tensor,
                                      bias: Tensor?,
                                      x: Tensor,
                                      deferUntilSync: Bool = true) async throws -> GPUReadback<Tensor> {
        precondition(x.shape.count == 3, "conv1DForward expects [B,L,Cin]")
        let B = x.shape[0]
        let L = x.shape[1]
        let Cin = x.shape[2]
        precondition(Cin == inChannels, "Cin mismatch: got \(Cin) expected \(inChannels)")
        if B == 0 || L == 0 {
            return GPUReadback(resolved: Tensor.zeros([B, L, outChannels]))
        }

        let weightCache = try ensureConv1DCache(
            key: key,
            version: version,
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            weight: weight,
            bias: bias
        )

        let colsX = weightCache.inputColumns
        let rows = B * L
        let elemHalf = MemoryLayout<Float16>.size
        let xRowBytes = alignedRowBytes(columns: colsX, elemSize: elemHalf)
        let yRowBytes = alignedRowBytes(columns: outChannels, elemSize: elemHalf)
        let xcolBuffer = buffer(length: rows * xRowBytes, label: "GPUActor.Conv1D.Xcol")
        memset(xcolBuffer.contents(), 0, rows * xRowBytes)

        try await im2colFP16ToBuffer(
            X: x,
            B: B,
            L: L,
            Cin: inChannels,
            K: kernelSize,
            dilation: dilation,
            outBuffer: xcolBuffer,
            outRowBytes: xRowBytes,
            outColsTotal: colsX
        )
        if weightCache.hasBias {
            try await fillBiasColumnFP16(
                outBuffer: xcolBuffer,
                rows: rows,
                outRowBytes: xRowBytes,
                biasIndex: inChannels * kernelSize
            )
        }

        let yBuffer = buffer(length: rows * yRowBytes, label: "GPUActor.Conv1D.Y")
        memset(yBuffer.contents(), 0, rows * yRowBytes)

        let xDesc = MPSMatrixDescriptor(rows: rows, columns: colsX, rowBytes: xRowBytes, dataType: .float16)
        let wDesc = MPSMatrixDescriptor(rows: outChannels, columns: colsX, rowBytes: weightCache.rowBytes, dataType: .float16)
        let yDesc = MPSMatrixDescriptor(rows: rows, columns: outChannels, rowBytes: yRowBytes, dataType: .float16)

        let xMat = MPSMatrix(buffer: xcolBuffer, descriptor: xDesc)
        let wMat = MPSMatrix(buffer: weightCache.weightBuffer, descriptor: wDesc)
        let yMat = MPSMatrix(buffer: yBuffer, descriptor: yDesc)

        let mm = MPSMatrixMultiplication(device: device,
                                         transposeLeft: false,
                                         transposeRight: true,
                                         resultRows: rows,
                                         resultColumns: outChannels,
                                         interiorColumns: colsX,
                                         alpha: 1.0,
                                         beta: 0.0)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("conv1DForward: failed to create command buffer")
        }
        commandBuffer.label = "GPUActor.Conv1D.forward"
        mm.encode(commandBuffer: commandBuffer, leftMatrix: xMat, rightMatrix: wMat, resultMatrix: yMat)
        
        let reader = StridedFloat16BufferReader(
            buffer: yBuffer,
            rows: rows,
            cols: outChannels,
            rowBytes: yRowBytes,
            shape: [B, L, outChannels]
        )
        
        return scheduleCommandBufferWithReader(
            label: "GPUActor.Conv1D.forward",
            commandBuffer: commandBuffer,
            deferUntilSync: deferUntilSync,
            reader: reader
        )
    }

    public func conv1DForwardFromHandleDeferred(key: UUID,
                                                version: UInt64,
                                                inChannels: Int,
                                                outChannels: Int,
                                                kernelSize: Int,
                                                dilation: Int,
                                                weight: Tensor,
                                                bias: Tensor?,
                                                xHandle: GPUTensorHandle,
                                                outputShape: [Int]? = nil,
                                                consumeInput: Bool = false,
                                                deferUntilSync: Bool = true) async throws -> GPUReadback<GPUTensorHandle> {
        precondition(xHandle.shape.count == 3, "conv1DForwardFromHandle expects handle shape [B,L,Cin]")
        let B = xHandle.shape[0]
        let L = xHandle.shape[1]
        let Cin = xHandle.shape[2]
        precondition(Cin == inChannels, "conv1DForwardFromHandle: Cin mismatch (handle \(Cin) vs expected \(inChannels))")
        if B == 0 || L == 0 {
            let emptyBuffer = buffer(length: 16, label: "GPUActor.Conv1D.forwardHandle.empty")
            let emptyHandle = registerHandle(buffer: emptyBuffer,
                                             shape: outputShape ?? [0, 0, outChannels],
                                             rows: 0,
                                             cols: outChannels,
                                             rowBytes: 0,
                                             elemType: .float16,
                                             label: "Conv1D.forwardHandle.empty")
            return GPUReadback(resolved: emptyHandle)
        }
        let rows = B * L
        precondition(xHandle.rows == rows, "conv1DForwardFromHandle expects handle rows \(rows), got \(xHandle.rows)")
        precondition(xHandle.cols == Cin, "conv1DForwardFromHandle expects handle cols \(Cin), got \(xHandle.cols)")

        let weightCache = try ensureConv1DCache(
            key: key,
            version: version,
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            weight: weight,
            bias: bias
        )

        // Pack handle data into a contiguous Float buffer for im2col.
        let elemFloat = MemoryLayout<Float>.size
        guard let xFloatBuffer = device.makeBuffer(length: rows * Cin * elemFloat, options: .storageModeShared) else {
            throw GPUActorError.commandBufferUnavailable("conv1DForwardFromHandle: failed to allocate xFloat buffer")
        }
        xFloatBuffer.label = "GPUActor.Conv1D.forwardHandle.xFloat"
        let xSourceBuffer = consumeInput
            ? consumeHandle(xHandle, expectRows: rows, expectCols: Cin)
            : peekHandle(xHandle, expectRows: rows, expectCols: Cin)
        let floatPtr = xFloatBuffer.contents().bindMemory(to: Float.self, capacity: rows * Cin)
        for row in 0..<rows {
            let dst = floatPtr.advanced(by: row * Cin)
            if xHandle.elemType == .float32 {
                let src = xSourceBuffer.contents().advanced(by: row * xHandle.rowBytes).bindMemory(to: Float.self, capacity: Cin)
                memcpy(dst, src, Cin * elemFloat)
            } else {
                let src = xSourceBuffer.contents().advanced(by: row * xHandle.rowBytes).bindMemory(to: Float16.self, capacity: Cin)
                for c in 0..<Cin {
                    dst[c] = Float(src[c])
                }
            }
        }

        let colsX = weightCache.inputColumns
        let elemHalf = MemoryLayout<Float16>.size
        let xRowBytes = alignedRowBytes(columns: colsX, elemSize: elemHalf)
        let yRowBytes = alignedRowBytes(columns: outChannels, elemSize: elemHalf)
        let xcolBuffer = buffer(length: rows * xRowBytes, label: "GPUActor.Conv1D.forwardHandle.Xcol")
        memset(xcolBuffer.contents(), 0, rows * xRowBytes)

        // Run im2col on the GPU to pack slices into FP16 columns.
        let pipelines = try ensureIm2ColPipelines()
        guard let im2colCommandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("conv1DForwardFromHandle: im2col command buffer creation failed")
        }
        im2colCommandBuffer.label = "GPUActor.Conv1D.forwardHandle.im2col"
        guard let im2colEncoder = im2colCommandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("conv1DForwardFromHandle: im2col encoder creation failed")
        }
        im2colEncoder.label = "GPUActor.Conv1D.forwardHandle.im2col.encoder"
        im2colEncoder.setComputePipelineState(pipelines.im2colF16)
        im2colEncoder.setBuffer(xFloatBuffer, offset: 0, index: 0)
        im2colEncoder.setBuffer(xcolBuffer, offset: 0, index: 1)
        var vB = Int32(B)
        var vL = Int32(L)
        var vCin = Int32(Cin)
        var vK = Int32(kernelSize)
        var vDil = Int32(max(1, dilation))
        var vRowStride = Int32(xRowBytes / elemHalf)
        var vColsTotal = Int32(colsX)
        im2colEncoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        im2colEncoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        im2colEncoder.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 4)
        im2colEncoder.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 5)
        im2colEncoder.setBytes(&vDil, length: MemoryLayout<Int32>.size, index: 6)
        im2colEncoder.setBytes(&vRowStride, length: MemoryLayout<Int32>.size, index: 7)
        im2colEncoder.setBytes(&vColsTotal, length: MemoryLayout<Int32>.size, index: 8)
        let total = rows * Cin * kernelSize
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (total + 255) / 256, height: 1, depth: 1)
        im2colEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        im2colEncoder.endEncoding()
        try await awaitCommandBuffer(label: "GPUActor.Conv1D.forwardHandle.im2col",
                                     commandBuffer: im2colCommandBuffer) { () }

        if weightCache.hasBias {
            try await fillBiasColumnFP16(
                outBuffer: xcolBuffer,
                rows: rows,
                outRowBytes: xRowBytes,
                biasIndex: inChannels * kernelSize
            )
        }

        let yBuffer = buffer(length: rows * yRowBytes, label: "GPUActor.Conv1D.forwardHandle.Y")
        memset(yBuffer.contents(), 0, rows * yRowBytes)

        let xDesc = MPSMatrixDescriptor(rows: rows, columns: colsX, rowBytes: xRowBytes, dataType: .float16)
        let wDesc = MPSMatrixDescriptor(rows: outChannels, columns: colsX, rowBytes: weightCache.rowBytes, dataType: .float16)
        let yDesc = MPSMatrixDescriptor(rows: rows, columns: outChannels, rowBytes: yRowBytes, dataType: .float16)

        let xMat = MPSMatrix(buffer: xcolBuffer, descriptor: xDesc)
        let wMat = MPSMatrix(buffer: weightCache.weightBuffer, descriptor: wDesc)
        let yMat = MPSMatrix(buffer: yBuffer, descriptor: yDesc)

        let mm = MPSMatrixMultiplication(device: device,
                                         transposeLeft: false,
                                         transposeRight: true,
                                         resultRows: rows,
                                         resultColumns: outChannels,
                                         interiorColumns: colsX,
                                         alpha: 1.0,
                                         beta: 0.0)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("conv1DForwardFromHandle: failed to create command buffer")
        }
        commandBuffer.label = "GPUActor.Conv1D.forwardHandle"
        mm.encode(commandBuffer: commandBuffer, leftMatrix: xMat, rightMatrix: wMat, resultMatrix: yMat)

        let handle = registerHandle(buffer: yBuffer,
                                    shape: outputShape ?? [B, L, outChannels],
                                    rows: rows,
                                    cols: outChannels,
                                    rowBytes: yRowBytes,
                                    elemType: .float16,
                                    label: "Conv1D.forwardHandle.output")

        return scheduleCommandBuffer(label: "GPUActor.Conv1D.forwardHandle",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
            handle
        }
    }

    private func ensureConv1DCache(key: UUID,
                                   version: UInt64,
                                   inChannels: Int,
                                   outChannels: Int,
                                   kernelSize: Int,
                                   weight: Tensor,
                                   bias: Tensor?) throws -> Conv1DCacheEntry {
        if let entry = conv1DCaches[key],
           entry.version == version,
           entry.inChannels == inChannels,
           entry.outChannels == outChannels,
           entry.kernelSize == kernelSize,
           entry.hasBias == (bias != nil) {
            return entry
        }
        let CinK = inChannels * kernelSize
        let hasBias = (bias != nil)
        let cols = hasBias ? (CinK + 1) : CinK
        let elemHalf = MemoryLayout<Float16>.size
        let rowBytes = alignedRowBytes(columns: cols, elemSize: elemHalf)
        guard let weightBuffer = device.makeBuffer(length: outChannels * rowBytes, options: .storageModeShared) else {
            throw GPUActorError.commandBufferUnavailable("conv1D cache: unable to allocate weight buffer")
        }
        weightBuffer.label = "GPUActor.Conv1D.weight.\(key)"
        memset(weightBuffer.contents(), 0, weightBuffer.length)

        var weightHalf = [Float16](repeating: 0, count: outChannels * cols)
        for o in 0..<outChannels {
            for i in 0..<inChannels {
                for k in 0..<kernelSize {
                    let src = (o * inChannels + i) * kernelSize + k
                    let dst = o * cols + (i * kernelSize + k)
                    weightHalf[dst] = Float16(weight.data[src])
                }
            }
            if hasBias, let biasTensor = bias {
                weightHalf[o * cols + CinK] = Float16(biasTensor.data[o])
            }
        }

        weightHalf.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                let rowSize = cols * elemHalf
                for row in 0..<outChannels {
                    memcpy(weightBuffer.contents().advanced(by: row * rowBytes), base.advanced(by: row * rowSize), rowSize)
                }
            }
        }

        let entry = Conv1DCacheEntry(
            version: version,
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            hasBias: hasBias,
            weightBuffer: weightBuffer,
            rowBytes: rowBytes,
            inputColumns: cols
        )
        conv1DCaches[key] = entry
        return entry
    }
}

struct Conv1DCacheEntry {
    let version: UInt64
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let hasBias: Bool
    let weightBuffer: MTLBuffer
    let rowBytes: Int
    let inputColumns: Int
}
