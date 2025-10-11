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
        return scheduleCommandBuffer(label: "GPUActor.Conv1D.forward",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
            var yHalf = [Float16](repeating: 0, count: rows * outChannels)
            let rowSize = outChannels * elemHalf
            yHalf.withUnsafeMutableBytes { raw in
                if let base = raw.baseAddress {
                    for r in 0..<rows {
                        memcpy(base.advanced(by: r * rowSize), yBuffer.contents().advanced(by: r * yRowBytes), rowSize)
                    }
                }
            }
            var yHost = [Float](repeating: 0, count: rows * outChannels)
            for i in 0..<yHost.count { yHost[i] = Float(yHalf[i]) }
            return Tensor(shape: [B, L, outChannels], data: yHost)
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
