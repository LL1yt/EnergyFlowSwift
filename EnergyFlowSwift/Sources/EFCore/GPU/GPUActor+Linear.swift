import MetalPerformanceShaders

extension GPUActor {
    public func linearForward(key: UUID,
                              version: UInt64,
                              inFeatures: Int,
                              outFeatures: Int,
                              weight: Tensor,
                              bias: Tensor?,
                              x: Tensor) async throws -> Tensor {
        let readback = try await linearForwardDeferred(key: key,
                                                       version: version,
                                                       inFeatures: inFeatures,
                                                       outFeatures: outFeatures,
                                                       weight: weight,
                                                       bias: bias,
                                                       x: x,
                                                       deferUntilSync: false)
        return try await readback.value()
    }

    public func linearForwardDeferred(key: UUID,
                                      version: UInt64,
                                      inFeatures: Int,
                                      outFeatures: Int,
                                      weight: Tensor,
                                      bias: Tensor?,
                                      x: Tensor,
                                      deferUntilSync: Bool = true) async throws -> GPUReadback<Tensor> {
        precondition(x.shape.count == 2 && x.shape[1] == inFeatures, "linearForward expects [B, inFeatures]")
        let batch = x.shape[0]
        if batch == 0 { return GPUReadback(resolved: Tensor.zeros([0, outFeatures])) }
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
        return scheduleCommandBuffer(label: "GPUActor.Linear.forward",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
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
    }

    public func linearGradients(key: UUID,
                                version: UInt64,
                                inFeatures: Int,
                                outFeatures: Int,
                                weight: Tensor,
                                X: Tensor,
                                dY: Tensor,
                                bias: Tensor?) async throws -> (Tensor, Tensor) {
        let readback = try await linearGradientsDeferred(key: key,
                                                         version: version,
                                                         inFeatures: inFeatures,
                                                         outFeatures: outFeatures,
                                                         weight: weight,
                                                         X: X,
                                                         dY: dY,
                                                         bias: bias,
                                                         deferUntilSync: false)
        return try await readback.value()
    }

    public func linearGradientsDeferred(key: UUID,
                                        version: UInt64,
                                        inFeatures: Int,
                                        outFeatures: Int,
                                        weight: Tensor,
                                        X: Tensor,
                                        dY: Tensor,
                                        bias: Tensor?,
                                        deferUntilSync: Bool = true) async throws -> GPUReadback<(Tensor, Tensor)> {
        precondition(X.shape.count == 2 && dY.shape.count == 2, "linearGradients expects 2D tensors")
        let batch = X.shape[0]
        precondition(X.shape[1] == inFeatures && dY.shape[0] == batch && dY.shape[1] == outFeatures,
                     "linearGradients shape mismatch")
        if batch == 0 {
            return GPUReadback(resolved: (Tensor.zeros([outFeatures, inFeatures]), Tensor.zeros([outFeatures])))
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
        return scheduleCommandBuffer(label: "GPUActor.Linear.gradients",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
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
    }

    public func linearInputGradients(key: UUID,
                                     version: UInt64,
                                     inFeatures: Int,
                                     outFeatures: Int,
                                     weight: Tensor,
                                     bias: Tensor?,
                                     dY: Tensor) async throws -> Tensor {
        let readback = try await linearInputGradientsDeferred(key: key,
                                                              version: version,
                                                              inFeatures: inFeatures,
                                                              outFeatures: outFeatures,
                                                              weight: weight,
                                                              bias: bias,
                                                              dY: dY,
                                                              deferUntilSync: false)
        return try await readback.value()
    }

    public func linearInputGradientsDeferred(key: UUID,
                                             version: UInt64,
                                             inFeatures: Int,
                                             outFeatures: Int,
                                             weight: Tensor,
                                             bias: Tensor?,
                                             dY: Tensor,
                                             deferUntilSync: Bool = false) async throws -> GPUReadback<Tensor> {
        precondition(dY.shape.count == 2 && dY.shape[1] == outFeatures, "linearInputGradients expects [B, outFeatures]")
        let batch = dY.shape[0]
        if batch == 0 { return GPUReadback(resolved: Tensor.zeros([0, inFeatures])) }
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
        return scheduleCommandBuffer(label: "GPUActor.Linear.inputGradients",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: deferUntilSync) {
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
}

struct LinearCacheEntry {
    let version: UInt64
    let weightBuffer: MTLBuffer
    let rowBytes: Int
    let inFeatures: Int
    let outFeatures: Int
}
