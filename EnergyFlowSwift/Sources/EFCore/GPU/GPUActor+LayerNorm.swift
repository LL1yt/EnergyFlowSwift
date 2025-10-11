import Metal

extension GPUActor {
    public func layerNormForward(x: Tensor, gamma: Tensor, beta: Tensor, eps: Float) async throws -> Tensor {
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
        let reader = Float16BufferReader(buffer: yBuffer, count: count, shape: x.shape)
        return try await awaitCommandBufferWithReader(
            label: "GPUActor.LayerNorm.forward",
            commandBuffer: commandBuffer,
            reader: reader
        )
    }

    public func layerNormForwardDeferred(x: Tensor, gamma: Tensor, beta: Tensor, eps: Float, deferUntilSync: Bool = true) async throws -> GPUReadback<Tensor> {
        precondition(x.shape.count == 2, "GPUActor.layerNormForwardDeferred expects [N,D]")
        precondition(gamma.shape == [x.shape[1]] && beta.shape == [x.shape[1]], "LayerNorm gamma/beta mismatch")
        let pipelines = try ensureLayerNormPipelines()
        let N = x.shape[0]
        let D = x.shape[1]
        let count = N * D
        if count == 0 { return GPUReadback(resolved: x) }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("LayerNorm.forwardDeferred: command buffer creation failed")
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
            throw GPUActorError.commandBufferUnavailable("LayerNorm.forwardDeferred: stats encoder creation failed")
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
            throw GPUActorError.commandBufferUnavailable("LayerNorm.forwardDeferred: normalize encoder creation failed")
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
        let reader = Float16BufferReader(buffer: yBuffer, count: count, shape: x.shape)
        return scheduleCommandBufferWithReader(
            label: "GPUActor.LayerNorm.forward",
            commandBuffer: commandBuffer,
            deferUntilSync: deferUntilSync,
            reader: reader
        )
    }

    public func layerNormBackward(x: Tensor, g: Tensor, gamma: Tensor, eps: Float) async throws -> (Tensor, Tensor, Tensor) {
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
        
        let dxReader = Float16BufferReader(
            buffer: dxBuffer,
            count: count,
            shape: [N, D]
        )
        let dGammaReader = FloatBufferReader(
            buffer: dGammaBuffer,
            count: D,
            shape: [D]
        )
        let dBetaReader = FloatBufferReader(
            buffer: dBetaBuffer,
            count: D,
            shape: [D]
        )
        let reader = BufferReader<(Tensor, Tensor, Tensor)>(
            buffer: dxBuffer
        ) {
            let dx = dxReader.read()
            let dGamma = dGammaReader.read()
            let dBeta = dBetaReader.read()
            return (dx, dGamma, dBeta)
        }
        
        return try await awaitCommandBufferWithReader(
            label: "GPUActor.LayerNorm.backward",
            commandBuffer: commandBuffer,
            reader: reader
        )
    }

    private func ensureLayerNormPipelines() throws -> LayerNormPipelines {
        if let pipelines = layerNormPipelines {
            return pipelines
        }
        let pipelines = try LayerNormPipelines(device: device)
        layerNormPipelines = pipelines
        return pipelines
    }
}

struct LayerNormPipelines {
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
        self.library = library
        self.stats = try device.makeComputePipelineState(function: statsFunction)
        self.normalize = try device.makeComputePipelineState(function: normFunction)
        self.backwardRow = try device.makeComputePipelineState(function: rowFunction)
        self.backwardDX = try device.makeComputePipelineState(function: dxFunction)
        self.backwardDGamma = try device.makeComputePipelineState(function: dgbFunction)
    }
}
