import Metal

extension GPUActor {
    public func im2col(X: Tensor, B: Int, L: Int, Cin: Int, K: Int, dilation: Int) async throws -> Tensor {
        precondition(X.shape == [B, L, Cin], "im2col expects X [B,L,Cin]")
        let pipelines = try ensureIm2ColPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("im2col: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.im2col"
        let rows = B * L
        let colsX = Cin * K
        let outCount = rows * colsX
        let inCount = B * L * Cin
        let elem = MemoryLayout<Float>.size
        let inBuffer = buffer(length: inCount * elem, label: "GPUActor.im2col.in")
        let outBuffer = buffer(length: outCount * elem, label: "GPUActor.im2col.out")
        X.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(inBuffer.contents(), base, inCount * elem)
            }
        }
        memset(outBuffer.contents(), 0, outCount * elem)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("im2col: encoder creation failed")
        }
        encoder.label = "GPUActor.im2col.encoder"
        encoder.setComputePipelineState(pipelines.im2colF32)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vCin = Int32(Cin), vK = Int32(K), vDil = Int32(max(1, dilation))
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 5)
        encoder.setBytes(&vDil, length: MemoryLayout<Int32>.size, index: 6)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (outCount + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        let reader = FloatBufferReader(
            buffer: outBuffer,
            count: outCount,
            shape: [rows, colsX]
        )
        
        return try await awaitCommandBufferWithReader(
            label: "GPUActor.im2col",
            commandBuffer: commandBuffer,
            reader: reader
        )
    }

    public func col2im(dXcol: Tensor, B: Int, L: Int, Cin: Int, K: Int, dilation: Int) async throws -> Tensor {
        precondition(dXcol.shape == [B * L, Cin * K], "col2im expects dXcol [B*L, Cin*K]")
        let pipelines = try ensureIm2ColPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("col2im: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.col2im"
        let rows = B * L
        let colsX = Cin * K
        let inCount = rows * colsX
        let outCount = B * L * Cin
        let elem = MemoryLayout<Float>.size
        let inBuffer = buffer(length: inCount * elem, label: "GPUActor.col2im.in")
        let outBuffer = buffer(length: outCount * elem, label: "GPUActor.col2im.out")
        dXcol.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(inBuffer.contents(), base, inCount * elem)
            }
        }
        memset(outBuffer.contents(), 0, outCount * elem)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("col2im: encoder creation failed")
        }
        encoder.label = "GPUActor.col2im.encoder"
        encoder.setComputePipelineState(pipelines.col2imF32)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vCin = Int32(Cin), vK = Int32(K), vDil = Int32(max(1, dilation))
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 5)
        encoder.setBytes(&vDil, length: MemoryLayout<Int32>.size, index: 6)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (outCount + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        let reader = FloatBufferReader(
            buffer: outBuffer,
            count: outCount,
            shape: [B, L, Cin]
        )
        
        return try await awaitCommandBufferWithReader(
            label: "GPUActor.col2im",
            commandBuffer: commandBuffer,
            reader: reader
        )
    }

    public func im2colFP16ToBuffer(X: Tensor,
                                   B: Int,
                                   L: Int,
                                   Cin: Int,
                                   K: Int,
                                   dilation: Int,
                                   outBuffer: MTLBuffer,
                                   outRowBytes: Int,
                                   outColsTotal: Int) async throws {
        precondition(X.shape == [B, L, Cin], "im2colFP16ToBuffer expects X [B,L,Cin]")
        let pipelines = try ensureIm2ColPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("im2colFP16ToBuffer: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.im2colFP16"
        let rows = B * L
        let colsX = Cin * K
        let inCount = B * L * Cin
        let elemF = MemoryLayout<Float>.size
        let inBuffer = buffer(length: inCount * elemF, label: "GPUActor.im2colFP16.in")
        X.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(inBuffer.contents(), base, inCount * elemF)
            }
        }
        memset(outBuffer.contents(), 0, rows * outRowBytes)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("im2colFP16ToBuffer: encoder creation failed")
        }
        encoder.label = "GPUActor.im2colFP16.encoder"
        encoder.setComputePipelineState(pipelines.im2colF16)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var vB = Int32(B), vL = Int32(L), vCin = Int32(Cin), vK = Int32(K), vDil = Int32(max(1, dilation))
        var vRowStride = Int32(outRowBytes / MemoryLayout<Float16>.size)
        var vColsTotal = Int32(outColsTotal)
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes(&vL, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 5)
        encoder.setBytes(&vDil, length: MemoryLayout<Int32>.size, index: 6)
        encoder.setBytes(&vRowStride, length: MemoryLayout<Int32>.size, index: 7)
        encoder.setBytes(&vColsTotal, length: MemoryLayout<Int32>.size, index: 8)
        let total = rows * colsX
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (total + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        try await awaitCommandBuffer(label: "GPUActor.im2colFP16ToBuffer",
                                     commandBuffer: commandBuffer) { () }
    }

    public func fillBiasColumnFP16(outBuffer: MTLBuffer, rows: Int, outRowBytes: Int, biasIndex: Int) async throws {
        let pipelines = try ensureIm2ColPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("fillBiasColumnFP16: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.fillBiasColumnFP16"
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("fillBiasColumnFP16: encoder creation failed")
        }
        encoder.label = "GPUActor.fillBiasColumnFP16.encoder"
        encoder.setComputePipelineState(pipelines.fillBiasF16)
        encoder.setBuffer(outBuffer, offset: 0, index: 0)
        var vRows = Int32(rows)
        var vRowStride = Int32(outRowBytes / MemoryLayout<Float16>.size)
        var vBias = Int32(biasIndex)
        encoder.setBytes(&vRows, length: MemoryLayout<Int32>.size, index: 1)
        encoder.setBytes(&vRowStride, length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes(&vBias, length: MemoryLayout<Int32>.size, index: 3)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (rows + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        try await awaitCommandBuffer(label: "GPUActor.fillBiasColumnFP16",
                                     commandBuffer: commandBuffer) { () }
    }

    func ensureIm2ColPipelines() throws -> Im2ColPipelines {
        if let pipelines = im2ColPipelines {
            return pipelines
        }
        let pipelines = try Im2ColPipelines(device: device)
        im2ColPipelines = pipelines
        return pipelines
    }
}

struct Im2ColPipelines {
    let library: MTLLibrary
    let im2colF32: MTLComputePipelineState
    let col2imF32: MTLComputePipelineState
    let im2colF16: MTLComputePipelineState
    let fillBiasF16: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: Im2ColCol2ImMetalLibrary.source, options: nil)
        guard let im2colF32 = library.makeFunction(name: "im2col_1d_causal_f32") else {
            throw GPUActorError.pipelineFunctionMissing("im2col_1d_causal_f32")
        }
        guard let col2imF32 = library.makeFunction(name: "col2im_1d_causal_f32") else {
            throw GPUActorError.pipelineFunctionMissing("col2im_1d_causal_f32")
        }
        guard let im2colF16 = library.makeFunction(name: "im2col_1d_causal_f32_to_f16") else {
            throw GPUActorError.pipelineFunctionMissing("im2col_1d_causal_f32_to_f16")
        }
        guard let fillBiasF16 = library.makeFunction(name: "fill_bias_col_fp16") else {
            throw GPUActorError.pipelineFunctionMissing("fill_bias_col_fp16")
        }
        self.library = library
        self.im2colF32 = try device.makeComputePipelineState(function: im2colF32)
        self.col2imF32 = try device.makeComputePipelineState(function: col2imF32)
        self.im2colF16 = try device.makeComputePipelineState(function: im2colF16)
        self.fillBiasF16 = try device.makeComputePipelineState(function: fillBiasF16)
    }
}
