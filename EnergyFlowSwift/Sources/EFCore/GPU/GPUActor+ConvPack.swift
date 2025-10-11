import Metal

extension GPUActor {
    public func packWToCol(W: Tensor, Cout: Int, Cin: Int, K: Int) async throws -> Tensor {
        precondition(W.shape == [Cout, Cin, K], "packWToCol expects weight [Cout,Cin,K]")
        let pipelines = try ensureConvPackPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("packWToCol: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.ConvPack.pack"
        let outCols = Cin * K
        let outCount = Cout * outCols
        let elem = MemoryLayout<Float>.size
        let inBuffer = buffer(length: Cout * Cin * K * elem, label: "GPUActor.ConvPack.in")
        let outBuffer = buffer(length: outCount * elem, label: "GPUActor.ConvPack.out")
        W.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(inBuffer.contents(), base, Cout * Cin * K * elem)
            }
        }
        memset(outBuffer.contents(), 0, outCount * elem)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("packWToCol: encoder creation failed")
        }
        encoder.label = "GPUActor.ConvPack.pack.encoder"
        encoder.setComputePipelineState(pipelines.pack)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var vCout = Int32(Cout), vCin = Int32(Cin), vK = Int32(K)
        encoder.setBytes(&vCout, length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 4)
        let total = Cout * Cin * K
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (total + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        struct ResultReader: Sendable {
            let bufferPtr: UInt
            let outCount: Int
            let elem: Int
            let Cout: Int
            let outCols: Int
            
            func read() -> Tensor {
                let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
                var outHost = [Float](repeating: 0, count: outCount)
                memcpy(&outHost, ptr, outCount * elem)
                return Tensor(shape: [Cout, outCols], data: outHost)
            }
        }
        
        let reader = ResultReader(
            bufferPtr: UInt(bitPattern: outBuffer.contents()),
            outCount: outCount,
            elem: elem,
            Cout: Cout,
            outCols: outCols
        )
        
        return try await awaitCommandBuffer(label: "GPUActor.ConvPack.pack",
                                            commandBuffer: commandBuffer) { [reader] in
            return reader.read()
        }
    }

    public func unpackDWCol(dWcol: Tensor, Cout: Int, Cin: Int, K: Int) async throws -> Tensor {
        precondition(dWcol.shape == [Cout, Cin * K], "unpackDWCol expects [Cout, Cin*K]")
        let pipelines = try ensureConvPackPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("unpackDWCol: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.ConvPack.unpack"
        let total = Cout * Cin * K
        let elem = MemoryLayout<Float>.size
        let inBuffer = buffer(length: Cout * Cin * K * elem, label: "GPUActor.ConvPack.unpack.in")
        let outBuffer = buffer(length: total * elem, label: "GPUActor.ConvPack.unpack.out")
        dWcol.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(inBuffer.contents(), base, Cout * Cin * K * elem)
            }
        }
        memset(outBuffer.contents(), 0, total * elem)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("unpackDWCol: encoder creation failed")
        }
        encoder.label = "GPUActor.ConvPack.unpack.encoder"
        encoder.setComputePipelineState(pipelines.unpack)
        encoder.setBuffer(inBuffer, offset: 0, index: 0)
        encoder.setBuffer(outBuffer, offset: 0, index: 1)
        var vCout = Int32(Cout), vCin = Int32(Cin), vK = Int32(K)
        encoder.setBytes(&vCout, length: MemoryLayout<Int32>.size, index: 2)
        encoder.setBytes(&vCin, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vK, length: MemoryLayout<Int32>.size, index: 4)
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadGroups = MTLSize(width: (total + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        struct ResultReader: Sendable {
            let bufferPtr: UInt
            let total: Int
            let elem: Int
            let Cout: Int
            let Cin: Int
            let K: Int
            
            func read() -> Tensor {
                let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
                var outHost = [Float](repeating: 0, count: total)
                memcpy(&outHost, ptr, total * elem)
                return Tensor(shape: [Cout, Cin, K], data: outHost)
            }
        }
        
        let reader = ResultReader(
            bufferPtr: UInt(bitPattern: outBuffer.contents()),
            total: total,
            elem: elem,
            Cout: Cout,
            Cin: Cin,
            K: K
        )
        
        return try await awaitCommandBuffer(label: "GPUActor.ConvPack.unpack",
                                            commandBuffer: commandBuffer) { [reader] in
            return reader.read()
        }
    }

    private func ensureConvPackPipelines() throws -> ConvPackPipelines {
        if let pipelines = convPackPipelines {
            return pipelines
        }
        let pipelines = try ConvPackPipelines(device: device)
        convPackPipelines = pipelines
        return pipelines
    }
}

struct ConvPackPipelines {
    let library: MTLLibrary
    let pack: MTLComputePipelineState
    let unpack: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: ConvPackMetalLibrary.source, options: nil)
        guard let packFn = library.makeFunction(name: "pack_w_to_col_f32") else {
            throw GPUActorError.pipelineFunctionMissing("pack_w_to_col_f32")
        }
        guard let unpackFn = library.makeFunction(name: "unpack_dwcol_f32") else {
            throw GPUActorError.pipelineFunctionMissing("unpack_dwcol_f32")
        }
        self.library = library
        self.pack = try device.makeComputePipelineState(function: packFn)
        self.unpack = try device.makeComputePipelineState(function: unpackFn)
    }
}
