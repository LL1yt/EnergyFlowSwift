import Metal

extension GPUActor {
    public func embeddingForward(ids: [[Int]], weight: Tensor) async throws -> Tensor {
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
        
        struct ResultReader: Sendable {
            let bufferPtr: UInt
            let outCount: Int
            let B: Int
            let L: Int
            let D: Int
            
            func read() -> Tensor {
                let ptr = UnsafeMutableRawPointer(bitPattern: bufferPtr)!
                var output = [Float](repeating: 0, count: outCount)
                memcpy(&output, ptr, outCount * MemoryLayout<Float>.size)
                return Tensor(shape: [B, L, D], data: output)
            }
        }
        
        let reader = ResultReader(
            bufferPtr: UInt(bitPattern: outBuffer.contents()),
            outCount: outCount,
            B: B,
            L: L,
            D: D
        )
        
        return try await awaitCommandBuffer(label: "GPUActor.Embedding.forward",
                                            commandBuffer: commandBuffer) { [reader] in
            return reader.read()
        }
    }

    private func ensureEmbeddingPipelines() throws -> EmbeddingPipelines {
        if let pipelines = embeddingPipelines {
            return pipelines
        }
        let pipelines = try EmbeddingPipelines(device: device)
        embeddingPipelines = pipelines
        return pipelines
    }
}

struct EmbeddingPipelines {
    let library: MTLLibrary
    let gather: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: EmbeddingMetalLibrary.source, options: nil)
        guard let gatherFunction = library.makeFunction(name: "embedding_gather_f32") else {
            throw GPUActorError.pipelineFunctionMissing("embedding_gather_f32")
        }
        self.library = library
        self.gather = try device.makeComputePipelineState(function: gatherFunction)
    }
}
