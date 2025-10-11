import Metal

extension GPUActor {
    public func kdMetricsMean(student: Tensor,
                              teacher: Tensor) async throws -> (mse: Float, cos: Float) {
        precondition(student.shape == teacher.shape, "kdMetricsMean shape mismatch")
        let shape = student.shape
        precondition(shape.count == 2, "kdMetricsMean expects rank-2 tensors")
        let batches = shape[0]
        let dim = shape[1]
        if batches == 0 || dim == 0 {
            return (0.0, 0.0)
        }
        let pipelines = try ensureMetricsPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Metrics.kdMetricsMean: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Metrics.kdMetricsMean"
        let elemSize = MemoryLayout<Float>.size
        let studentBytes = student.count * elemSize
        let teacherBytes = teacher.count * elemSize
        let accumBytes = 2 * elemSize
        let studentBuffer = buffer(length: studentBytes, label: "GPUActor.Metrics.kd.student")
        let teacherBuffer = buffer(length: teacherBytes, label: "GPUActor.Metrics.kd.teacher")
        let accumBuffer = buffer(length: accumBytes, label: "GPUActor.Metrics.kd.accum")
        student.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(studentBuffer.contents(), base, studentBytes)
            }
        }
        teacher.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(teacherBuffer.contents(), base, teacherBytes)
            }
        }
        memset(accumBuffer.contents(), 0, accumBytes)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Metrics.kdMetricsMean: encoder creation failed")
        }
        encoder.label = "GPUActor.Metrics.kdMetricsMean.encoder"
        encoder.setComputePipelineState(pipelines.kdMetricsReduce)
        encoder.setBuffer(studentBuffer, offset: 0, index: 0)
        encoder.setBuffer(teacherBuffer, offset: 0, index: 1)
        encoder.setBuffer(accumBuffer, offset: 0, index: 2)
        var vB = Int32(batches)
        var vD = Int32(dim)
        var eps: Float = 1e-12
        encoder.setBytes(&vB, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vD, length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes(&eps, length: MemoryLayout<Float>.size, index: 5)
        let threadsPerGroup = MTLSize(width: min(batches, 256), height: 1, depth: 1)
        let groups = MTLSize(width: (batches + threadsPerGroup.width - 1) / threadsPerGroup.width,
                             height: 1,
                             depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        return try await awaitCommandBuffer(label: "GPUActor.Metrics.kdMetricsMean",
                                            commandBuffer: commandBuffer) {
            let ptr = accumBuffer.contents().bindMemory(to: Float.self, capacity: 2)
            let sumMSE = ptr[0]
            let sumCos = ptr[1]
            let invB = 1.0 / Float(batches)
            return (sumMSE * invB, sumCos * invB)
        }
    }

    private func ensureMetricsPipelines() throws -> MetricsPipelines {
        if let cached = metricsPipelines {
            return cached
        }
        let pipelines = try MetricsPipelines(device: device)
        metricsPipelines = pipelines
        return pipelines
    }
}
