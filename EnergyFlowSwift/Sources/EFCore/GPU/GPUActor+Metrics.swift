import Metal

extension GPUActor {
    public func kdMetricsMean(student: Tensor,
                              teacher: Tensor) async throws -> (mse: Float, cos: Float) {
        let readback = try await kdMetricsMeanDeferred(student: student, teacher: teacher)
        return try await readback.value()
    }

    public func kdMetricsMeanDeferred(student: Tensor,
                                      teacher: Tensor) async throws -> GPUReadback<(Float, Float)> {
        precondition(student.shape == teacher.shape, "kdMetricsMean shape mismatch")
        let shape = student.shape
        precondition(shape.count == 2, "kdMetricsMean expects rank-2 tensors")
        let batches = shape[0]
        let dim = shape[1]
        if batches == 0 || dim == 0 {
            return GPUReadback(resolved: (Float(0), Float(0)))
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
        return scheduleCommandBuffer(label: "GPUActor.Metrics.kdMetricsMean",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: true) {
            let ptr = accumBuffer.contents().bindMemory(to: Float.self, capacity: 2)
            let sumMSE = ptr[0]
            let sumCos = ptr[1]
            let invB = 1.0 / Float(batches)
            return (sumMSE * invB, sumCos * invB)
        }
    }

    public func crossEntropyMean(logits: Tensor,
                                 targets: [[Int]]) async throws -> Float {
        let readback = try await crossEntropyMeanDeferred(logits: logits, targets: targets)
        return try await readback.value()
    }

    public func crossEntropyMeanDeferred(logits: Tensor,
                                         targets: [[Int]]) async throws -> GPUReadback<Float> {
        precondition(logits.shape.count == 3, "crossEntropyMean expects logits [B,L,V]")
        let B = logits.shape[0]
        let L = logits.shape[1]
        let V = logits.shape[2]
        precondition(targets.count == B && targets.allSatisfy { $0.count == L },
                     "crossEntropyMean targets shape mismatch")
        let samples = B * L
        if samples == 0 || V == 0 {
            return GPUReadback(resolved: Float(0))
        }
        let pipelines = try ensureMetricsPipelines()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw GPUActorError.commandBufferUnavailable("Metrics.crossEntropyMean: command buffer creation failed")
        }
        commandBuffer.label = "GPUActor.Metrics.crossEntropyMean"
        let elemSize = MemoryLayout<Float>.size
        let logitsBytes = logits.count * elemSize
        let accumBytes = 2 * elemSize
        let targetsBytes = samples * MemoryLayout<Int32>.size
        let logitsBuffer = buffer(length: logitsBytes, label: "GPUActor.Metrics.ce.logits")
        let targetsBuffer = buffer(length: targetsBytes, label: "GPUActor.Metrics.ce.targets")
        let accumBuffer = buffer(length: accumBytes, label: "GPUActor.Metrics.ce.accum")
        logits.data.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(logitsBuffer.contents(), base, logitsBytes)
            }
        }
        var flatTargets = [Int32](repeating: 0, count: samples)
        for b in 0..<B {
            for t in 0..<L {
                let id = targets[b][t]
                precondition(id >= 0 && id < V, "crossEntropyMean target out of range")
                flatTargets[b * L + t] = Int32(id)
            }
        }
        flatTargets.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                memcpy(targetsBuffer.contents(), base, targetsBytes)
            }
        }
        memset(accumBuffer.contents(), 0, accumBytes)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw GPUActorError.commandBufferUnavailable("Metrics.crossEntropyMean: encoder creation failed")
        }
        encoder.label = "GPUActor.Metrics.crossEntropyMean.encoder"
        encoder.setComputePipelineState(pipelines.crossEntropyMean)
        encoder.setBuffer(logitsBuffer, offset: 0, index: 0)
        encoder.setBuffer(targetsBuffer, offset: 0, index: 1)
        encoder.setBuffer(accumBuffer, offset: 0, index: 2)
        var vSamples = Int32(samples)
        var vVocab = Int32(V)
        var eps: Float = 1e-6
        encoder.setBytes(&vSamples, length: MemoryLayout<Int32>.size, index: 3)
        encoder.setBytes(&vVocab, length: MemoryLayout<Int32>.size, index: 4)
        encoder.setBytes(&eps, length: MemoryLayout<Float>.size, index: 5)
        let threadsPerGroup = MTLSize(width: min(samples, 256), height: 1, depth: 1)
        let groups = MTLSize(width: (samples + threadsPerGroup.width - 1) / threadsPerGroup.width,
                             height: 1,
                             depth: 1)
        encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        return scheduleCommandBuffer(label: "GPUActor.Metrics.crossEntropyMean",
                                     commandBuffer: commandBuffer,
                                     deferUntilSync: true) {
            let ptr = accumBuffer.contents().bindMemory(to: Float.self, capacity: 2)
            let totalLoss = ptr[0]
            let count = max(ptr[1], 1.0)
            return totalLoss / count
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
