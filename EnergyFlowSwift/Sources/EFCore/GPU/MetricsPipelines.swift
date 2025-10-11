import Metal

struct MetricsPipelines {
    let library: MTLLibrary
    let kdMetricsReduce: MTLComputePipelineState
    let crossEntropyMean: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: MetricsMetalLibrary.source, options: nil)
        guard let kdFunction = library.makeFunction(name: "kd_metrics_reduce_f32") else {
            throw GPUActorError.pipelineFunctionMissing("kd_metrics_reduce_f32")
        }
        guard let ceFunction = library.makeFunction(name: "cross_entropy_mean_logits_f32") else {
            throw GPUActorError.pipelineFunctionMissing("cross_entropy_mean_logits_f32")
        }
        self.library = library
        self.kdMetricsReduce = try device.makeComputePipelineState(function: kdFunction)
        self.crossEntropyMean = try device.makeComputePipelineState(function: ceFunction)
    }
}
