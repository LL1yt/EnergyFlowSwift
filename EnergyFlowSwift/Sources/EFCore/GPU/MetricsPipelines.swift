import Metal

struct MetricsPipelines {
    let library: MTLLibrary
    let kdMetricsReduce: MTLComputePipelineState

    init(device: MTLDevice) throws {
        let library = try device.makeLibrary(source: MetricsMetalLibrary.source, options: nil)
        guard let kdFunction = library.makeFunction(name: "kd_metrics_reduce_f32") else {
            throw GPUActorError.pipelineFunctionMissing("kd_metrics_reduce_f32")
        }
        self.library = library
        self.kdMetricsReduce = try device.makeComputePipelineState(function: kdFunction)
    }
}
