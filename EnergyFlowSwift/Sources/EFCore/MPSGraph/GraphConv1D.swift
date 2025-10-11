import Foundation

public final class GraphConv1D {
    public let inChannels: Int
    public let outChannels: Int
    public let kernelSize: Int
    public let dilation: Int
    public var weight: Tensor   // [Cout, Cin, K]
    public var bias: Tensor?

    private let cacheID: UUID
    private var cacheVersion: UInt64

    public init(inChannels: Int,
                outChannels: Int,
                kernelSize: Int,
                dilation: Int = 1,
                bias: Bool = true,
                seed: UInt64 = 42) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.dilation = max(1, dilation)
        self.weight = Tensor.randomUniform(
            [outChannels, inChannels, kernelSize],
            min: -1.0 / Float(inChannels * kernelSize),
            max: 1.0 / Float(inChannels * kernelSize),
            seed: seed
        )
        self.bias = bias
            ? Tensor.randomUniform([outChannels], min: -0.001, max: 0.001, seed: seed &+ 1)
            : nil
        self.cacheID = UUID()
        self.cacheVersion = 0
    }

    public func invalidateCache() {
        cacheVersion &+= 1
    }

    public func forwardAsync(_ x: Tensor, on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        precondition(x.shape.count == 3, "GraphConv1D.forward expects [B,L,Cin]")
        precondition(x.shape[2] == inChannels, "Cin mismatch: got \\(x.shape[2]), expected \\(inChannels)")
        let key = cacheID
        let ver = cacheVersion
        let inC = inChannels
        let outC = outChannels
        let ksz = kernelSize
        let dil = dilation
        let w = weight
        let b = bias
        return try await gpu.conv1DForward(
            key: key,
            version: ver,
            inChannels: inC,
            outChannels: outC,
            kernelSize: ksz,
            dilation: dil,
            weight: w,
            bias: b,
            x: x
        )
    }

}
