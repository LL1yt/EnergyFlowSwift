import Foundation

public struct GraphLinear {
    public let inFeatures: Int
    public let outFeatures: Int
    public var weight: Tensor
    public var bias: Tensor?

    private let cacheID: UUID
    private var cacheVersion: UInt64

    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, seed: UInt64 = 42) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        self.weight = Tensor.randomUniform([outFeatures, inFeatures],
            min: -1.0 / Float(inFeatures),
            max: 1.0 / Float(inFeatures),
            seed: seed)
        self.bias = bias
            ? Tensor.randomUniform([outFeatures], min: -0.001, max: 0.001, seed: seed &+ 1)
            : nil
        self.cacheID = UUID()
        self.cacheVersion = 0
    }

    public mutating func invalidateCache() {
        cacheVersion &+= 1
    }

    public func forwardAsync(_ x: Tensor, on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        precondition(x.shape.count == 2 && x.shape[1] == inFeatures, "GraphLinear.forward expects [B, inFeatures]")
        let key = cacheID
        let ver = cacheVersion
        let w = weight
        let b = bias
        return try await gpu.linearForward(key: key,
                                          version: ver,
                                          inFeatures: inFeatures,
                                          outFeatures: outFeatures,
                                          weight: w,
                                          bias: b,
                                          x: x)
    }

    public func forwardDeferred(_ x: Tensor, on gpu: GPUActor = GPU.shared) async throws -> GPUReadback<Tensor> {
        precondition(x.shape.count == 2 && x.shape[1] == inFeatures, "GraphLinear.forward expects [B, inFeatures]")
        let key = cacheID
        let ver = cacheVersion
        let w = weight
        let b = bias
        return try await gpu.linearForwardDeferred(key: key,
                                                   version: ver,
                                                   inFeatures: inFeatures,
                                                   outFeatures: outFeatures,
                                                   weight: w,
                                                   bias: b,
                                                   x: x)
    }

    public func gradientsGPUAsync(X: Tensor, dY: Tensor, on gpu: GPUActor = GPU.shared) async throws -> (dW: Tensor, dB: Tensor) {
        precondition(X.shape.count == 2 && dY.shape.count == 2, "GraphLinear.gradientsGPU expects 2D tensors")
        let batch = X.shape[0]
        precondition(X.shape[1] == inFeatures && dY.shape[0] == batch && dY.shape[1] == outFeatures,
                     "Shape mismatch: X [B, In], dY [B, Out]")
        let key = cacheID
        let ver = cacheVersion
        let w = weight
        let b = bias
        return try await gpu.linearGradients(key: key,
                                            version: ver,
                                            inFeatures: inFeatures,
                                            outFeatures: outFeatures,
                                            weight: w,
                                            X: X,
                                            dY: dY,
                                            bias: b)
    }

    public func gradientsGPUDeferred(X: Tensor, dY: Tensor, on gpu: GPUActor = GPU.shared) async throws -> GPUReadback<(Tensor, Tensor)> {
        precondition(X.shape.count == 2 && dY.shape.count == 2, "GraphLinear.gradientsGPU expects 2D tensors")
        let batch = X.shape[0]
        precondition(X.shape[1] == inFeatures && dY.shape[0] == batch && dY.shape[1] == outFeatures,
                     "Shape mismatch: X [B, In], dY [B, Out]")
        let key = cacheID
        let ver = cacheVersion
        let w = weight
        let b = bias
        return try await gpu.linearGradientsDeferred(key: key,
                                                     version: ver,
                                                     inFeatures: inFeatures,
                                                     outFeatures: outFeatures,
                                                     weight: w,
                                                     X: X,
                                                     dY: dY,
                                                     bias: b)
    }

    public func gradientsGPUFromHandleDeferred(xHandle: GPUTensorHandle,
                                               dY: Tensor,
                                               on gpu: GPUActor = GPU.shared,
                                               consumeInput: Bool = false) async throws -> GPUReadback<(Tensor, Tensor)> {
        precondition(xHandle.cols == inFeatures, "GraphLinear.gradientsGPUFromHandle expects handle cols == inFeatures")
        let batch = xHandle.rows
        precondition(dY.shape.count == 2 && dY.shape[0] == batch && dY.shape[1] == outFeatures,
                     "Shape mismatch: dY [B, Out]")
        let key = cacheID
        let ver = cacheVersion
        let w = weight
        let b = bias
        return try await gpu.linearGradientsFromHandleDeferred(key: key,
                                                               version: ver,
                                                               inFeatures: inFeatures,
                                                               outFeatures: outFeatures,
                                                               weight: w,
                                                               xHandle: xHandle,
                                                               dY: dY,
                                                               bias: b,
                                                               consumeInput: consumeInput)
    }

    public func inputGradientsGPUAsync(dY: Tensor, on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        precondition(dY.shape.count == 2 && dY.shape[1] == outFeatures, "GraphLinear.inputGradientsGPU expects [B, outFeatures]")
        let key = cacheID
        let ver = cacheVersion
        let w = weight
        let b = bias
        return try await gpu.linearInputGradients(key: key,
                                                version: ver,
                                                inFeatures: inFeatures,
                                                outFeatures: outFeatures,
                                                weight: w,
                                                bias: b,
                                                dY: dY)
    }

    public func inputGradientsGPUDeferred(dY: Tensor, on gpu: GPUActor = GPU.shared) async throws -> GPUReadback<Tensor> {
        precondition(dY.shape.count == 2 && dY.shape[1] == outFeatures, "GraphLinear.inputGradientsGPU expects [B, outFeatures]")
        let key = cacheID
        let ver = cacheVersion
        let w = weight
        let b = bias
        return try await gpu.linearInputGradientsDeferred(key: key,
                                                          version: ver,
                                                          inFeatures: inFeatures,
                                                          outFeatures: outFeatures,
                                                          weight: w,
                                                          bias: b,
                                                          dY: dY)
    }

    // New: forward from a GPU handle without host copy
    public func forwardFromHandleDeferred(_ handle: GPUTensorHandle,
                                          on gpu: GPUActor = GPU.shared) async throws -> GPUReadback<Tensor> {
        precondition(handle.cols == inFeatures, "GraphLinear.forwardFromHandle: handle cols (\(handle.cols)) must equal inFeatures (\(inFeatures))")
        let key = cacheID
        let ver = cacheVersion
        let w = weight
        let b = bias
        return try await gpu.linearForwardFromHandleDeferred(key: key,
                                                             version: ver,
                                                             inFeatures: inFeatures,
                                                             outFeatures: outFeatures,
                                                             weight: w,
                                                             bias: b,
                                                             xHandle: handle)
    }

}
