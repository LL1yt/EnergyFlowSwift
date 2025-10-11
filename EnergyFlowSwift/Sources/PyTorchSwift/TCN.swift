import Foundation
import EFCore

public struct TCNBlock {
    public let dim: Int           // in/out channels (residual)
    public let hidden: Int        // bottleneck channels
    public let kernelSize: Int
    public let dilation: Int
    public var conv1: GraphConv1D      // dim -> hidden, kernel=k, dilation=d
    public var conv2: GraphConv1D      // hidden -> dim, kernel=1
    public var ln: LayerNorm

    public init(dim: Int, hidden: Int, kernelSize: Int, dilation: Int, seed: UInt64) {
        self.dim = dim
        self.hidden = hidden
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.conv1 = GraphConv1D(inChannels: dim, outChannels: hidden, kernelSize: kernelSize, dilation: dilation, bias: true, seed: Seed.derive(seed, label: "conv1"))
        self.conv2 = GraphConv1D(inChannels: hidden, outChannels: dim, kernelSize: 1, dilation: 1, bias: true, seed: Seed.derive(seed, label: "conv2"))
        self.ln = LayerNorm(dim: dim)
    }

    // x: [B,L,dim], mask: [B][L]
    public func forward(_ x: Tensor,
                        mask: [[Int]],
                        on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        let B = x.shape[0], L = x.shape[1], D = x.shape[2]
        precondition(D == dim)
        // Path: LN -> Conv1D -> GELU -> Conv1D -> Residual
        let xFlat = x.reshaped([B * L, D])
        // GPU LayerNorm via Metal (FP16 IO, FP32 accumulators)
        let normFlat = try await gpu.layerNormForward(x: xFlat,
                                                      gamma: ln.gamma,
                                                      beta: ln.beta,
                                                      eps: ln.eps)
        let norm = normFlat.reshaped([B, L, D])
        // Conv1D -> GELU -> Conv1D(1x1) using MPSMatrix path
        let h = try await conv1.forwardAsync(norm, on: gpu)         // [B,L,hidden]
        let h1a = try await gpu.geluForward(x: h)
        var y = try await conv2.forwardAsync(h1a, on: gpu)
        // Residual and mask on GPU
        y = try await gpu.residualAdd(y: y, x: x)
        y = try await gpu.maskZero(y: y, mask: mask)
        return y
    }
}

public struct TCNEncoder {
    public var blocks: [TCNBlock]

    public init(numBlocks: Int, dim: Int, hidden: Int, kernelSize: Int, dilationSchedule: [Int], seed: UInt64) {
        self.blocks = []
        self.blocks.reserveCapacity(numBlocks)
        for i in 0..<numBlocks {
            let dil = i < dilationSchedule.count ? dilationSchedule[i] : 1
            let blkSeed = Seed.derive(seed, label: "tcn_\(i)")
            blocks.append(TCNBlock(dim: dim, hidden: hidden, kernelSize: kernelSize, dilation: dil, seed: blkSeed))
        }
    }

    // x: [B,L,D]
    public func forward(_ x: Tensor,
                        mask: [[Int]],
                        on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        var out = x
        for block in blocks {
            out = try await block.forward(out, mask: mask, on: gpu)
        }
        return out
    }
}
