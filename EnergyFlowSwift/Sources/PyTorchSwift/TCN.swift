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
    public func forward(_ x: Tensor, mask: [[Int]]) -> Tensor {
        let B = x.shape[0], L = x.shape[1], D = x.shape[2]
        precondition(D == dim)
        // Path: LN -> Conv1D -> GELU -> Conv1D -> Residual
        let xFlat = x.reshaped([B * L, D])
        let normFlat = LNExecCache.shared.runForward(x: xFlat, gamma: ln.gamma, beta: ln.beta, eps: ln.eps)
        let norm = normFlat.reshaped([B, L, D])
        let h = conv1.forward(norm)         // [B,L,hidden]
        // Try fused GELU+MatMul (conv2 1x1) via MPSGraph executable cache
        let N = B * L
        let hFlat = h.reshaped([N, hidden])
        let w2 = conv2.weight2DPointwise() // [dim, hidden], cached
        let b2 = conv2.bias
        let yFlat: Tensor
        if let b = b2 {
            yFlat = GeLUGemmCache.shared.runForward(x: hFlat, W: w2, bias: b)
        } else {
            yFlat = GeLUGemmCache.shared.runForward(x: hFlat, W: w2, bias: nil)
        }
        var y = yFlat.reshaped([B, L, dim])
        // Residual
        for idx in 0..<(B*L*D) { y.data[idx] += x.data[idx] }
        // Zero masked positions
        for b in 0..<B { for t in 0..<L { if mask[b][t] == 0 {
            let base = (b * L + t) * D
            for d in 0..<D { y.data[base + d] = 0 }
        } } }
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
    public func forward(_ x: Tensor, mask: [[Int]]) -> Tensor {
        var out = x
        for b in blocks { out = b.forward(out, mask: mask) }
        return out
    }
}