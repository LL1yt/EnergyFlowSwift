import Foundation
import EFCore

public struct TCNStack {
    public var blocks: [TCNBlock]

    public init(numBlocks: Int, dim: Int, hidden: Int, kernelSize: Int, dilationSchedule: [Int], seed: UInt64) {
        self.blocks = []
        self.blocks.reserveCapacity(numBlocks)
        for i in 0..<numBlocks {
            let dil = i < dilationSchedule.count ? dilationSchedule[i] : 1
            let blkSeed = Seed.derive(seed, label: "tcn_stack_\(i)")
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
