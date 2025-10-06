import Foundation
import EFCore
import PyTorchSwift

// Minimal decoder without attention, conditioned on z via additive projection.
// ids: [B][L], z: [B, dim] -> logits: [B, L, V]
public final class TextDecoder {
    public let config: TextDecoderConfig
    private var embedding: Embedding
    private var blocks: [TCNBlock]
    private var condProj: GraphLinear   // z -> [dim]
    private var outProj: GraphLinear    // dim -> [vocab]

    public init(config: TextDecoderConfig, seed: UInt64 = 1234) {
        self.config = config
        self.embedding = Embedding(vocabSize: config.vocabSize, embeddingDim: config.dim, seed: Seed.derive(seed, label: "dec.embed"))
        // Build causal TCN blocks
        self.blocks = []
        self.blocks.reserveCapacity(config.nBlocks)
        for i in 0..<config.nBlocks {
            let dil = i < config.dilationSchedule.count ? config.dilationSchedule[i] : 1
            let blkSeed = Seed.derive(seed, label: "dec.tcn.\(i)")
            blocks.append(TCNBlock(dim: config.dim, hidden: config.hidden, kernelSize: config.kernelSize, dilation: dil, seed: blkSeed))
        }
        self.condProj = GraphLinear(inFeatures: config.dim, outFeatures: config.dim, bias: true, seed: Seed.derive(seed, label: "dec.cond"))
        self.outProj = GraphLinear(inFeatures: config.dim, outFeatures: config.vocabSize, bias: true, seed: Seed.derive(seed, label: "dec.out"))
    }

    // Forward with teacher-forcing tokens (no CE here, just logits)
    public func forward(ids: [[Int]], z: Tensor) -> Tensor {
        let B = ids.count
        let L = ids.first?.count ?? 0
        precondition(L == config.maxLength, "ids must be fixed-length to match static graph; got L=\(L) expected \(config.maxLength)")
        precondition(z.shape == [B, config.dim], "z must be [B, dim]")
        // 1) Embedding
        var x = embedding.forward(ids: ids) // [B,L,dim]
        // 2) Conditioning (additive): cond = condProj(z) -> [B,dim], broadcast-add over time
        var cproj = condProj
        let cond = try! cproj.forward(z) // [B,dim]
        self.condProj = cproj
        for b in 0..<B { for t in 0..<L {
            let baseX = (b * L + t) * config.dim
            let baseC = b * config.dim
            for d in 0..<config.dim { x.data[baseX + d] += cond.data[baseC + d] }
        } }
        // 3) Causal TCN blocks
        var y = x
        let maskFull: [[Int]] = Array(repeating: Array(repeating: 1, count: L), count: B)
        for blk in blocks { y = blk.forward(y, mask: maskFull) }
        // 4) Vocab projection per time-step: reshape to [B*L, dim] -> [B*L, V]
        let flat = y.reshaped([B * L, config.dim])
        var oproj = outProj
        let logitsFlat = try! oproj.forward(flat)
        self.outProj = oproj
        let logits = logitsFlat.reshaped([B, L, config.vocabSize])
        return logits
    }
}
