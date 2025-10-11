import Foundation
import EFCore
import PyTorchSwift

// Minimal decoder without attention, conditioned on z via additive projection.
// ids: [B][L], z: [B, dim] -> logits: [B, L, V]
public final class TextDecoder {
    public let config: TextDecoderConfig
    private var embedding: Embedding
    public private(set) var stack: TCNStack
    private var condProj: GraphLinear   // z -> [dim]
    private var outProj: GraphLinear    // dim -> [vocab]

    public struct LastTCNCache {
        public let xIn: Tensor   // [B,L,D]
        public let norm: Tensor  // [B,L,D]
        public let h1: Tensor    // [B,L,H]
        public let h1a: Tensor   // [B,L,H]
    }

    public init(config: TextDecoderConfig, seed: UInt64 = 1234) {
        self.config = config
        self.embedding = Embedding(vocabSize: config.vocabSize, embeddingDim: config.dim, seed: Seed.derive(seed, label: "dec.embed"))
        // Build causal TCN stack (shared-type blocks)
        self.stack = TCNStack(numBlocks: config.nBlocks,
                              dim: config.dim,
                              hidden: config.hidden,
                              kernelSize: config.kernelSize,
                              dilationSchedule: config.dilationSchedule,
                              seed: seed)
        self.condProj = GraphLinear(inFeatures: config.dim, outFeatures: config.dim, bias: true, seed: Seed.derive(seed, label: "dec.cond"))
        self.outProj = GraphLinear(inFeatures: config.dim, outFeatures: config.vocabSize, bias: true, seed: Seed.derive(seed, label: "dec.out"))
    }

    // Forward with teacher-forcing tokens (no CE here, just logits)
    public func forward(ids: [[Int]],
                        z: Tensor,
                        on gpu: GPUActor = GPU.shared) async throws -> Tensor {
        let B = ids.count
        let L = ids.first?.count ?? 0
        precondition(L == config.maxLength, "ids must be fixed-length to match static graph; got L=\(L) expected \(config.maxLength)")
        precondition(z.shape == [B, config.dim], "z must be [B, dim]")
        // 1) Embedding
        var x = embedding.forward(ids: ids) // [B,L,dim]
        // 2) Conditioning (additive): cond = condProj(z) -> [B,dim], broadcast-add over time (GPU)
        var cproj = condProj
        let cond: Tensor
        do {
            cond = try await cproj.forwardAsync(z, on: gpu) // [B,dim]
            condProj = cproj
        } catch {
            fatalError("TextDecoder.condProj forward failed: \(error)")
        }
        x = try await gpu.addBroadcast2DInto3D(y: x, addBD: cond, sequenceLength: L)
        // 3) Causal TCN blocks
        let maskFull: [[Int]] = Array(repeating: Array(repeating: 1, count: L), count: B)
        let y = try await stack.forward(x, mask: maskFull, on: gpu)
        // 4) Vocab projection per time-step: reshape to [B*L, dim] -> [B*L, V]
        let flat = y.reshaped([B * L, config.dim])
        var oproj = outProj
        let logitsFlat: Tensor
        do {
            logitsFlat = try await oproj.forwardAsync(flat, on: gpu)
            outProj = oproj
        } catch {
            fatalError("TextDecoder.outProj forward failed: \(error)")
        }
        let logits = logitsFlat.reshaped([B, L, config.vocabSize])
        return logits
    }

    // Training helper: returns (flatFeatures [B*L, dim], logits [B,L,V])
    public func forwardForTraining(ids: [[Int]],
                                   z: Tensor,
                                   on gpu: GPUActor = GPU.shared) async throws -> (flatFeatures: Tensor, logits: Tensor, cache: LastTCNCache) {
        let B = ids.count
        let L = ids.first?.count ?? 0
        precondition(L == config.maxLength)
        precondition(z.shape == [B, config.dim])
        var x = embedding.forward(ids: ids)
        var cproj = condProj
        let cond: Tensor
        do {
            cond = try await cproj.forwardAsync(z, on: gpu)
            condProj = cproj
        } catch {
            fatalError("TextDecoder.forwardForTraining condProj failed: \(error)")
        }
        x = try await gpu.addBroadcast2DInto3D(y: x, addBD: cond, sequenceLength: L)
        let maskFull: [[Int]] = Array(repeating: Array(repeating: 1, count: L), count: B)
        // Run all but last block
        let nb = stack.blocks.count
        precondition(nb > 0)
        var xin = x
        if nb > 1 {
            for i in 0..<(nb - 1) {
                xin = try await stack.blocks[i].forward(xin, mask: maskFull, on: gpu)
            }
        }
        let last = stack.blocks[nb - 1]
        // LN on [B*L,D]
        let D = config.dim
        let xFlat = xin.reshaped([B * L, D])
        let normFlat = try await gpu.layerNormForward(x: xFlat,
                                                      gamma: last.ln.gamma,
                                                      beta: last.ln.beta,
                                                      eps: last.ln.eps)
        let norm = normFlat.reshaped([B, L, D])
        let h1 = try await last.conv1.forwardAsync(norm, on: gpu)
        let h1a = try await gpu.geluForward(x: h1)
        var y = try await last.conv2.forwardAsync(h1a, on: gpu)
        y = try await gpu.residualAdd(y: y, x: xin)
        // Flat for outProj
        let flat = y.reshaped([B * L, D])
        var oproj = outProj
        let logitsFlat: Tensor
        do {
            logitsFlat = try await oproj.forwardAsync(flat, on: gpu)
            outProj = oproj
        } catch {
            fatalError("TextDecoder.forwardForTraining outProj failed: \(error)")
        }
        let logits = logitsFlat.reshaped([B, L, config.vocabSize])
        let cache = LastTCNCache(xIn: xin, norm: norm, h1: h1, h1a: h1a)
        return (flat, logits, cache)
    }
    // Accessors for out projection (for training)
    public func getOutProjParams() -> (weight: Tensor, bias: Tensor?) { (outProj.weight, outProj.bias) }
    public func setOutProjParams(weight: Tensor, bias: Tensor?) { outProj.weight = weight; outProj.bias = bias }
    public func invalidateOutProjCache() { outProj.invalidateCache() }
    // Accessors for last TCN block params (for training)
    public func getLastBlockParams() -> (w1: Tensor, b1: Tensor?, w2: Tensor, b2: Tensor?, gamma: Tensor, beta: Tensor) {
        let nb = stack.blocks.count
        precondition(nb > 0)
        let last = stack.blocks[nb - 1]
        return (last.conv1.weight, last.conv1.bias ?? Tensor.zeros([0]), last.conv2.weight, last.conv2.bias ?? Tensor.zeros([0]), last.ln.gamma, last.ln.beta)
    }
    public func setLastBlockParams(w1: Tensor, b1: Tensor?, w2: Tensor, b2: Tensor?, gamma: Tensor, beta: Tensor) {
        let nb = stack.blocks.count
        precondition(nb > 0)
        stack.blocks[nb - 1].conv1.weight = w1
        stack.blocks[nb - 1].conv1.bias = b1
        stack.blocks[nb - 1].conv2.weight = w2
        stack.blocks[nb - 1].conv2.bias = b2
        stack.blocks[nb - 1].ln.gamma = gamma
        stack.blocks[nb - 1].ln.beta = beta
    }
    public func invalidateLastBlockCaches() {
        let nb = stack.blocks.count
        precondition(nb > 0)
        stack.blocks[nb - 1].conv1.invalidateCache()
        stack.blocks[nb - 1].conv2.invalidateCache()
    }
}
