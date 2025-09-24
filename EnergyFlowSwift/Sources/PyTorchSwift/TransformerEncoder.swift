import Foundation
import EFCore

// Minimal TransformerEncoder (CPU, pre-LN, single-head attention)
// Shapes: x is [B, L, H]
public struct TransformerEncoderLayer {
    public let hidden: Int
    public let ffDim: Int
    public var attn: SingleHeadSelfAttention
    public var ln1: LayerNorm
    public var ln2: LayerNorm
    public var ffn1: Linear
    public var ffn2: Linear

    public init(hidden: Int, ffDim: Int, seed: UInt64) {
        self.hidden = hidden
        self.ffDim = ffDim
        self.attn = SingleHeadSelfAttention(hidden: hidden, seed: Seed.derive(seed, label: "attn"))
        self.ln1 = LayerNorm(dim: hidden)
        self.ln2 = LayerNorm(dim: hidden)
        self.ffn1 = Linear(inFeatures: hidden, outFeatures: ffDim, seed: Seed.derive(seed, label: "ff1"))
        self.ffn2 = Linear(inFeatures: ffDim, outFeatures: hidden, seed: Seed.derive(seed, label: "ff2"))
    }

    // x: [B,L,H], mask: [B][L]
    public func forward(_ x: Tensor, mask: [[Int]]) -> Tensor {
        let b = x.shape[0], l = x.shape[1], h = x.shape[2]
        precondition(h == hidden)
        // Pre-LN
        let xFlat1 = x.reshaped([b * l, h])
        let norm1 = ln1.forward(xFlat1).reshaped([b, l, h])
        let a = attn.forward(norm1, mask: mask)          // [B,L,H]
        var y = Tensor.zeros([b, l, h])
        for i in 0..<(b*l*h) { y.data[i] = x.data[i] + a.data[i] } // residual 1

        let yFlat = y.reshaped([b * l, h])
        let norm2 = ln2.forward(yFlat)                    // [B*L,H]
        var f = ffn1.forward(norm2)                       // [B*L, ff]
        f = Activations.gelu(f)
        f = ffn2.forward(f)                               // [B*L, H]
        var z = Tensor.zeros([b * l, h])
        for i in 0..<(b*l*h) { z.data[i] = yFlat.data[i] + f.data[i] } // residual 2
        return z.reshaped([b, l, h])
    }
}

public struct TransformerEncoder {
    public var layers: [TransformerEncoderLayer]

    public init(numLayers: Int, hidden: Int, ffDim: Int, seed: UInt64) {
        self.layers = []
        self.layers.reserveCapacity(numLayers)
        for i in 0..<numLayers {
            let layerSeed = Seed.derive(seed, label: "layer_\(i)")
            layers.append(TransformerEncoderLayer(hidden: hidden, ffDim: ffDim, seed: layerSeed))
        }
    }

    public func forward(_ x: Tensor, mask: [[Int]]) -> Tensor {
        var out = x
        for layer in layers {
            out = layer.forward(out, mask: mask)
        }
        return out
    }
}
