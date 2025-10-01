import Foundation

public struct TextToCubeEncoderConfig {
    // Architecture
    public var hiddenDim: Int              // embedding_dim (teacher model default: 768)
    public var maxLength: Int
    public var maxPosition: Int
    public var outputDim: Int              // encoder output dimension (e.g., 768 for DistilBERT targets)
    public var useTanhOutput: Bool         // whether to squash output to [-1,1]

    // TCN vs Transformer
    public var useTCN: Bool                // true => use TCN per roadmap
    public var tcnBlocks: Int              // number of TCN blocks
    public var kernelSize: Int             // conv kernel size
    public var dilationSchedule: [Int]     // size >= tcnBlocks recommended

    // Transformer params (CPU reference)
    public var numLayers: Int              // number of encoder layers
    public var numHeads: Int               // attention heads (start with 1)
    public var ffDim: Int                  // feedforward dim (e.g., 512); also used as TCN hidden
    public var dropout: Float              // kept for parity; not used in CPU ref

    // Projection
    public var useGPUProjection: Bool      // use FP16 GraphLinear for final projection

    // Seeding
    public var baseSeed: UInt64

    // Init
    public init(hiddenDim: Int = 256,
                maxLength: Int = 128,
                maxPosition: Int = 512,
                outputDim: Int = 768,
                useTanhOutput: Bool = false,
                useTCN: Bool = true,
                tcnBlocks: Int = 4,
                kernelSize: Int = 5,
                dilationSchedule: [Int] = [1,2,4,8],
                numLayers: Int = 2,
                numHeads: Int = 1,
                ffDim: Int = 512,
                dropout: Float = 0.0,
                useGPUProjection: Bool = true,
                baseSeed: UInt64 = 42) {
        self.hiddenDim = hiddenDim
        self.maxLength = maxLength
        self.maxPosition = maxPosition
        self.outputDim = outputDim
        self.useTanhOutput = useTanhOutput
        self.useTCN = useTCN
        self.tcnBlocks = tcnBlocks
        self.kernelSize = kernelSize
        self.dilationSchedule = dilationSchedule
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.ffDim = ffDim
        self.dropout = dropout
        self.useGPUProjection = useGPUProjection
        self.baseSeed = baseSeed
    }
}
