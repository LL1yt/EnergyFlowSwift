import Foundation

public struct TextToCubeEncoderConfig {
    // Architecture
    public var hiddenDim: Int
    public var maxLength: Int
    public var maxPosition: Int
    public var outputDim: Int // dimension of the encoder output (e.g., 768 for DistilBERT targets)
    public var useTanhOutput: Bool // whether to squash output to [-1,1]; for embeddings, usually false

    // Seeding
    public var baseSeed: UInt64

    // Init
    public init(hiddenDim: Int = 256,
                maxLength: Int = 128,
                maxPosition: Int = 512,
                outputDim: Int = 768,
                useTanhOutput: Bool = false,
                baseSeed: UInt64 = 42) {
        self.hiddenDim = hiddenDim
        self.maxLength = maxLength
        self.maxPosition = maxPosition
        self.outputDim = outputDim
        self.useTanhOutput = useTanhOutput
        self.baseSeed = baseSeed
    }
}
