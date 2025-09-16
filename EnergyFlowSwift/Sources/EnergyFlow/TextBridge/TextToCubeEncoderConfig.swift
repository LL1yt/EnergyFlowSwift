import Foundation

public struct TextToCubeEncoderConfig {
    // Architecture
    public var hiddenDim: Int
    public var maxLength: Int
    public var maxPosition: Int

    // Seeding
    public var baseSeed: UInt64

    // Init
    public init(hiddenDim: Int = 256,
                maxLength: Int = 128,
                maxPosition: Int = 512,
                baseSeed: UInt64 = 42) {
        self.hiddenDim = hiddenDim
        self.maxLength = maxLength
        self.maxPosition = maxPosition
        self.baseSeed = baseSeed
    }
}
