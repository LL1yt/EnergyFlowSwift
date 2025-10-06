import Foundation

public struct TextToCubeEncoderConfig {
    // Architecture (TCN-first)
    public var hiddenDim: Int              // embedding_dim (student)
    public var maxLength: Int
    public var outputDim: Int              // projection dim (e.g., 768)
    public var useTanhOutput: Bool

    // TCN parameters
    public var tcnBlocks: Int              // number of TCN blocks
    public var kernelSize: Int             // conv kernel size
    public var dilationSchedule: [Int]     // size >= tcnBlocks recommended
    public var ffDim: Int                  // hidden channels in TCN

    // Projection
    public var useGPUProjection: Bool      // FP16 GPU projection

    // Backward helpers
    public var useGPUIm2ColCol2Im: Bool    // GPU im2col/col2im for last TCN backward

    // Seeding
    public var baseSeed: UInt64

    // Init
    public init(hiddenDim: Int = 256,
                maxLength: Int = 128,
                outputDim: Int = 768,
                useTanhOutput: Bool = false,
                tcnBlocks: Int = 4,
                kernelSize: Int = 5,
                dilationSchedule: [Int] = [1,2,4,8],
                ffDim: Int = 512,
                useGPUProjection: Bool = true,
                useGPUIm2ColCol2Im: Bool = true,
                baseSeed: UInt64 = 42) {
        self.hiddenDim = hiddenDim
        self.maxLength = maxLength
        self.outputDim = outputDim
        self.useTanhOutput = useTanhOutput
        self.tcnBlocks = tcnBlocks
        self.kernelSize = kernelSize
        self.dilationSchedule = dilationSchedule
        self.ffDim = ffDim
        self.useGPUProjection = useGPUProjection
        self.useGPUIm2ColCol2Im = useGPUIm2ColCol2Im
        self.baseSeed = baseSeed
    }
}
