import Foundation
import EFCore
import PyTorchSwift

public struct TextDecoderConfig {
    public var vocabSize: Int
    public var dim: Int
    public var hidden: Int
    public var nBlocks: Int
    public var kernelSize: Int
    public var dilationSchedule: [Int]
    public var maxLength: Int

    public init(vocabSize: Int = 30000,
                dim: Int = 256,
                hidden: Int = 512,
                nBlocks: Int = 4,
                kernelSize: Int = 5,
                dilationSchedule: [Int] = [1,2,4,8],
                maxLength: Int = 128) {
        self.vocabSize = vocabSize
        self.dim = dim
        self.hidden = hidden
        self.nBlocks = nBlocks
        self.kernelSize = kernelSize
        self.dilationSchedule = dilationSchedule
        self.maxLength = maxLength
    }
}
