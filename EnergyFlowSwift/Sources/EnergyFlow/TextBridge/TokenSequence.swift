import Foundation

public struct TokenSequence {
    public var ids: [Int32]      // fixed length 256 with padding
    public var mask: [UInt8]     // 0/1, length 256
    public var len: Int          // real length before PAD

    public init(ids: [Int32], mask: [UInt8], len: Int) {
        precondition(ids.count == 256, "ids must be length 256")
        precondition(mask.count == 256, "mask must be length 256")
        precondition(len >= 0 && len <= 256, "len must be in 0...256")
        self.ids = ids
        self.mask = mask
        self.len = len
    }
}