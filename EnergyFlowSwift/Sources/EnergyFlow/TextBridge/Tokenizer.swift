import Foundation

public struct EncodedBatch {
    public let ids: [[Int]]       // [B][L]
    public let attentionMask: [[Int]] // [B][L]
    public init(ids: [[Int]], attentionMask: [[Int]]) {
        self.ids = ids
        self.attentionMask = attentionMask
    }
}

public struct SimpleTokenizer {
    // Minimal vocab with dynamic growth for demo purposes
    private var vocab: [String: Int] = ["[PAD]": 0, "[UNK]": 1]
    private var nextId: Int = 2

    public init() {}

    public mutating func encodeBatch(_ texts: [String], maxLength: Int = 128) -> EncodedBatch {
        var ids: [[Int]] = []
        var mask: [[Int]] = []
        ids.reserveCapacity(texts.count)
        mask.reserveCapacity(texts.count)
        for t in texts {
            let toks = t.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).map { String($0).lowercased() }
            var row: [Int] = []
            row.reserveCapacity(maxLength)
            var mrow: [Int] = []
            mrow.reserveCapacity(maxLength)
            for i in 0..<maxLength {
                if i < toks.count {
                    let tok = toks[i]
                    if let id = vocab[tok] {
                        row.append(id)
                    } else {
                        vocab[tok] = nextId
                        row.append(nextId)
                        nextId += 1
                    }
                    mrow.append(1)
                } else {
                    row.append(0) // [PAD]
                    mrow.append(0)
                }
            }
            ids.append(row)
            mask.append(mrow)
        }
        return EncodedBatch(ids: ids, attentionMask: mask)
    }
}
