import Foundation

struct HeaderBuilder {
    static func buildHeader(symbols: [String]) -> ([Int], Range<Int>) {
        var ids: [Int] = []
        let start = ids.count
        ids.append(Vocab.HDR_BOS.rawValue)
        for (i, name) in symbols.enumerated() {
            ids.append(VocabRanges.MAP_PAIR)
            ids.append(VocabRanges.IDX_BOS)
            ids.append(VocabUtil.idxID(i))
            ids.append(VocabRanges.IDX_END)
            ids.append(VocabRanges.BYTES_BOS)
            for b in name.utf8 { ids.append(VocabUtil.byteID(b)) }
            ids.append(VocabRanges.BYTES_END)
        }
        ids.append(Vocab.HDR_END.rawValue)
        let end = ids.count // mask covers [start, end)
        return (ids, start..<end)
    }
}