import Foundation

// MARK: - Simple dataset loader (JSONL or compact EFB binary)
// Supports:
// 1) JSONL: each line is a JSON object with optional tokens and required target
//    {
//      "text": "...",                    // optional
//      "input_ids": [Int]? ,              // optional
//      "attention_mask": [Int]? ,         // optional
//      "target": [Float]                  // required
//    }
// 2) EFB binary (.efb): compact binary with tokens + masks + target (no text)
//    magic: "EFB1" (ASCII), then little-endian fields
//    [4] magic, [4] num_samples (u32), [4] embedding_dim (u32), then per-sample:
//      [4] L (u32), [4*L] input_ids (u32), [1*L] attention_mask (u8), [4*E] target (f32)
//
// Usage example:
// let ds = try SimpleJSONLDataset(path: "/path/to/data.efb")
// for batch in ds.batches(batchSize: 32) { ... }

public struct JSONLSample: Sendable {
    public let text: String?
    public let inputIDs: [Int]?
    public let attentionMask: [Int]?
    public let target: [Float]
}

public enum JSONLBatch: Sendable {
    case text(texts: [String], targets: [[Float]])
    case tokens(inputIDs: [[Int]], attentionMask: [[Int]], targets: [[Float]])
}

public final class SimpleJSONLDataset: @unchecked Sendable {
    public let samples: [JSONLSample]
    public let hasTokens: Bool
    public let embeddingDim: Int

    public enum LoaderError: Error {
        case emptyFile
        case truncatedFile
        case badMagic
    }

    public init(path: String) throws {
        let url = URL(fileURLWithPath: path)
        // Try EFB first (by magic), otherwise fall back to JSONL text
        if let efb = try Self.tryLoadEFB(url: url) {
            self.samples = efb.samples
            self.embeddingDim = efb.embeddingDim
            self.hasTokens = true
            return
        }
        // JSONL path
        let text = try String(contentsOf: url, encoding: .utf8)
        var loaded: [JSONLSample] = []
        loaded.reserveCapacity(1024)
        let lines = text.split(whereSeparator: { $0 == "\n" || $0 == "\r\n" })
        for lineSub in lines {
            let line = String(lineSub)
            if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
            guard let obj = Self.parseJSONLine(line) else { continue }
            guard let target = obj["target"] as? [Any] else { continue }
            let targetFloats = target.compactMap { v -> Float? in
                if let f = v as? Float { return f }
                if let d = v as? Double { return Float(d) }
                if let i = v as? Int { return Float(i) }
                if let s = v as? String, let d = Double(s) { return Float(d) }
                return nil
            }
            if targetFloats.isEmpty { continue }

            var textField: String? = nil
            if let t = obj["text"] as? String, !t.isEmpty { textField = t }

            var inputIDs: [Int]? = nil
            if let idsAny = obj["input_ids"] as? [Any] {
                let ints = idsAny.compactMap { v -> Int? in
                    if let i = v as? Int { return i }
                    if let d = v as? Double { return Int(d) }
                    if let s = v as? String, let i = Int(s) { return i }
                    return nil
                }
                if !ints.isEmpty { inputIDs = ints }
            }

            var attention: [Int]? = nil
            if let amAny = obj["attention_mask"] as? [Any] {
                let ints = amAny.compactMap { v -> Int? in
                    if let i = v as? Int { return i }
                    if let d = v as? Double { return Int(d) }
                    if let s = v as? String, let i = Int(s) { return i }
                    return nil
                }
                if !ints.isEmpty { attention = ints }
            }

            loaded.append(JSONLSample(text: textField, inputIDs: inputIDs, attentionMask: attention, target: targetFloats))
        }

        guard let first = loaded.first else { throw LoaderError.emptyFile }
        self.embeddingDim = first.target.count
        self.samples = loaded
        self.hasTokens = loaded.first(where: { $0.inputIDs != nil && $0.attentionMask != nil }) != nil
    }

    // MARK: - Public API
    public func count() -> Int { samples.count }

    public func batches(batchSize: Int, dropLast: Bool = false, padTokens: Bool = true, padTokenID: Int = 0) -> AnyIterator<JSONLBatch> {
        precondition(batchSize > 0, "batchSize must be > 0")
        var idx = 0
        return AnyIterator {
            if idx >= self.samples.count { return nil }
            let end = min(idx + batchSize, self.samples.count)
            let slice = Array(self.samples[idx..<end])
            idx = end
            if dropLast && slice.count < batchSize { return nil }

            if self.hasTokens,
               slice.allSatisfy({ $0.inputIDs != nil && $0.attentionMask != nil }) {
                var ids: [[Int]] = []
                var mask: [[Int]] = []
                var targets: [[Float]] = []
                ids.reserveCapacity(slice.count)
                mask.reserveCapacity(slice.count)
                targets.reserveCapacity(slice.count)
                for s in slice {
                    ids.append(s.inputIDs!)
                    mask.append(s.attentionMask!)
                    targets.append(s.target)
                }
                if padTokens {
                    let maxLen = ids.map { $0.count }.max() ?? 0
                    for i in 0..<ids.count {
                        if ids[i].count < maxLen {
                            let padCount = maxLen - ids[i].count
                            ids[i].append(contentsOf: Array(repeating: padTokenID, count: padCount))
                            mask[i].append(contentsOf: Array(repeating: 0, count: padCount))
                        }
                    }
                }
                return .tokens(inputIDs: ids, attentionMask: mask, targets: targets)
            } else {
                // Fallback to text mode
                var texts: [String] = []
                var targets: [[Float]] = []
                texts.reserveCapacity(slice.count)
                targets.reserveCapacity(slice.count)
                for s in slice {
                    texts.append(s.text ?? "")
                    targets.append(s.target)
                }
                return .text(texts: texts, targets: targets)
            }
        }
    }

    // MARK: - JSON helper
    private static func parseJSONLine(_ line: String) -> [String: Any]? {
        guard let data = line.data(using: .utf8) else { return nil }
        do {
            if let dict = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                return dict
            }
            return nil
        } catch {
            return nil
        }
    }

    // MARK: - EFB loader
    private static func tryLoadEFB(url: URL) throws -> (samples: [JSONLSample], embeddingDim: Int)? {
        let fh = try FileHandle(forReadingFrom: url)
        defer { try? fh.close() }
        // Peek magic
        let headerSize = 12
        guard let header = try fh.read(upToCount: headerSize), header.count == headerSize else {
            return nil // too small to be EFB, fall back to JSONL
        }
        if header[0] != 0x45 || header[1] != 0x46 || header[2] != 0x42 || header[3] != 0x31 { // 'E''F''B''1'
            return nil
        }
        // Read rest of file into memory (for simplicity)
        let remaining = try fh.readToEnd() ?? Data()
        var data = Data()
        data.append(header)
        data.append(remaining)
        var offset = 4 // after magic
        let numSamples = Int(readUInt32LE(data, &offset))
        let embedDim = Int(readUInt32LE(data, &offset))
        var out: [JSONLSample] = []
        out.reserveCapacity(numSamples)
        for _ in 0..<numSamples {
            // L
            guard let l = readOptionalUInt32LE(data, &offset) else { throw LoaderError.truncatedFile }
            let L = Int(l)
            // input_ids: L * u32
            let idsBytes = L * 4
            guard offset + idsBytes <= data.count else { throw LoaderError.truncatedFile }
            var ids: [Int] = []
            ids.reserveCapacity(L)
            data.withUnsafeBytes { rawBuf in
                let base = rawBuf.baseAddress!.advanced(by: offset)
                let u32Ptr = base.assumingMemoryBound(to: UInt32.self)
                for i in 0..<L { ids.append(Int(UInt32(littleEndian: u32Ptr[i]))) }
            }
            offset += idsBytes
            // attention_mask: L * u8
            guard offset + L <= data.count else { throw LoaderError.truncatedFile }
            var mask: [Int] = []
            mask.reserveCapacity(L)
            data.withUnsafeBytes { rawBuf in
                let base = rawBuf.baseAddress!.advanced(by: offset)
                let u8Ptr = base.assumingMemoryBound(to: UInt8.self)
                for i in 0..<L { mask.append(Int(u8Ptr[i])) }
            }
            offset += L
            // target: embedDim * f32
            let tgtBytes = embedDim * 4
            guard offset + tgtBytes <= data.count else { throw LoaderError.truncatedFile }
            var target: [Float] = Array(repeating: 0, count: embedDim)
            _ = target.withUnsafeMutableBytes { tgtBuf in
                data.copyBytes(to: tgtBuf, from: offset..<(offset + tgtBytes))
            }
            // If running on big-endian (unlikely on macOS/arm64), we'd need to swap. Skipped for brevity.
            offset += tgtBytes
            out.append(JSONLSample(text: nil, inputIDs: ids, attentionMask: mask, target: target))
        }
        return (out, embedDim)
    }

    @inline(__always) private static func readUInt32LE(_ data: Data, _ offset: inout Int) -> UInt32 {
        let val: UInt32 = data.withUnsafeBytes { rawBuf in
            let base = rawBuf.baseAddress!.advanced(by: offset).assumingMemoryBound(to: UInt32.self)
            return UInt32(littleEndian: base.pointee)
        }
        offset += 4
        return val
    }

    @inline(__always) private static func readOptionalUInt32LE(_ data: Data, _ offset: inout Int) -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        return readUInt32LE(data, &offset)
    }
}
