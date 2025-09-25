import Foundation

public struct TextCubeTokenizer: TextTokenizer {
    public init() {}

    public func encode(_ text: String) throws -> TokenSequence {
        // 1) Lex and parse
        let tokens = try Lexer.lex(text)
        let ast = try Parser.parse(tokens)
        // 2) Build symbol table (order of first appearance across identifiers)
        var sym = SymbolTable()
        for s in ast.stmts {
            if case .ident(let n) = s.left { _ = try? sym.index(for: n) }
            if case .ident(let n) = s.right { _ = try? sym.index(for: n) }
        }
        // 3) Header
        let (header, headerRange) = HeaderBuilder.buildHeader(symbols: sym.indexToName)
        // 4) Body
        let body = try BodyBuilder.buildBody(ast: ast, symtab: sym)
        // 5) EOS and PAD
        var ids = header + body
        ids.append(Vocab.EOS.rawValue)
        let realLen = ids.count
        if realLen > 256 { throw TextTokenizerError.sequenceTooLong(realLen) }
        if ids.count < 256 { ids.append(contentsOf: Array(repeating: Vocab.PAD.rawValue, count: 256 - ids.count)) }
        // 6) mask: header protected
        var mask = Array(repeating: UInt8(0), count: 256)
        var pos = 0
        for i in 0..<header.count { if pos + i < 256 { mask[pos + i] = 1 } }
        // Return
        let out = TokenSequence(ids: ids.map { Int32($0) }, mask: mask, len: realLen)
        return out
    }

    public func decode(_ seq: TokenSequence) throws -> String {
        let ids = seq.ids.map { Int($0) }
        // 1) Find header
        guard let hdrStart = ids.firstIndex(of: Vocab.HDR_BOS.rawValue) else { throw TextTokenizerError.headerMissing }
        guard let hdrEnd = ids.firstIndex(of: Vocab.HDR_END.rawValue), hdrEnd > hdrStart else { throw TextTokenizerError.headerMissing }
        var idxToName: [String] = []
        var i = hdrStart + 1
        while i < hdrEnd {
            guard i < hdrEnd && ids[i] == VocabRanges.MAP_PAIR else { throw TextTokenizerError.internalError("MAP_PAIR expected in header") }
            i += 1
            guard i + 3 <= hdrEnd else { throw TextTokenizerError.internalError("Truncated IDX in header") }
            guard ids[i] == VocabRanges.IDX_BOS else { throw TextTokenizerError.internalError("IDX_BOS expected") }
            i += 1
            guard let idx = VocabUtil.idToIdx(ids[i]) else { throw TextTokenizerError.internalError("IDX_k expected") }
            i += 1
            guard ids[i] == VocabRanges.IDX_END else { throw TextTokenizerError.internalError("IDX_END expected") }
            i += 1
            guard i < hdrEnd && ids[i] == VocabRanges.BYTES_BOS else { throw TextTokenizerError.internalError("BYTES_BOS expected") }
            i += 1
            var bytes: [UInt8] = []
            while i < hdrEnd && ids[i] != VocabRanges.BYTES_END {
                guard let b = VocabUtil.idToByte(ids[i]) else { throw TextTokenizerError.internalError("BYTE_* expected") }
                bytes.append(b); i += 1
            }
            guard i < hdrEnd && ids[i] == VocabRanges.BYTES_END else { throw TextTokenizerError.internalError("BYTES_END expected") }
            i += 1
            let name = String(decoding: bytes, as: UTF8.self)
            // Ensure capacity
            if idx >= idxToName.count { idxToName.append(contentsOf: Array(repeating: "", count: idx - idxToName.count + 1)) }
            idxToName[idx] = name
        }
        // 2) Find body
        guard let bodyStart = ids.firstIndex(of: Vocab.BODY_BOS.rawValue) else { throw TextTokenizerError.bodyMissing }
        guard let bodyEnd = ids.firstIndex(of: Vocab.BODY_END.rawValue), bodyEnd > bodyStart else { throw TextTokenizerError.bodyMissing }
        var out = "("
        var j = bodyStart + 1
        // Expect LP
        guard j < bodyEnd && ids[j] == Vocab.LP.rawValue else { throw TextTokenizerError.internalError("LP expected in body") }
        j += 1
        var first = true
        while j < bodyEnd {
            if ids[j] == Vocab.RP.rawValue { j += 1; break }
            if !first {
                out.append(", ")
            }
            // term
            out.append(try decodeTerm(ids: ids, pos: &j, idxToName: idxToName))
            // rel
            guard j < bodyEnd else { throw TextTokenizerError.internalError("Relation expected") }
            out.append(decodeRel(ids[j]))
            j += 1
            // term
            out.append(try decodeTerm(ids: ids, pos: &j, idxToName: idxToName))
            // optional comma between statements
            if j < bodyEnd && ids[j] == Vocab.COMMA.rawValue {
                j += 1
            }
            first = false
        }
        out.append(")")
        return out
    }

    private func decodeRel(_ id: Int) -> String {
        switch id {
        case Vocab.LT.rawValue: return "<"
        case Vocab.LE.rawValue: return "<="
        case Vocab.EQ.rawValue: return "="
        case Vocab.NE.rawValue: return "!="
        case Vocab.GT.rawValue: return ">"
        case Vocab.GE.rawValue: return ">="
        default: return "?" // shouldn't happen
        }
    }

    private func decodeTerm(ids: [Int], pos j: inout Int, idxToName: [String]) throws -> String {
        if ids[j] == VocabRanges.IDX_BOS {
            j += 1
            guard j < ids.count, let idx = VocabUtil.idToIdx(ids[j]) else { throw TextTokenizerError.internalError("IDX_k expected") }
            j += 1
            guard j < ids.count, ids[j] == VocabRanges.IDX_END else { throw TextTokenizerError.internalError("IDX_END expected") }
            j += 1
            guard idx < idxToName.count else { throw TextTokenizerError.unknownIndex(idx) }
            return idxToName[idx]
        }
        if ids[j] == Vocab.NUM_BOS.rawValue {
            j += 1
            var s = ""
            if j < ids.count && (ids[j] == Vocab.SIGN_PLUS.rawValue || ids[j] == Vocab.SIGN_MINUS.rawValue) {
                s.append(ids[j] == Vocab.SIGN_PLUS.rawValue ? "+" : "-")
                j += 1
            }
            var hadDigit = false
            while j < ids.count, let d = VocabUtil.idToDigit(ids[j]) {
                s.append(String(d))
                hadDigit = true
                j += 1
            }
            guard hadDigit, j < ids.count, ids[j] == Vocab.NUM_END.rawValue else { throw TextTokenizerError.internalError("NUM_END expected") }
            j += 1
            return s
        }
        throw TextTokenizerError.internalError("Unexpected token in term at pos \(j)")
    }
}