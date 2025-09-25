import Foundation

struct BodyBuilder {
    static func buildBody(ast: BlockAST, symtab: SymbolTable) throws -> [Int] {
        var ids: [Int] = []
        ids.append(Vocab.BODY_BOS.rawValue)
        ids.append(Vocab.LP.rawValue)
        for (sidx, s) in ast.stmts.enumerated() {
            // left term
            ids.append(contentsOf: try encodeTerm(s.left, symtab: symtab))
            // relation
            ids.append(encodeRel(s.rel))
            // right term
            ids.append(contentsOf: try encodeTerm(s.right, symtab: symtab))
            if sidx < ast.stmts.count - 1 {
                ids.append(Vocab.COMMA.rawValue)
            }
        }
        ids.append(Vocab.RP.rawValue)
        ids.append(Vocab.BODY_END.rawValue)
        return ids
    }

    private static func encodeRel(_ r: Rel) -> Int {
        switch r {
        case .lt: return Vocab.LT.rawValue
        case .le: return Vocab.LE.rawValue
        case .eq: return Vocab.EQ.rawValue
        case .ne: return Vocab.NE.rawValue
        case .gt: return Vocab.GT.rawValue
        case .ge: return Vocab.GE.rawValue
        }
    }

    private static func encodeTerm(_ t: TermAST, symtab: SymbolTable) throws -> [Int] {
        switch t {
        case .ident(let name):
            guard let idx = symtab.nameToIndex[name] else { throw TextTokenizerError.internalError("Missing index for ident \(name)") }
            return [VocabRanges.IDX_BOS, VocabUtil.idxID(idx), VocabRanges.IDX_END]
        case .number(let s):
            var out: [Int] = [Vocab.NUM_BOS.rawValue]
            var chars = Array(s)
            var pos = 0
            if pos < chars.count && (chars[pos] == "+" || chars[pos] == "-") {
                out.append(chars[pos] == "+" ? Vocab.SIGN_PLUS.rawValue : Vocab.SIGN_MINUS.rawValue)
                pos += 1
            }
            while pos < chars.count {
                let c = chars[pos]
                guard let d = c.wholeNumberValue else { throw TextTokenizerError.internalError("Non-digit in number: \(s)") }
                out.append(VocabUtil.digitID(d))
                pos += 1
            }
            out.append(Vocab.NUM_END.rawValue)
            return out
        }
    }
}