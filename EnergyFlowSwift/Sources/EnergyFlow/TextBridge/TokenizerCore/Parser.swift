import Foundation

struct BlockAST {
    var stmts: [Stmt]
}

enum TermAST {
    case ident(String)
    case number(String) // canonical string with optional sign
}

struct Stmt {
    var left: TermAST
    var rel: Rel
    var right: TermAST
}

struct Parser {
    static func parse(_ tokens: [LexToken]) throws -> BlockAST {
        var i = 0
        func peek() -> LexToken? { i < tokens.count ? tokens[i] : nil }
        func advance() { i += 1 }
        func expect(_ t: LexToken) throws {
            guard let p = peek(), p == t else { throw TextTokenizerError.invalidSyntax("Expected \(t), got \(String(describing: peek()))") }
            advance()
        }
        func parseTerm() throws -> TermAST {
            guard let p = peek() else { throw TextTokenizerError.invalidSyntax("Unexpected end in term") }
            switch p {
            case .ident(let s): advance(); return .ident(s)
            case .number(let s): advance(); return .number(s)
            default:
                throw TextTokenizerError.invalidSyntax("Expected term, got \(p)")
            }
        }
        func parseStmt() throws -> Stmt {
            let l = try parseTerm()
            guard let p = peek() else { throw TextTokenizerError.invalidSyntax("Expected relation") }
            guard case .rel(let r) = p else { throw TextTokenizerError.invalidSyntax("Expected relation, got \(p)") }
            advance()
            let rterm = try parseTerm()
            return Stmt(left: l, rel: r, right: rterm)
        }

        // Block: '(' stmt (',' stmt)* ')'
        guard let p0 = peek(), p0 == .lparen else { throw TextTokenizerError.invalidSyntax("Block must start with '('") }
        advance()
        var stmts: [Stmt] = []
        if let p = peek(), p != .rparen {
            let s = try parseStmt()
            stmts.append(s)
            while let p = peek(), p == .comma {
                advance()
                let s2 = try parseStmt()
                stmts.append(s2)
            }
        }
        try expect(.rparen)
        return BlockAST(stmts: stmts)
    }
}