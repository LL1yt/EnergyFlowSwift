import Foundation

enum LexToken: Equatable {
    case lparen
    case rparen
    case comma
    case rel(Rel)
    case ident(String)
    case number(String) // canonical string, may include leading sign
}

enum Rel: Equatable {
    case lt, le, eq, ne, gt, ge
}

struct Lexer {
    static func lex(_ input: String) throws -> [LexToken] {
        var tokens: [LexToken] = []
        let scalars = Array(input.unicodeScalars)
        var i = 0

        func peek(_ k: Int = 0) -> UnicodeScalar? { (i + k) < scalars.count ? scalars[i + k] : nil }
        func advance(_ n: Int = 1) { i += n }
        func isIdentStart(_ u: UnicodeScalar) -> Bool {
            return CharacterSet.letters.contains(u) || u == "_"
        }
        func isIdentCont(_ u: UnicodeScalar) -> Bool {
            return isIdentStart(u) || CharacterSet.decimalDigits.contains(u)
        }

        while let u = peek() {
            // Skip whitespace
            if CharacterSet.whitespacesAndNewlines.contains(u) {
                advance(); continue
            }
            // Delimiters
            if u == "(" { tokens.append(.lparen); advance(); continue }
            if u == ")" { tokens.append(.rparen); advance(); continue }
            if u == "," { tokens.append(.comma); advance(); continue }

            // Relations (two-char first)
            if u == "<" {
                if peek(1) == "=" { tokens.append(.rel(.le)); advance(2); continue }
                tokens.append(.rel(.lt)); advance(); continue
            }
            if u == ">" {
                if peek(1) == "=" { tokens.append(.rel(.ge)); advance(2); continue }
                tokens.append(.rel(.gt)); advance(); continue
            }
            if u == "!" && peek(1) == "=" { tokens.append(.rel(.ne)); advance(2); continue }
            if u == "=" { tokens.append(.rel(.eq)); advance(); continue }

            // Number: [+-]?[0-9]+
            if u == "+" || u == "-" || CharacterSet.decimalDigits.contains(u) {
                var j = i
                var hasSign = false
                if u == "+" || u == "-" { hasSign = true; j += 1 }
                var k = j
                var hadDigit = false
                while let d = (k < scalars.count ? scalars[k] : nil), CharacterSet.decimalDigits.contains(d) {
                    hadDigit = true; k += 1
                }
                if hadDigit {
                    let str = String(String.UnicodeScalarView(scalars[i..<k]))
                    tokens.append(.number(str))
                    i = k
                    continue
                }
                // fallthrough if not actually a number (e.g., "+x"): treat as ident
            }

            // Identifier
            if isIdentStart(u) {
                var k = i + 1
                while let c = (k < scalars.count ? scalars[k] : nil), isIdentCont(c) { k += 1 }
                let name = String(String.UnicodeScalarView(scalars[i..<k]))
                tokens.append(.ident(name))
                i = k
                continue
            }

            throw TextTokenizerError.invalidSyntax("Unexpected character: \(u)")
        }

        return tokens
    }
}