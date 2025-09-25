import Foundation

public protocol TextTokenizer {
    func encode(_ text: String) throws -> TokenSequence
    func decode(_ seq: TokenSequence) throws -> String
}

public enum TextTokenizerError: Error, CustomStringConvertible {
    case invalidSyntax(String)
    case tooManyVariables(Int)
    case sequenceTooLong(Int)
    case headerMissing
    case bodyMissing
    case unknownIndex(Int)
    case internalError(String)

    public var description: String {
        switch self {
        case .invalidSyntax(let m): return "invalidSyntax: \(m)"
        case .tooManyVariables(let n): return "tooManyVariables: \(n) > 64"
        case .sequenceTooLong(let n): return "sequenceTooLong: \(n) > 256"
        case .headerMissing: return "headerMissing"
        case .bodyMissing: return "bodyMissing"
        case .unknownIndex(let i): return "unknownIndex: \(i)"
        case .internalError(let m): return "internalError: \(m)"
        }
    }
}