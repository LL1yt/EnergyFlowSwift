import Foundation

struct SymbolTable {
    private(set) var nameToIndex: [String: Int] = [:]
    private(set) var indexToName: [String] = []

    mutating func index(for name: String) throws -> Int {
        if let idx = nameToIndex[name] { return idx }
        if indexToName.count >= 64 { throw TextTokenizerError.tooManyVariables(indexToName.count + 1) }
        let idx = indexToName.count
        nameToIndex[name] = idx
        indexToName.append(name)
        return idx
    }
}