import Foundation

// Pad or truncate mask rows to fixed length
public func padOrTruncateMask(_ mask: [[Int]], _ len: Int) -> [[Int]] {
    var out: [[Int]] = []
    out.reserveCapacity(mask.count)
    for var row in mask {
        if row.count > len {
            row = Array(row.prefix(len))
        } else if row.count < len {
            row.append(contentsOf: Array(repeating: 0, count: len - row.count))
        }
        out.append(row)
    }
    return out
}
