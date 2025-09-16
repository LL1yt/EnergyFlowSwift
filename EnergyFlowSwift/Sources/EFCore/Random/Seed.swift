import Foundation

// MARK: - Deterministic seed derivation
// Derive a stable sub-seed from a base seed and a label.
// We keep it simple and reproducible without crypto.
public enum Seed {
    public static func derive(_ base: UInt64, label: String) -> UInt64 {
        // FNV-1a like mixing
        var hash: UInt64 = 1469598103934665603 // offset basis
        let prime: UInt64 = 1099511628211
        func mix(_ byte: UInt8) {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        // mix base
        withUnsafeBytes(of: base) { raw in
            for b in raw { mix(b) }
        }
        // mix label bytes
        for b in label.utf8 { mix(b) }
        return hash
    }
}
