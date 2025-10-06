import Foundation
import EFCore

// Projection-only checkpoint I/O utilities for EFTrain
// Binary format: magic "EFCK1" + dims + weights/bias

public func saveProjectionCheckpoint(path: String, weight: Tensor, bias: Tensor?) {
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    var data = Data()
    data.append(contentsOf: [0x45, 0x46, 0x43, 0x4B, 0x31]) // "EFCK1"
    func putU32(_ v: UInt32) { var le = v.littleEndian; withUnsafeBytes(of: &le) { data.append(contentsOf: $0) } }
    func putF32(_ v: Float) { var le = v.bitPattern.littleEndian; withUnsafeBytes(of: &le) { data.append(contentsOf: $0) } }
    putU32(UInt32(weight.shape[0]))
    putU32(UInt32(weight.shape[1]))
    putU32(UInt32(bias != nil ? 1 : 0))
    for v in weight.data { putF32(v) }
    if let b = bias { for v in b.data { putF32(v) } }
    try? data.write(to: url)
}

public func loadProjectionCheckpoint(path: String) -> (Tensor, Tensor?)? {
    let url = URL(fileURLWithPath: path)
    guard let data = try? Data(contentsOf: url) else { return nil }
    var idx = 0
    func getU8() -> UInt8 { defer { idx += 1 }; return data[idx] }
    guard idx + 5 <= data.count else { return nil }
    if getU8() != 0x45 || getU8() != 0x46 || getU8() != 0x43 || getU8() != 0x4B || getU8() != 0x31 { return nil }
    func getU32() -> UInt32 { defer { idx += 4 }; return data[idx..<(idx+4)].withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian }
    func getF32() -> Float { defer { idx += 4 }; let u = data[idx..<(idx+4)].withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian; return Float(bitPattern: u) }
    let out = Int(getU32())
    let inf = Int(getU32())
    let hasB = getU32() != 0
    let wCount = out * inf
    if idx + 4 * wCount > data.count { return nil }
    var wHost = [Float](repeating: 0, count: wCount)
    for i in 0..<wCount { wHost[i] = getF32() }
    let weight = Tensor(shape: [out, inf], data: wHost)
    var bias: Tensor? = nil
    if hasB {
        if idx + 4 * out > data.count { return nil }
        var bHost = [Float](repeating: 0, count: out)
        for i in 0..<out { bHost[i] = getF32() }
        bias = Tensor(shape: [out], data: bHost)
    }
    return (weight, bias)
}
