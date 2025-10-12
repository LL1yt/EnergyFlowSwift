import Foundation

public enum GPUElementType: Int, Sendable {
    case float32
    case float16
}

public struct GPUTensorHandle: Sendable {
    public let id: UUID
    public let shape: [Int]
    public let rows: Int
    public let cols: Int
    public let rowBytes: Int
    public let elemType: GPUElementType
    public let label: String
    public let epoch: UInt64
}