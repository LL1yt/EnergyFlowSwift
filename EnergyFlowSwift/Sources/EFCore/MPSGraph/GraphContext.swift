import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - Minimal MPSGraph context and helpers
// This provides a shared device/queue and a way to run small graphs.
// We start with float32 for numerical parity with the CPU reference.

public final class MPSGContext: @unchecked Sendable {
    public static let shared = MPSGContext()
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let graph: MPSGraph

    private init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("No Metal device available for MPSGraph")
        }
        self.device = dev
        guard let q = dev.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.commandQueue = q
        self.graph = MPSGraph()
        // Prefer default options; we can tweak later for FP16/profiling
    }
}

public enum MPSGError: Error, CustomStringConvertible {
    case contextUnavailable
    case commandBufferFailed

    public var description: String {
        switch self {
        case .contextUnavailable: return "MPSGraph context unavailable (no Metal device)"
        case .commandBufferFailed: return "Metal command buffer failed"
        }
    }
}
