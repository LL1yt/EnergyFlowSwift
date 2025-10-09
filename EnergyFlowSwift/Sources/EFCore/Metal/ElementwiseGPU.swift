import Foundation

public enum ElementwiseGPU {
    public static func residualAdd(y: Tensor, x: Tensor) -> Tensor {
        do {
            return try GPU.blocking(label: "ElementwiseGPU.residualAdd") { actor in
                try await actor.residualAdd(y: y, x: x)
            }
        } catch {
            fatalError("ElementwiseGPU.residualAdd failed: \(error)")
        }
    }

    public static func addBroadcast2DInto3D(y: Tensor, addBD: Tensor, L: Int) -> Tensor {
        do {
            return try GPU.blocking(label: "ElementwiseGPU.addBroadcast2DInto3D") { actor in
                try await actor.addBroadcast2DInto3D(y: y, addBD: addBD, sequenceLength: L)
            }
        } catch {
            fatalError("ElementwiseGPU.addBroadcast2DInto3D failed: \(error)")
        }
    }

    public static func maskZero(y: Tensor, mask: [[Int]]) -> Tensor {
        do {
            return try GPU.blocking(label: "ElementwiseGPU.maskZero") { actor in
                try await actor.maskZero(y: y, mask: mask)
            }
        } catch {
            fatalError("ElementwiseGPU.maskZero failed: \(error)")
        }
    }

    public static func maskedMean(x: Tensor, mask: [[Int]]) -> Tensor {
        do {
            return try GPU.blocking(label: "ElementwiseGPU.maskedMean") { actor in
                try await actor.maskedMean(x: x, mask: mask)
            }
        } catch {
            fatalError("ElementwiseGPU.maskedMean failed: \(error)")
        }
    }

    public static func maskedMeanBackward(dPooled: Tensor, mask: [[Int]], seqLen: Int) -> Tensor {
        do {
            return try GPU.blocking(label: "ElementwiseGPU.maskedMeanBackward") { actor in
                try await actor.maskedMeanBackward(dPooled: dPooled, mask: mask, seqLen: seqLen)
            }
        } catch {
            fatalError("ElementwiseGPU.maskedMeanBackward failed: \(error)")
        }
    }
}
