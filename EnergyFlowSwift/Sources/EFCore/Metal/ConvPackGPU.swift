import Foundation

public enum ConvPackGPU {
    public static func packWToCol(W: Tensor, Cout: Int, Cin: Int, K: Int) -> Tensor {
        do {
            return try GPU.blocking(label: "ConvPackGPU.packWToCol") { actor in
                try await actor.packWToCol(W: W, Cout: Cout, Cin: Cin, K: K)
            }
        } catch {
            fatalError("ConvPackGPU.packWToCol failed: \(error)")
        }
    }

    public static func unpackDWCol(dWcol: Tensor, Cout: Int, Cin: Int, K: Int) -> Tensor {
        do {
            return try GPU.blocking(label: "ConvPackGPU.unpackDWCol") { actor in
                try await actor.unpackDWCol(dWcol: dWcol, Cout: Cout, Cin: Cin, K: K)
            }
        } catch {
            fatalError("ConvPackGPU.unpackDWCol failed: \(error)")
        }
    }
}
