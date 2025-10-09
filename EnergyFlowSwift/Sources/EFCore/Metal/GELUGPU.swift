import Foundation

public enum GELUGPU {
    public static func forward(_ x: Tensor) -> Tensor {
        do {
            return try GPU.blocking(label: "GELUGPU.forward") { actor in
                try await actor.geluForward(x: x)
            }
        } catch {
            fatalError("GELUGPU.forward failed: \(error)")
        }
    }

    public static func backward(x: Tensor, dY: Tensor) -> Tensor {
        do {
            return try GPU.blocking(label: "GELUGPU.backward") { actor in
                try await actor.geluBackward(x: x, dY: dY)
            }
        } catch {
            fatalError("GELUGPU.backward failed: \(error)")
        }
    }
}
