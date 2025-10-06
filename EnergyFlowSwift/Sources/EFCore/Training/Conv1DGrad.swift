import Foundation

// CPU reference gradients for causal 1D convolution with dilation.
// Shapes: X [B,L,Cin], W [Cout,Cin,K], dY [B,L,Cout]
// Causality: forward uses x[b, t - k*d, i] with zero padding for t-k*d<0.

public struct Conv1DGrad {
    public static func backward(X: Tensor, W: Tensor, dY: Tensor, dilation: Int) -> (dW: Tensor, dB: Tensor, dX: Tensor) {
        precondition(X.shape.count == 3 && W.shape.count == 3 && dY.shape.count == 3)
        let B = X.shape[0]
        let L = X.shape[1]
        let Cin = X.shape[2]
        let Cout = W.shape[0]
        precondition(W.shape[1] == Cin)
        let K = W.shape[2]
        precondition(dY.shape[0] == B && dY.shape[1] == L && dY.shape[2] == Cout)
        let d = max(1, dilation)
        var dW = Tensor.zeros([Cout, Cin, K])
        var dB = Tensor.zeros([Cout])
        var dX = Tensor.zeros([B, L, Cin])
        // dB
        for b in 0..<B {
            for t in 0..<L {
                let dyBase = (b * L + t) * Cout
                for o in 0..<Cout { dB.data[o] += dY.data[dyBase + o] }
            }
        }
        // dW and dX
        for b in 0..<B {
            for t in 0..<L {
                let dyBase = (b * L + t) * Cout
                for o in 0..<Cout {
                    let dy = dY.data[dyBase + o]
                    if dy == 0 { continue }
                    // dW[o,i,k]
                    for k in 0..<K {
                        let ti = t - k * d
                        if ti >= 0 {
                            let xBase = (b * L + ti) * Cin
                            for i in 0..<Cin {
                                dW.data[(o * Cin + i) * K + k] += dy * X.data[xBase + i]
                            }
                        }
                    }
                }
                // dX[b,t,i] = sum_o sum_k dY[b,t+k*d,o] * W[o,i,k]
                for i in 0..<Cin {
                    var acc: Float = 0
                    for o in 0..<Cout {
                        for k in 0..<K {
                            let tp = t + k * d
                            if tp >= L { continue }
                            let dy = dY.data[(b * L + tp) * Cout + o]
                            let w = W.data[(o * Cin + i) * K + k]
                            acc += dy * w
                        }
                    }
                    dX.data[(b * L + t) * Cin + i] = acc
                }
            }
        }
        return (dW, dB, dX)
    }
}