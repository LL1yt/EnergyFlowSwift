import Foundation
import EFCore

// MARK: - Gradient helpers (CPU reference)
// For now we implement MSE gradients for GraphLinear only (encoder projection).

// dL/dY for L_mse = mean_{i in B, d in D} (Y - T)^2
// => dY = 2/(B*D) * (Y - T)
public func dY_MSEMean(y: Tensor, target: Tensor) -> Tensor {
    precondition(y.shape.count == 2 && target.shape.count == 2 && y.shape == target.shape)
    let B = y.shape[0]
    let D = y.shape[1]
    var out = Tensor.zeros([B, D])
    let scale: Float = 2.0 / Float(max(B * D, 1))
    for i in 0..<(B*D) {
        out.data[i] = scale * (y.data[i] - target.data[i])
    }
    return out
}

// Gradient of cosine loss component: d/dy of (1 - mean_i cosine(y_i, t_i))
// For each row i: c_i = (y·t) / (||y||·||t||); d/dy (1 - mean c) = -(1/B) * dc/dy
// dc/dy = t/(||y||·||t||) - (y·t) * y / (||y||^3 · ||t||)
public func dY_CosineMeanLoss(y: Tensor, target: Tensor, eps: Float = 1e-6) -> Tensor {
    precondition(y.shape.count == 2 && target.shape.count == 2 && y.shape == target.shape)
    let B = y.shape[0]
    let D = y.shape[1]
    var out = Tensor.zeros([B, D])
    let invB: Float = 1.0 / Float(max(B, 1))
    for bi in 0..<B {
        // norms and dot
        var dot: Float = 0
        var ny2: Float = 0
        var nt2: Float = 0
        let base = bi * D
        for d in 0..<D {
            let vy = y.data[base + d]
            let vt = target.data[base + d]
            dot += vy * vt
            ny2 += vy * vy
            nt2 += vt * vt
        }
        let ay = max(sqrt(ny2), eps)
        let at = max(sqrt(nt2), eps)
        let s1: Float = 1.0 / (ay * at)
        let s2: Float = dot / (ay * ay * ay * at)
        // grad = -(1/B) * ( t * s1 - y * s2 )
        for d in 0..<D {
            let vy = y.data[base + d]
            let vt = target.data[base + d]
            let g = -invB * (vt * s1 - vy * s2)
            out.data[base + d] = g
        }
    }
    return out
}

// Given Y = X * W^T + b, where X:[B,In], W:[Out,In], b:[Out], and dY:[B,Out]
// grads: dW:[Out,In] = dY^T * X; dB:[Out] = sum_b dY[b,:]
public func gradsGraphLinear(X: Tensor, dY: Tensor, outFeatures: Int, inFeatures: Int) -> (dW: Tensor, dB: Tensor) {
    precondition(X.shape.count == 2 && dY.shape.count == 2)
    precondition(X.shape[0] == dY.shape[0])
    precondition(X.shape[1] == inFeatures && dY.shape[1] == outFeatures)
    let B = X.shape[0]
    var dW = Tensor.zeros([outFeatures, inFeatures])
    var dB = Tensor.zeros([outFeatures])
    for b in 0..<B {
        let xBase = b * inFeatures
        let dyBase = b * outFeatures
        // dB += dY[b]
        for o in 0..<outFeatures {
            dB.data[o] += dY.data[dyBase + o]
        }
        // dW[o,i] += dY[b,o] * X[b,i]
        for o in 0..<outFeatures {
            let dyVal = dY.data[dyBase + o]
            if dyVal == 0 { continue }
            for i in 0..<inFeatures {
                dW.data[o * inFeatures + i] += dyVal * X.data[xBase + i]
            }
        }
    }
    return (dW, dB)
}