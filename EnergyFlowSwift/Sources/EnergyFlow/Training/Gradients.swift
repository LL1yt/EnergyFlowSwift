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

// MARK: - GELU backward (approx derivative matching Activations.gelu)
// Inputs: x (same shape as upstream), upstream gradient g -> returns dx
public func dGELU(x: Tensor, upstream g: Tensor) -> Tensor {
    precondition(x.shape == g.shape)
    var out = Tensor.zeros(x.shape)
    let c: Float = 0.7978845608028654 // sqrt(2/pi)
    for i in 0..<x.count {
        let xi = x.data[i]
        let x3 = xi * xi * xi
        let u = c * (xi + 0.044715 * x3)
        let tanh_u = Float(Darwin.tanh(Double(u)))
        let sech2 = 1 - tanh_u * tanh_u
        // d gelu / dx
        let d1 = 0.5 * (1 + tanh_u)
        let d2 = 0.5 * xi * sech2 * c * (1 + 3 * 0.044715 * xi * xi)
        let dgelu_dx = d1 + d2
        out.data[i] = g.data[i] * dgelu_dx
    }
    return out
}

// MARK: - LayerNorm backward (row-wise LN used in TCN)
// x: [N, D], upstream g: [N, D], gamma: [D]
// returns (dx [N,D], dGamma [D], dBeta [D])
public func layerNormBackward(x: Tensor, upstream g: Tensor, gamma: Tensor, eps: Float = 1e-5) -> (Tensor, Tensor, Tensor) {
    precondition(x.shape.count == 2 && g.shape == x.shape)
    let N = x.shape[0]
    let D = x.shape[1]
    precondition(gamma.shape.count == 1 && gamma.shape[0] == D)
    var dx = Tensor.zeros([N, D])
    var dGamma = Tensor.zeros([D])
    var dBeta = Tensor.zeros([D])
    for n in 0..<N {
        let base = n * D
        // mean
        var mean: Float = 0
        for j in 0..<D { mean += x.data[base + j] }
        mean /= Float(D)
        // variance
        var varAcc: Float = 0
        for j in 0..<D { let d = x.data[base + j] - mean; varAcc += d*d }
        let varVal = varAcc / Float(D)
        let invStd = 1.0 / Float(sqrt(Double(varVal + eps)))
        // xhat and sums
        var sumG: Float = 0
        var sumGXhat: Float = 0
        var xhat = [Float](repeating: 0, count: D)
        for j in 0..<D {
            let xh = (x.data[base + j] - mean) * invStd
            xhat[j] = xh
            let gj = g.data[base + j]
            sumG += gj * gamma.data[j]
            sumGXhat += (gj * gamma.data[j]) * xh
        }
        for j in 0..<D {
            // dx formula aggregated
            let gj = g.data[base + j]
            let ghat = gj * gamma.data[j]
            let term = Float(D) * ghat - sumG - xhat[j] * sumGXhat
            dx.data[base + j] = (invStd / Float(D)) * term
            dGamma.data[j] += gj * xhat[j]
            dBeta.data[j] += gj
        }
    }
    return (dx, dGamma, dBeta)
}
