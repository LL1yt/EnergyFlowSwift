import Foundation
import MetalPerformanceShadersGraph
import Metal

// Prototype: MPSGraph Executable cache for static-shape block: LayerNorm -> GELU -> MatMul(+bias)
// Shapes (static during executable lifetime):
// - X:     [N, D]
// - gamma: [D]
// - beta:  [D]
// - W:     [Out, D]
// - b:     [Out] (optional)
// Output:  [N, Out]
// Notes:
// - We cache the compiled executable and persistent MPSNDArray-backed feeds to avoid re-allocation.
// - CPU<->GPU copies happen by writing into the MPSNDArray storage each call.

public final class LNGeLUGemmCache {
    public struct Key: Hashable {
        public let N: Int
        public let D: Int
        public let Out: Int
        public let hasBias: Bool
        public let eps: Float
        public init(N: Int, D: Int, Out: Int, hasBias: Bool, eps: Float) {
            self.N = N; self.D = D; self.Out = Out; self.hasBias = hasBias; self.eps = eps
        }
    }

    private struct Record {
        let graph: MPSGraph
        let executable: MPSGraphExecutable
        let xPH: MPSGraphTensor
        let gammaPH: MPSGraphTensor
        let betaPH: MPSGraphTensor
        let wPH: MPSGraphTensor
        let bPH: MPSGraphTensor?
        let yT: MPSGraphTensor
        let xArr: MPSNDArray
        let gammaArr: MPSNDArray
        let betaArr: MPSNDArray
        let wArr: MPSNDArray
        let bArr: MPSNDArray?
        let yArr: MPSNDArray
        let xData: MPSGraphTensorData
        let gammaData: MPSGraphTensorData
        let betaData: MPSGraphTensorData
        let wData: MPSGraphTensorData
        let bData: MPSGraphTensorData?
    }

    nonisolated(unsafe) public static let shared = LNGeLUGemmCache()
    private var cache: [Key: Record] = [:]

    private func ndarray(device: MTLDevice, shape: [NSNumber]) -> MPSNDArray {
        let desc = MPSNDArrayDescriptor(dataType: .float32, shape: shape)
        return MPSNDArray(device: device, descriptor: desc)
    }

    private func buildIfNeeded(key: Key) throws -> Record {
        if let r = cache[key] { return r }
        let ctx = MPSGContext.shared
        let device = ctx.device
        let graph = MPSGraph()
        let N = key.N, D = key.D, Out = key.Out

        // Placeholders
        func ph(_ shape: [NSNumber], _ name: String) -> MPSGraphTensor {
            graph.placeholder(shape: shape, dataType: .float32, name: name)
        }
        let xPH = ph([NSNumber(value: N), NSNumber(value: D)], "x")
        let gammaPH = ph([NSNumber(value: D)], "gamma")
        let betaPH = ph([NSNumber(value: D)], "beta")
        let wPH = ph([NSNumber(value: Out), NSNumber(value: D)], "w")
        let bPH = key.hasBias ? ph([NSNumber(value: Out)], "b") : nil

        // LayerNorm across last dimension D
        // mean: [N,1]
        let mean = graph.mean(of: xPH, axes: [NSNumber(value: 1)], name: nil)
        // var: E[(x-mean)^2]
        let xMinusMean = graph.subtraction(xPH, mean, name: nil)
        let sq = graph.multiplication(xMinusMean, xMinusMean, name: nil)
        let varT = graph.mean(of: sq, axes: [NSNumber(value: 1)], name: nil)
        let epsT = graph.constant(Double(key.eps), dataType: .float32)
        let invStd = graph.reciprocalSquareRoot(graph.addition(varT, epsT, name: nil), name: nil)
        let xhat = graph.multiplication(xMinusMean, invStd, name: nil)
        // scale+shift
        // broadcast gamma, beta to [N,D]
        let gammaB = graph.reshape(gammaPH, shape: [1, NSNumber(value: D)], name: nil)
        let betaB = graph.reshape(betaPH, shape: [1, NSNumber(value: D)], name: nil)
        let yLN = graph.addition(graph.multiplication(xhat, gammaB, name: nil), betaB, name: nil)

        // GELU approx: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        let c = graph.constant(0.7978845608028654, dataType: .float32) // sqrt(2/pi)
        let a = graph.constant(0.044715, dataType: .float32)
        let x2 = graph.multiplication(yLN, yLN, name: nil)
        let x3 = graph.multiplication(x2, yLN, name: nil)
        let inner = graph.addition(yLN, graph.multiplication(a, x3, name: nil), name: nil)
        let u = graph.multiplication(c, inner, name: nil)
        let tanhU = graph.tanh(with: u, name: nil)
        let one = graph.constant(1.0, dataType: .float32)
        let half = graph.constant(0.5, dataType: .float32)
        let gelu = graph.multiplication(half, graph.multiplication(yLN, graph.addition(one, tanhU, name: nil), name: nil), name: nil)

        // Matmul Y = gelu @ W^T + b
        // Need W^T: [D,Out]
        let wT = graph.transpose(wPH, permutation: [NSNumber(value: 1), NSNumber(value: 0)], name: nil)
        let mm = graph.matrixMultiplication(primary: gelu, secondary: wT, name: nil)
        let y: MPSGraphTensor
        if let bPH = bPH {
            let bB = graph.reshape(bPH, shape: [1, NSNumber(value: Out)], name: nil)
            y = graph.addition(mm, bB, name: nil)
        } else {
            y = mm
        }

        // Compile executable
        var feeds: [MPSGraphTensor: MPSGraphShapedType] = [:]
        feeds[xPH] = MPSGraphShapedType(shape: [NSNumber(value: N), NSNumber(value: D)], dataType: .float32)
        feeds[gammaPH] = MPSGraphShapedType(shape: [NSNumber(value: D)], dataType: .float32)
        feeds[betaPH] = MPSGraphShapedType(shape: [NSNumber(value: D)], dataType: .float32)
        feeds[wPH] = MPSGraphShapedType(shape: [NSNumber(value: Out), NSNumber(value: D)], dataType: .float32)
        if let bPH = bPH { feeds[bPH] = MPSGraphShapedType(shape: [NSNumber(value: Out)], dataType: .float32) }
        let executable = graph.compile(with: MPSGraphDevice(mtlDevice: ctx.device), feeds: feeds, targetTensors: [y], targetOperations: nil, compilationDescriptor: nil)

        // Persistent NDArrays
        let xArr = ndarray(device: device, shape: [NSNumber(value: N), NSNumber(value: D)])
        let gammaArr = ndarray(device: device, shape: [NSNumber(value: D)])
        let betaArr = ndarray(device: device, shape: [NSNumber(value: D)])
        let wArr = ndarray(device: device, shape: [NSNumber(value: Out), NSNumber(value: D)])
        let bArr = bPH != nil ? ndarray(device: device, shape: [NSNumber(value: Out)]) : nil
        let yArr = ndarray(device: device, shape: [NSNumber(value: N), NSNumber(value: Out)])

        let rec = Record(
            graph: graph,
            executable: executable,
            xPH: xPH, gammaPH: gammaPH, betaPH: betaPH, wPH: wPH, bPH: bPH, yT: y,
            xArr: xArr, gammaArr: gammaArr, betaArr: betaArr, wArr: wArr, bArr: bArr, yArr: yArr,
            xData: MPSGraphTensorData(xArr), gammaData: MPSGraphTensorData(gammaArr), betaData: MPSGraphTensorData(betaArr), wData: MPSGraphTensorData(wArr), bData: bArr != nil ? MPSGraphTensorData(bArr!) : nil
        )
        cache[key] = rec
        return rec
    }

    // Run forward: x,gamma,beta,W,(b) => y
    public func runForward(x: Tensor, gamma: Tensor, beta: Tensor, W: Tensor, bias: Tensor?, eps: Float = 1e-5) throws -> Tensor {
        precondition(x.shape.count == 2, "x must be [N,D]")
        precondition(gamma.shape == [x.shape[1]] && beta.shape == [x.shape[1]], "gamma/beta shape mismatch")
        precondition(W.shape.count == 2 && W.shape[1] == x.shape[1], "W shape mismatch")
        if let b = bias { precondition(b.shape == [W.shape[0]], "bias shape mismatch") }
        let N = x.shape[0], D = x.shape[1], Out = W.shape[0]
        let key = Key(N: N, D: D, Out: Out, hasBias: bias != nil, eps: eps)
        let rec = try buildIfNeeded(key: key)
        // Write inputs
        x.data.withUnsafeBytes { raw in if let base = raw.baseAddress { rec.xArr.writeBytes(UnsafeMutableRawPointer(mutating: base), strideBytes: nil) } }
        gamma.data.withUnsafeBytes { raw in if let base = raw.baseAddress { rec.gammaArr.writeBytes(UnsafeMutableRawPointer(mutating: base), strideBytes: nil) } }
        beta.data.withUnsafeBytes { raw in if let base = raw.baseAddress { rec.betaArr.writeBytes(UnsafeMutableRawPointer(mutating: base), strideBytes: nil) } }
        W.data.withUnsafeBytes { raw in if let base = raw.baseAddress { rec.wArr.writeBytes(UnsafeMutableRawPointer(mutating: base), strideBytes: nil) } }
        if let b = bias, let bArr = rec.bArr { b.data.withUnsafeBytes { raw in if let base = raw.baseAddress { bArr.writeBytes(UnsafeMutableRawPointer(mutating: base), strideBytes: nil) } } }
        // Execute
        // Inputs array must follow the same order as feeds in compile
        var inputsArray: [MPSGraphTensorData] = [rec.xData, rec.gammaData, rec.betaData, rec.wData]
        if let bData = rec.bData { inputsArray.append(bData) }
        let resultsArray = rec.executable.run(with: MPSGContext.shared.commandQueue, inputs: inputsArray, results: nil, executionDescriptor: nil)
        guard let yTD = resultsArray.first else { throw MPSGError.commandBufferFailed }
        var host = [Float](repeating: 0, count: N * Out)
        host.withUnsafeMutableBytes { raw in if let base = raw.baseAddress { yTD.mpsndarray().readBytes(base, strideBytes: nil) } }
        return Tensor(shape: [N, Out], data: host)
    }
}
