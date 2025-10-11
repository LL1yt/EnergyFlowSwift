import Foundation
import EFCore

// MARK: - KD Weights and Batch/Metric types

public struct KDWeights {
    public var alphaCos: Float = 1.0  // weight for (1 - cosine)
    public var betaMSE: Float = 0.5   // weight for MSE
}

public struct TrainBatch {
    public let inputIDs: [[Int]]
    public let attentionMask: [[Int]]
    public let teacher: [[Float]]      // [B][D]
}

public struct TrainMetrics {
    public let loss: Float
    public let meanMSE: Float
    public let meanCosine: Float
}

// MARK: - Trainer skeleton (forward-only for now)

public final class Trainer {
    public let kd: KDWeights
    public init(kd: KDWeights) { self.kd = kd }

    // Computes KD loss and metrics. No backprop yet.
    public func computeLossAndMetrics(encoder: TextToCubeEncoder,
                                      batch: TrainBatch,
                                      on gpu: GPUActor = GPU.shared) async throws -> TrainMetrics {
        let out = try await encoder.encodeTokens(inputIDs: batch.inputIDs,
                                                 attentionMask: batch.attentionMask,
                                                 on: gpu) // [B, D]
        let B = out.shape[0]
        let D = out.shape[1]
        precondition(!batch.teacher.isEmpty && batch.teacher.first?.count == D)
        // Convert teacher to Tensor
        var tHost = [Float](repeating: 0, count: B * D)
        for bi in 0..<B { for di in 0..<D { tHost[bi*D+di] = batch.teacher[bi][di] } }
        let t = Tensor(shape: [B, D], data: tHost)
        // Loss/metrics
        let mse = Losses.mseRowwise(out, t)
        let cos = Losses.cosineSimilarityRowwise(out, t)
        let loss = kd.alphaCos * (1 - cos.mean) + kd.betaMSE * mse.mean
        return TrainMetrics(loss: loss, meanMSE: mse.mean, meanCosine: cos.mean)
    }
}
