import Foundation
import EFCore

// Overlay TrainConfig values (from JSON/env) into TrainArgs used by EFTrain
// Kept inside EFTrain target to avoid cross-target dependency on TrainArgs type.
extension TrainConfig {
    func apply(to args: inout TrainArgs) {
        if let v = dataPath { args.dataPath = v }
        if let v = batchSize { args.batchSize = v }
        if let v = maxLength { args.maxLength = v }
        if let v = maxBatches { args.maxBatches = v }
        if let v = epochs { args.epochs = v }
        if let v = lr { args.lr = v }
        if let v = weightDecay { args.weightDecay = v }
        if let v = warmupSteps { args.warmupSteps = max(0, v) }
        if let v = cosineDecaySteps { args.cosineDecaySteps = max(0, v) }
        if let v = minLR { args.minLR = max(0, v) }
        if let v = clipNorm { args.clipNorm = max(0, v) }
        if let v = accumSteps { args.accumSteps = max(1, v) }
        if let v = alphaCos { args.alphaCos = v }
        if let v = betaMSE { args.betaMSE = v }
        if let v = microBatch { args.microBatch = v }
        if let v = valFraction { args.valFraction = v }
        if let v = saveCheckpoint { args.saveCheckpoint = v }
        if let v = loadCheckpoint { args.loadCheckpoint = v }
        if let v = saveOptState { args.saveOptState = v }
        if let v = loadOptState { args.loadOptState = v }
    }
}
