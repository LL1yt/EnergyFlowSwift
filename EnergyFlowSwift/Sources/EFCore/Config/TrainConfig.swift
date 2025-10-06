import Foundation

// Centralized training configuration for EFTrain.
// Supports JSON file loading with optional fields, so partial configs are fine.
// Typical usage:
// let cfg = TrainConfig.load(path: "Configs/train_debug.json")
// Then overlay environment and CLI flags in EFTrain.

public struct TrainConfig: Codable {
    // Data
    public var dataPath: String?
    public var batchSize: Int?
    public var maxLength: Int?
    public var maxBatches: Int?
    public var epochs: Int?

    // Optim
    public var lr: Float?
    public var weightDecay: Float?
    public var warmupSteps: Int?
    public var cosineDecaySteps: Int?
    public var minLR: Float?
    public var clipNorm: Float?
    public var accumSteps: Int?

    // KD loss
    public var alphaCos: Float?
    public var betaMSE: Float?

    // Batching
    public var microBatch: Int?
    public var valFraction: Float?

    // Checkpoints/state
    public var saveCheckpoint: String?
    public var loadCheckpoint: String?
    public var saveOptState: String?
    public var loadOptState: String?

    // Unfreeze controls
    public var unfreezeLastTCN: Bool?

    public init() {}

    public static func load(path: String) throws -> TrainConfig {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        let dec = JSONDecoder()
        return try dec.decode(TrainConfig.self, from: data)
    }
}

public extension TrainConfig {
    // Merge b over a: if b.field != nil -> use it; else keep a.field
    func merged(over base: TrainConfig) -> TrainConfig {
        var out = base
        // Data
        if let v = dataPath { out.dataPath = v }
        if let v = batchSize { out.batchSize = v }
        if let v = maxLength { out.maxLength = v }
        if let v = maxBatches { out.maxBatches = v }
        if let v = epochs { out.epochs = v }
        // Optim
        if let v = lr { out.lr = v }
        if let v = weightDecay { out.weightDecay = v }
        if let v = warmupSteps { out.warmupSteps = v }
        if let v = cosineDecaySteps { out.cosineDecaySteps = v }
        if let v = minLR { out.minLR = v }
        if let v = clipNorm { out.clipNorm = v }
        if let v = accumSteps { out.accumSteps = v }
        // KD
        if let v = alphaCos { out.alphaCos = v }
        if let v = betaMSE { out.betaMSE = v }
        // Batching
        if let v = microBatch { out.microBatch = v }
        if let v = valFraction { out.valFraction = v }
        // Checkpoints/state
        if let v = saveCheckpoint { out.saveCheckpoint = v }
        if let v = loadCheckpoint { out.loadCheckpoint = v }
        if let v = saveOptState { out.saveOptState = v }
        if let v = loadOptState { out.loadOptState = v }
        if let v = unfreezeLastTCN { out.unfreezeLastTCN = v }
        return out
    }
}

