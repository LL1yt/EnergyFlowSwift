import Foundation
import EnergyFlow
import EFCore

struct ResolvedTrainConfig {
    var dataPath: String
    var batchSize: Int
    var maxLength: Int
    var maxBatches: Int
    var epochs: Int
    var lr: Float
    var weightDecay: Float
    var alphaCos: Float   // weight for (1 - cosine)
    var betaMSE: Float    // weight for MSE
    var microBatch: Int
    var valFraction: Float
    var saveCheckpoint: String   // path to save best checkpoint
    var loadCheckpoint: String   // path to load checkpoint
    var accumSteps: Int
    // New training controls
    var warmupSteps: Int
    var cosineDecaySteps: Int
    var minLR: Float
    var clipNorm: Float
    var saveOptState: String
    var loadOptState: String
    // Unfreeze controls
    var unfreezeLastTCN: Bool
    // Central config
    var configPath: String
}

func loadResolvedConfig() -> ResolvedTrainConfig? {
    // Accept either: EFTrain [path.json] or EFTrain --config path.json; fallback to default path
    let argv = Array(CommandLine.arguments.dropFirst())
    var configPath: String? = nil
    var i = 0
    while i < argv.count {
        let tok = argv[i]
        if tok == "--config" {
            if i + 1 < argv.count { configPath = argv[i + 1] }
            break
        } else if tok.hasPrefix("--") {
            // unknown flag, skip its possible value conservatively
            i += 1
            continue
        } else {
            // first positional considered as config path
            configPath = tok
            break
        }
    }
    let path = configPath ?? "Configs/train_debug.json"
    guard let c = try? TrainConfig.load(path: path) else {
        let msg = "EFTrain error: cannot load config file at \(path). Please create and fill it (see EnergyFlowSwift/Configs/train_debug.json).\n"
        FileHandle.standardError.write(Data(msg.utf8))
        return nil
    }
    // Strict validation: require all fields in the config
    guard
        let dataPath = c.dataPath,
        let batchSize = c.batchSize,
        let maxLength = c.maxLength,
        let maxBatches = c.maxBatches,
        let epochs = c.epochs,
        let lr = c.lr,
        let weightDecay = c.weightDecay,
        let alphaCos = c.alphaCos,
        let betaMSE = c.betaMSE,
        let microBatch = c.microBatch,
        let valFraction = c.valFraction,
        let accumSteps = c.accumSteps,
        let warmupSteps = c.warmupSteps,
        let cosineDecaySteps = c.cosineDecaySteps,
        let minLR = c.minLR,
        let clipNorm = c.clipNorm,
        let saveOptState = c.saveOptState,
        let loadOptState = c.loadOptState,
        let unfreezeLastTCN = c.unfreezeLastTCN
    else {
        FileHandle.standardError.write(Data("EFTrain error: invalid config structure. Ensure all required fields are present. See sample: EnergyFlowSwift/Configs/train_debug.json\n".utf8))
        return nil
    }
    if dataPath.isEmpty {
        FileHandle.standardError.write(Data("EFTrain error: dataPath is empty in config file \(path).\n".utf8))
        return nil
    }
    return ResolvedTrainConfig(
        dataPath: dataPath,
        batchSize: batchSize,
        maxLength: maxLength,
        maxBatches: maxBatches,
        epochs: epochs,
        lr: lr,
        weightDecay: weightDecay,
        alphaCos: alphaCos,
        betaMSE: betaMSE,
        microBatch: microBatch,
        valFraction: valFraction,
        saveCheckpoint: c.saveCheckpoint ?? "",
        loadCheckpoint: c.loadCheckpoint ?? "",
        accumSteps: accumSteps,
        warmupSteps: warmupSteps,
        cosineDecaySteps: cosineDecaySteps,
        minLR: minLR,
        clipNorm: clipNorm,
        saveOptState: saveOptState,
        loadOptState: loadOptState,
        unfreezeLastTCN: unfreezeLastTCN,
        configPath: path
    )
}

func usage() {
    print("Usage: EFTrain [path/to/config.json]  (default: Configs/train_debug.json)")
}

func run() throws {
guard let args = loadResolvedConfig() else { usage(); return }
    var globalStep = 0
    let logger = Logger.shared
    logger.info("EFTrain start: data=\(args.dataPath) batchSize=\(args.batchSize) maxLen=\(args.maxLength) epochs=\(args.epochs) alphaCos=\(args.alphaCos) betaMSE=\(args.betaMSE)", category: Logger.Category.dataset)
    
    let ds = try SimpleJSONLDataset(path: args.dataPath)
// Configure encoder; set only what differs from defaults
    let modelCfg = TextToCubeEncoderConfig(
        maxLength: args.maxLength,
        outputDim: ds.embeddingDim
    )
    let enc = TextToCubeEncoder(modelConfig: modelCfg)
    
    // Load checkpoint if provided
    if !args.loadCheckpoint.isEmpty, let (wLoaded, bLoaded) = loadProjectionCheckpoint(path: args.loadCheckpoint) {
        if wLoaded.shape == [modelCfg.outputDim, modelCfg.hiddenDim] {
            enc.setProjParams(weight: wLoaded, bias: bLoaded)
            enc.invalidateProjectionCache()
            logger.info("loaded checkpoint: \(args.loadCheckpoint) W=\(wLoaded.prettyShape) B=\(bLoaded != nil)", category: Logger.Category.dataset)
        } else {
            logger.warn("checkpoint shape mismatch: expected [\(modelCfg.outputDim), \(modelCfg.hiddenDim)] got \(wLoaded.prettyShape) — skipping load", category: Logger.Category.dataset)
        }
    }
    
    // Optimizer: we update only projector weights/bias for now
    let opt = AdamW(lr: args.lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: args.weightDecay)
    // Optionally load optimizer state (projection-only)
    do {
        let (w0, b0) = enc.getProjParams()
        var counts: [Int] = [w0.count]
        if let b0t = b0 { counts.append(b0t.count) }
        if !args.loadOptState.isEmpty {
            let ok = opt.loadState(path: args.loadOptState, expectedParamCounts: counts)
            logger.info("load opt state: \(ok) from=\(args.loadOptState)", category: Logger.Category.training)
        }
    }
    
    // Split train/val
    let total = ds.count()
    let valCount = args.valFraction > 0 ? max(1, Int(Float(total) * args.valFraction)) : 0
    let trainCount = total - valCount
    var indices = Array(0..<total)
    // simple shuffle
    indices.shuffle()
    let valIdx = valCount > 0 ? Array(indices.prefix(valCount)) : []
    let trainIdx = Array(indices.suffix(trainCount))
    func samplesByIndex(_ idx: [Int]) -> [JSONLSample] { idx.map { ds.samples[$0] } }
    let trainSamples = samplesByIndex(trainIdx)
    let valSamples = samplesByIndex(valIdx)
    
    // Helper: pad or truncate mask to fixed length (aligns with encoder's pad/truncate)
    func padOrTruncateMask(_ mask: [[Int]], _ len: Int) -> [[Int]] {
        var out: [[Int]] = []
        out.reserveCapacity(mask.count)
        for var row in mask {
            if row.count > len {
                row = Array(row.prefix(len))
            } else if row.count < len {
                row.append(contentsOf: Array(repeating: 0, count: len - row.count))
            }
            out.append(row)
        }
        return out
    }
    
    var bestValCos: Double = -Double.greatestFiniteMagnitude
    
    for epoch in 0..<args.epochs {
        logger.info("epoch=\(epoch+1)/\(args.epochs)", category: Logger.Category.dataset)
        
        var seen = 0
        var totalMSE: Double = 0
        var totalCos: Double = 0
        // Iterate train samples in batches of batchSize, each split to micro-batches
        var ptr = 0
        epoch_loop: while ptr < trainSamples.count {
            let end = min(ptr + args.batchSize, trainSamples.count)
            let slice = Array(trainSamples[ptr..<end])
            ptr = end
            if args.maxBatches > 0 && (ptr / max(args.batchSize,1)) > args.maxBatches { break epoch_loop }
            // Decide mode per slice
            let hasTokens = slice.allSatisfy { $0.inputIDs != nil && $0.attentionMask != nil }
            if hasTokens {
                // Build token batch arrays
                var ids: [[Int]] = []
                var mask: [[Int]] = []
                var tgts: [[Float]] = []
                ids.reserveCapacity(slice.count)
                mask.reserveCapacity(slice.count)
                tgts.reserveCapacity(slice.count)
                for s in slice {
                    ids.append(s.inputIDs!)
                    mask.append(s.attentionMask!)
                    tgts.append(s.target)
                }
                // micro-batching
                let B = ids.count
                let micro = args.microBatch > 0 ? args.microBatch : B
                let numChunks = (B + micro - 1) / micro
                var offset = 0
                var stepCount = 0
                var accW: Tensor? = nil
                var accB: Tensor? = nil
                // Last TCN block accumulators
                var accLastW1: Tensor? = nil
                var accLastB1: Tensor? = nil
                var accLastW2: Tensor? = nil
                var accLastB2: Tensor? = nil
                var accGamma: Tensor? = nil
                var accBeta: Tensor? = nil
                for _ in 0..<numChunks {
                    let take = min(micro, B - offset)
                    let idsChunk = Array(ids[offset..<(offset+take)])
                    let maskChunk = Array(mask[offset..<(offset+take)])
                    let tgtChunk = Array(tgts[offset..<(offset+take)])
                    let pooled: Tensor
                    let out: Tensor
                    var lastCache: TextToCubeEncoder.LastTCNCache? = nil
                    var maskFixedLocal: [[Int]] = []
                    if args.unfreezeLastTCN {
                        let res = enc.forwardForTrainingWithLastBlockCache(inputIDs: idsChunk, attentionMask: maskChunk)
                        pooled = res.pooled
                        out = res.out
                        lastCache = res.cache
                        maskFixedLocal = res.maskFixed
                    } else {
                        let r = enc.forwardForTraining(inputIDs: idsChunk, attentionMask: maskChunk)
                        pooled = r.pooled
                        out = r.out
                    }
                    let b = out.shape[0]; let d = out.shape[1]
                    var tHost = [Float](repeating: 0, count: b*d)
                    for bi in 0..<b { for di in 0..<d { tHost[bi*d + di] = tgtChunk[bi][di] } }
                    let t = Tensor(shape: [b, d], data: tHost)
                    // Pre-update metrics
                    let mse = Losses.mseRowwise(out, t)
                    let cos = Losses.cosineSimilarityRowwise(out, t)
                    totalMSE += Double(mse.mean) * Double(b)
                    totalCos += Double(cos.mean) * Double(b)
                    seen += b
                    logger.debug(String(format: "train chunk: b=%d d=%d MSE=%.6f Cos=%.6f", b, d, mse.mean, cos.mean), category: Logger.Category.dataset)
                    // Backward combine
                    var dY = dY_MSEMean(y: out, target: t)
                    if args.alphaCos != 0 {
                        let dYcos = dY_CosineMeanLoss(y: out, target: t)
                        let B2 = dY.shape[0]; let D2 = dY.shape[1]
                        for idx in 0..<(B2*D2) { dY.data[idx] = args.betaMSE * dY.data[idx] + args.alphaCos * dYcos.data[idx] }
                    } else {
                        let scale = args.betaMSE
                        if scale != 1.0 { for i in 0..<dY.count { dY.data[i] *= scale } }
                    }
                    // Gradients for projector on GPU
                    let (dW, dB) = try enc.projectionGradientsGPU(X: pooled, dY: dY)
                    // Optional: upstream to encoder for future unfreeze (token-mode only)
                    do {
                        let dXin = try enc.projectionInputGradientsGPU(dY: dY)
                        let maskFixed = args.unfreezeLastTCN ? maskFixedLocal : padOrTruncateMask(maskChunk, modelCfg.maxLength)
                        let dEnc = enc.maskedMeanBackward(dPooled: dXin, mask: maskFixed, seqLen: modelCfg.maxLength)
                        // Log gradient norms for sanity
                        var norm2: Float = 0
                        for v in dEnc.data { norm2 += v*v }
                        let gnorm = sqrt(norm2)
                        logger.debug(String(format: "dEnc L2=%.6f", gnorm), category: Logger.Category.training)
                        // If unfreezing last TCN, compute grads for that block
                        if args.unfreezeLastTCN, let cache = lastCache {
                            // Apply mask to dEnc (zero masked positions)
                            var dOut = dEnc
                            let Bm = cache.xIn.shape[0]; let Lm = cache.xIn.shape[1]; let Dm = cache.xIn.shape[2]
                            for b2 in 0..<Bm {
                                for t2 in 0..<Lm {
                                    if maskFixed[b2][t2] == 0 {
                                        let base = (b2 * Lm + t2) * Dm
                                        for di2 in 0..<Dm { dOut.data[base + di2] = 0 }
                                    }
                                }
                            }
                            let params = enc.getLastBlockParams()
                            let grads = try lastTCNBackward(cache: cache, mask: maskFixed, dOut: dOut, modelCfg: modelCfg, params: LastTCNParams(w1: params.w1, b1: params.b1, w2: params.w2, b2: params.b2, gamma: params.gamma, beta: params.beta))
                            if accLastW1 == nil { accLastW1 = Tensor.zeros(params.w1.shape) }
                            if accLastB1 == nil { accLastB1 = params.b1 != nil ? Tensor.zeros([params.b1!.count]) : Tensor.zeros([0]) }
                            if accLastW2 == nil { accLastW2 = Tensor.zeros(params.w2.shape) }
                            if accLastB2 == nil { accLastB2 = params.b2 != nil ? Tensor.zeros([params.b2!.count]) : Tensor.zeros([0]) }
                            if accGamma == nil { accGamma = Tensor.zeros([params.gamma.count]) }
                            if accBeta == nil { accBeta = Tensor.zeros([params.beta.count]) }
                            for i2 in 0..<grads.dW1.count { accLastW1!.data[i2] += grads.dW1.data[i2] }
                            if let db1 = grads.dB1 { for i2 in 0..<db1.count { accLastB1!.data[i2] += db1.data[i2] } }
                            for i2 in 0..<grads.dW2.count { accLastW2!.data[i2] += grads.dW2.data[i2] }
                            if let db2 = grads.dB2 { for i2 in 0..<db2.count { accLastB2!.data[i2] += db2.data[i2] } }
                            for i2 in 0..<grads.dGamma.count { accGamma!.data[i2] += grads.dGamma.data[i2] }
                            for i2 in 0..<grads.dBeta.count { accBeta!.data[i2] += grads.dBeta.data[i2] }
                        }
                    } catch {
                        logger.warn("projection input gradients failed: \(error)", category: Logger.Category.training)
                    }
                    // Accumulate
                    if accW == nil { accW = Tensor.zeros(dW.shape) }
                    for i in 0..<dW.count { accW!.data[i] += dW.data[i] }
                    if accB == nil { accB = Tensor.zeros([dB.count]) }
                    for i in 0..<dB.count { accB!.data[i] += dB.data[i] }
                    stepCount += 1
                    // Step if reached accumSteps or last chunk
                    if stepCount % args.accumSteps == 0 || offset + take >= B {
                        // Average grads
                        let scale = 1.0 / Float(stepCount % args.accumSteps == 0 ? args.accumSteps : stepCount)
                        var gW = accW!
                        var gB = accB!
                        // LR schedule (warmup + cosine)
                        let lrNow = LRSchedulers.warmupCosine(baseLR: args.lr, minLR: args.minLR, warmupSteps: args.warmupSteps, decaySteps: args.cosineDecaySteps, step: globalStep)
                        logger.debug(String(format: "opt step=%d lr=%.6g", globalStep, lrNow), category: Logger.Category.training)
                        var lastPack: LastTCNGrads? = nil
                        if args.unfreezeLastTCN, let aW1 = accLastW1, let aW2 = accLastW2, let aG = accGamma, let aBt = accBeta {
                            let dB1 = (accLastB1 != nil && accLastB1!.count > 0) ? accLastB1 : nil
                            let dB2 = (accLastB2 != nil && accLastB2!.count > 0) ? accLastB2 : nil
                            lastPack = LastTCNGrads(dW1: aW1, dB1: dB1, dW2: aW2, dB2: dB2, dGamma: aG, dBeta: aBt)
                        }
                        optimizerStepProjectionAndLastBlock(enc: enc, opt: opt, inputs: OptimStepInputs(projGradW: gW, projGradB: gB, lastGrads: lastPack, lrNow: lrNow, scale: scale, clipNorm: args.clipNorm))
                        globalStep += 1
                        // Reset accumulators
                        accW = nil; accB = nil; stepCount = 0
                        if args.unfreezeLastTCN { accLastW1 = nil; accLastB1 = nil; accLastW2 = nil; accLastB2 = nil; accGamma = nil; accBeta = nil }
                        // Post-update metrics using project-only
                        let outPost = enc.projectOnly(pooled)
                        let msePost = Losses.mseRowwise(outPost, t)
                        let cosPost = Losses.cosineSimilarityRowwise(outPost, t)
                        logger.debug(String(format: "post-upd: b=%d d=%d MSE=%.6f Cos=%.6f", b, d, msePost.mean, cosPost.mean), category: Logger.Category.dataset)
                        // Reset accumulators
                        accW = nil; accB = nil; stepCount = 0
                    }
                    offset += take
                }
            } else {
                // text-mode path
                let texts = slice.map { $0.text ?? "" }
                let tgts = slice.map { $0.target }
                let B = texts.count
                let micro = args.microBatch > 0 ? args.microBatch : B
                let numChunks = (B + micro - 1) / micro
                var offset = 0
                var stepCount = 0
                var accW: Tensor? = nil
                var accB: Tensor? = nil
                // Last TCN block accumulators (mirror token-mode)
                var accLastW1: Tensor? = nil
                var accLastB1: Tensor? = nil
                var accLastW2: Tensor? = nil
                var accLastB2: Tensor? = nil
                var accGamma: Tensor? = nil
                var accBeta: Tensor? = nil
                for _ in 0..<numChunks {
                    let take = min(micro, B - offset)
                    let txtChunk = Array(texts[offset..<(offset+take)])
                    let tgtChunk = Array(tgts[offset..<(offset+take)])
                    let pooled: Tensor
                    let out: Tensor
                    var lastCacheText: TextToCubeEncoder.LastTCNCache? = nil
                    var maskFixedText: [[Int]] = []
                    if args.unfreezeLastTCN {
                        let res = enc.forwardForTrainingWithLastBlockCache(texts: txtChunk)
                        pooled = res.pooled
                        out = res.out
                        lastCacheText = res.cache
                        maskFixedText = res.maskFixed
                    } else {
                        let r = enc.forwardForTraining(texts: txtChunk)
                        pooled = r.pooled
                        out = r.out
                    }
                    let b = out.shape[0]; let d = out.shape[1]
                    var tHost = [Float](repeating: 0, count: b*d)
                    for bi in 0..<b { for di in 0..<d { tHost[bi*d + di] = tgtChunk[bi][di] } }
                    let t = Tensor(shape: [b, d], data: tHost)
                    let mse = Losses.mseRowwise(out, t)
                    let cos = Losses.cosineSimilarityRowwise(out, t)
                    totalMSE += Double(mse.mean) * Double(b)
                    totalCos += Double(cos.mean) * Double(b)
                    seen += b
                    logger.debug(String(format: "train chunk (text): b=%d d=%d MSE=%.6f Cos=%.6f", b, d, mse.mean, cos.mean), category: Logger.Category.dataset)
                    var dY = dY_MSEMean(y: out, target: t)
                    if args.alphaCos != 0 {
                        let dYcos = dY_CosineMeanLoss(y: out, target: t)
                        let B2 = dY.shape[0]; let D2 = dY.shape[1]
                        for idx in 0..<(B2*D2) { dY.data[idx] = args.betaMSE * dY.data[idx] + args.alphaCos * dYcos.data[idx] }
                    } else {
                        let scale = args.betaMSE
                        if scale != 1.0 { for i in 0..<dY.count { dY.data[i] *= scale } }
                    }
                    let (dW, dB) = try enc.projectionGradientsGPU(X: pooled, dY: dY)
                    // Upstream to encoder (text-mode)
                    if args.unfreezeLastTCN, let cache = lastCacheText {
                        do {
                            let dXin = try enc.projectionInputGradientsGPU(dY: dY)
                            let dEnc = enc.maskedMeanBackward(dPooled: dXin, mask: maskFixedText, seqLen: modelCfg.maxLength)
                            var norm2: Float = 0
                            for v in dEnc.data { norm2 += v*v }
                            let gnorm = sqrt(norm2)
                            logger.debug(String(format: "dEnc L2=%.6f (text)", gnorm), category: Logger.Category.training)
                            // TCN block backward (same as token-mode)
                            var dOut = dEnc
                            let Bm = cache.xIn.shape[0]; let Lm = cache.xIn.shape[1]; let Dm = cache.xIn.shape[2]
                            for b2 in 0..<Bm {
                                for t2 in 0..<Lm {
                                    if maskFixedText[b2][t2] == 0 {
                                        let base = (b2 * Lm + t2) * Dm
                                        for di2 in 0..<Dm { dOut.data[base + di2] = 0 }
                                    }
                                }
                            }
                            let params = enc.getLastBlockParams()
                            let grads = try lastTCNBackward(cache: cache, mask: maskFixedText, dOut: dOut, modelCfg: modelCfg, params: LastTCNParams(w1: params.w1, b1: params.b1, w2: params.w2, b2: params.b2, gamma: params.gamma, beta: params.beta))
                            if accLastW1 == nil { accLastW1 = Tensor.zeros(params.w1.shape) }
                            if accLastB1 == nil { accLastB1 = params.b1 != nil ? Tensor.zeros([params.b1!.count]) : Tensor.zeros([0]) }
                            if accLastW2 == nil { accLastW2 = Tensor.zeros(params.w2.shape) }
                            if accLastB2 == nil { accLastB2 = params.b2 != nil ? Tensor.zeros([params.b2!.count]) : Tensor.zeros([0]) }
                            if accGamma == nil { accGamma = Tensor.zeros([params.gamma.count]) }
                            if accBeta == nil { accBeta = Tensor.zeros([params.beta.count]) }
                            for i2 in 0..<grads.dW1.count { accLastW1!.data[i2] += grads.dW1.data[i2] }
                            if let db1 = grads.dB1 { for i2 in 0..<db1.count { accLastB1!.data[i2] += db1.data[i2] } }
                            for i2 in 0..<grads.dW2.count { accLastW2!.data[i2] += grads.dW2.data[i2] }
                            if let db2 = grads.dB2 { for i2 in 0..<db2.count { accLastB2!.data[i2] += db2.data[i2] } }
                            for i2 in 0..<grads.dGamma.count { accGamma!.data[i2] += grads.dGamma.data[i2] }
                            for i2 in 0..<grads.dBeta.count { accBeta!.data[i2] += grads.dBeta.data[i2] }
                        } catch {
                            logger.warn("text-mode TCN backward failed: \(error)", category: Logger.Category.training)
                        }
                    }
                    if accW == nil { accW = Tensor.zeros(dW.shape) }
                    for i in 0..<dW.count { accW!.data[i] += dW.data[i] }
                    if accB == nil { accB = Tensor.zeros([dB.count]) }
                    for i in 0..<dB.count { accB!.data[i] += dB.data[i] }
                    stepCount += 1
                    if stepCount % args.accumSteps == 0 || offset + take >= B {
                        let scale = 1.0 / Float(stepCount % args.accumSteps == 0 ? args.accumSteps : stepCount)
                        var gW = accW!
                        var gB = accB!
                        // LR schedule (warmup + cosine)
                        let lrNow = LRSchedulers.warmupCosine(baseLR: args.lr, minLR: args.minLR, warmupSteps: args.warmupSteps, decaySteps: args.cosineDecaySteps, step: globalStep)
                        logger.debug(String(format: "opt step=%d lr=%.6g", globalStep, lrNow), category: Logger.Category.training)
                        optimizerStepProjectionAndLastBlock(enc: enc, opt: opt, inputs: OptimStepInputs(projGradW: gW, projGradB: gB, lastGrads: nil, lrNow: lrNow, scale: scale, clipNorm: args.clipNorm))
                        globalStep += 1
                        let outPost = enc.projectOnly(pooled)
                        let msePost = Losses.mseRowwise(outPost, t)
                        let cosPost = Losses.cosineSimilarityRowwise(outPost, t)
                        logger.debug(String(format: "post-upd: b=%d d=%d MSE=%.6f Cos=%.6f", b, d, msePost.mean, cosPost.mean), category: Logger.Category.dataset)
                        accW = nil; accB = nil; stepCount = 0
                    }
                    offset += take
                }
            }
            if seen > 0 {
                let avgMSE = totalMSE / Double(seen)
                let avgCos = totalCos / Double(seen)
                logger.info(String(format: "epoch %d summary: samples=%d avgMSE=%.6f avgCos=%.6f", epoch+1, seen, avgMSE, avgCos), category: Logger.Category.dataset)
            }
            // Validation
            if !valSamples.isEmpty {
                let (valMSE, valCos) = evaluate(enc: enc, samples: valSamples, batchSize: args.batchSize, microBatch: args.microBatch, maxLen: args.maxLength)
                logger.info(String(format: "epoch %d VAL: mse=%.6f cos=%.6f", epoch+1, valMSE, valCos), category: Logger.Category.dataset)
                if valCos > bestValCos {
                    bestValCos = valCos
                    if !args.saveCheckpoint.isEmpty {
                        let p = enc.getProjParams()
                        saveProjectionCheckpoint(path: args.saveCheckpoint, weight: p.weight, bias: p.bias)
                        logger.info("saved checkpoint: \(args.saveCheckpoint) (best cos=\(String(format: "%.6f", bestValCos)))", category: Logger.Category.dataset)
                    }
                    if !args.saveOptState.isEmpty {
                        let (projW, projB) = enc.getProjParams()
                        var counts: [Int] = [projW.count]
                        if let bT = projB { counts.append(bT.count) }
                        let ok = opt.saveState(path: args.saveOptState, paramCounts: counts)
                        logger.info("saved opt state: \(ok) -> \(args.saveOptState)", category: Logger.Category.training)
                    }
                }
            }
        }
    }
    
}

// Entry point
do {
    try run()
} catch {
    Logger.shared.error("EFTrain failed: \(error)", category: Logger.Category.dataset)
    exit(1)
}