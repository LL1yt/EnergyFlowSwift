import Foundation
import EnergyFlow
import EFCore

// Mixed-precision helpers
@inline(__always) func hasNaNOrInf(_ t: Tensor) -> Bool {
    for v in t.data { if !v.isFinite { return true } }
    return false
}

struct ResolvedTrainConfig {
    // Core encoder training
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
    // AB controls
    var enableAB: Bool
    var abRatio: Float
    // Data strictness
    var requireTokens: Bool
    // Decoder config for Mode B
    var decVocabSize: Int
    var decBlocks: Int
    var decHidden: Int
    var decKernelSize: Int
    var decDilation: [Int]
    var decLR: Float
    var decWeightDecay: Float
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
        let unfreezeLastTCN = c.unfreezeLastTCN,
        let enableAB = c.enableAB,
        let abRatio = c.abRatio,
        let decVocab = c.decVocabSize,
        let decBlocks = c.decBlocks,
        let decHidden = c.decHidden,
        let decKernel = c.decKernelSize,
        let decDil = c.decDilation,
        let decLR = c.decLR,
        let decWD = c.decWeightDecay
    else {
        FileHandle.standardError.write(Data("EFTrain error: invalid config structure. Ensure all required fields are present. See sample: EnergyFlowSwift/Configs/train_debug.json\n".utf8))
        return nil
    }
    if dataPath.isEmpty {
        FileHandle.standardError.write(Data("EFTrain error: dataPath is empty in config file \(path).\n".utf8))
        return nil
    }
    // Defaults for flags
    let requireTokens = c.requireTokens ?? true

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
        configPath: path,
        enableAB: enableAB,
        abRatio: abRatio,
        requireTokens: requireTokens,
        decVocabSize: decVocab,
        decBlocks: decBlocks,
        decHidden: decHidden,
        decKernelSize: decKernel,
        decDilation: decDil,
        decLR: decLR,
        decWeightDecay: decWD
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
    if args.requireTokens && !ds.hasTokens {
        FileHandle.standardError.write(Data("EFTrain error: requireTokens=true but dataset has no tokens. Regenerate dataset with input_ids and attention_mask.\n".utf8))
        return
    }
    // Configure encoder; set only what differs from defaults
    let modelCfg = TextToCubeEncoderConfig(
        maxLength: args.maxLength,
        outputDim: ds.embeddingDim
    )
    let enc = TextToCubeEncoder(modelConfig: modelCfg)
    // Optional decoder for Mode B
    var decTrainer: DecoderTrainer? = nil
    if args.enableAB {
        let decCfg = TextDecoderConfig(vocabSize: args.decVocabSize,
                                       dim: modelCfg.outputDim,
                                       hidden: args.decHidden,
                                       nBlocks: args.decBlocks,
                                       kernelSize: args.decKernelSize,
                                       dilationSchedule: args.decDilation,
                                       maxLength: args.maxLength)
        decTrainer = DecoderTrainer(config: decCfg, lr: args.decLR, weightDecay: args.decWeightDecay)
        logger.info("Decoder initialized: V=\(decCfg.vocabSize) D=\(decCfg.dim) blocks=\(decCfg.nBlocks) k=\(decCfg.kernelSize) dil=\(decCfg.dilationSchedule)", category: Logger.Category.training)
    }
    
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
    
    
    var bestValCos: Double = -Double.greatestFiniteMagnitude
    
    // Build unified encoder/decoder via CombinedTrainer
    // Enforce encoder hiddenDim == teacher embeddingDim and trunk hyperparams match decoder for last-block sync
    let encCfg = TextToCubeEncoderConfig(hiddenDim: ds.embeddingDim,
                                         maxLength: args.maxLength,
                                         outputDim: ds.embeddingDim,
                                         tcnBlocks: args.decBlocks,
                                         kernelSize: args.decKernelSize,
                                         dilationSchedule: args.decDilation,
                                         ffDim: args.decHidden)
    let decCfg = TextDecoderConfig(vocabSize: args.decVocabSize,
                                   dim: encCfg.outputDim,
                                   hidden: args.decHidden,
                                   nBlocks: args.decBlocks,
                                   kernelSize: args.decKernelSize,
                                   dilationSchedule: args.decDilation,
                                   maxLength: args.maxLength)
    let trainer = CombinedTrainer(encConfig: encCfg,
                                  decConfig: decCfg,
                                  lrEncProj: args.lr,
                                  weightDecayEnc: args.weightDecay,
                                  alphaCos: args.alphaCos,
                                  betaMSE: args.betaMSE,
                                  warmupSteps: args.warmupSteps,
                                  cosineDecaySteps: args.cosineDecaySteps,
                                  minLREncProj: args.minLR,
                                  clipNorm: args.clipNorm)
    // Set decoder optimizer hyperparams from config
    trainer.decTrainer.opt.lr = args.decLR
    trainer.decTrainer.opt.weightDecay = args.decWeightDecay
    
    for epoch in 0..<args.epochs {
        logger.info("epoch=\(epoch+1)/\(args.epochs)", category: Logger.Category.dataset)
        let epochStart = Date()
        
        var seen = 0
        var totalMSE: Double = 0
        var totalCos: Double = 0
        var totalCE: Double = 0
        var totalCESamples: Int = 0
        
        // Iterate train samples in batches of batchSize, each split to micro-batches
        var ptr = 0
        let totalBatches = (trainSamples.count + max(args.batchSize,1) - 1) / max(args.batchSize,1)
        var batchIndex = 0
        epoch_loop: while ptr < trainSamples.count {
            let end = min(ptr + args.batchSize, trainSamples.count)
            let slice = Array(trainSamples[ptr..<end])
            ptr = end
            batchIndex += 1
            // Batch start log (mid-verbosity)
            let plannedChunks = (min(args.batchSize, slice.count) + max(args.microBatch,1) - 1) / max(args.microBatch,1)
            let plannedAB = args.enableAB ? Int(round(Float(plannedChunks) * args.abRatio)) : plannedChunks
            Logger.shared.info1(String(format: "epoch %d/%d — batch %d/%d start: size=%d micro=%d chunks=%d A/B=%d/%d (hint: A=KD cos↑→1 mse↓→0, B=CE↓→0)",
                                       epoch+1, args.epochs, batchIndex, totalBatches, slice.count, args.microBatch, plannedChunks, plannedAB, max(plannedChunks - plannedAB,0)),
                                category: Logger.Category.training)
            let batchStart = Date()
            if args.maxBatches > 0 && (ptr / max(args.batchSize,1)) > args.maxBatches { break epoch_loop }
            // Enforce tokens present
            let hasTokens = slice.allSatisfy { $0.inputIDs != nil && $0.attentionMask != nil }
            if !hasTokens {
                FileHandle.standardError.write(Data("EFTrain error: batch contains samples without tokens. Regenerate dataset with input_ids and attention_mask.\n".utf8))
                return
            }
            // Build arrays
            var ids: [[Int]] = []
            var mask: [[Int]] = []
            var tgts: [[Float]] = []
            ids.reserveCapacity(slice.count)
            mask.reserveCapacity(slice.count)
            tgts.reserveCapacity(slice.count)
            for s in slice { ids.append(s.inputIDs!); mask.append(s.attentionMask!); tgts.append(s.target) }
            
            // Micro-batching
            let B = ids.count
            let micro = args.microBatch > 0 ? args.microBatch : B
            let numChunks = (B + micro - 1) / micro
            var offset = 0
            let aBudget = args.enableAB ? Int(round(Float(numChunks) * args.abRatio)) : numChunks
            var chunksDone = 0
            
            var lastA_mse: Float = 0
            var lastA_cos: Float = 0
            var lastB_ce: Float = 0
            // Track actual executed A/B chunks
            var actualAChunks = 0
            var actualBChunks = 0
            for _ in 0..<numChunks {
                let take = min(micro, B - offset)
                let idsChunk = Array(ids[offset..<(offset+take)])
                let maskChunk = Array(mask[offset..<(offset+take)])
                let tgtChunk = Array(tgts[offset..<(offset+take)])

                // Decide Mode A (KD) or Mode B (Decoder CE)
                let runA = (!args.enableAB) ? true : (chunksDone < aBudget)
                chunksDone += 1

                if runA {
                    actualAChunks += 1
                    // KD targets: z_teacher
                    var zData: [Float] = []
                    zData.reserveCapacity(take * encCfg.outputDim)
                    for i in 0..<take { zData.append(contentsOf: tgtChunk[i]) }
                    let zt = Tensor(shape: [take, encCfg.outputDim], data: zData)
                    do {
                        let (mse, cos) = try trainer.stepA(inputIDs: idsChunk, attentionMask: maskChunk, zTeacher: zt, unfreezeLastTCN: args.unfreezeLastTCN)
                        lastA_mse = mse; lastA_cos = cos
                        totalMSE += Double(mse) * Double(take)
                        totalCos += Double(cos) * Double(take)
                        seen += take
                    } catch {
                        logger.error("trainer.stepA failed: \(error)", category: Logger.Category.training)
                        return
                    }
                } else {
                    actualBChunks += 1
                    // Decoder CE targets
                    let L = args.maxLength
                    var idsPad: [[Int]] = []
                    idsPad.reserveCapacity(take)
                    for row in idsChunk {
                        if row.count > L { idsPad.append(Array(row.prefix(L))) }
                        else if row.count < L { idsPad.append(row + Array(repeating: 0, count: L - row.count)) }
                        else { idsPad.append(row) }
                    }
                    var targets: [[Int]] = []
                    targets.reserveCapacity(take)
                    for r in idsPad { var t = Array(r.dropFirst()); t.append(r.first ?? 0); targets.append(t) }

                    // z_teacher from dataset target floats
                    var zData: [Float] = []
                    zData.reserveCapacity(take * encCfg.outputDim)
                    for i in 0..<take { zData.append(contentsOf: tgtChunk[i]) }
                    let zt = Tensor(shape: [take, encCfg.outputDim], data: zData)
                    do {
                        let ce = try trainer.stepB(ids: idsPad, targets: targets, zTeacher: zt, unfreezeLastTCN: args.unfreezeLastTCN)
                        lastB_ce = ce
                        totalCE += Double(ce) * Double(take)
                        totalCESamples += take
                        seen += take
                    } catch {
                        logger.error("trainer.stepB failed: \(error)", category: Logger.Category.training)
                        return
                    }
                }
                offset += take
            }
            // Batch end summary (mid-verbosity)
            let bt = Date().timeIntervalSince(batchStart)
            let thrBatch = bt > 0 ? Double(B) / bt : 0
            let avgMSErun = seen > 0 ? totalMSE / Double(seen) : 0
            let avgCOSrun = seen > 0 ? totalCos / Double(seen) : 0
            let avgCErun = totalCESamples > 0 ? totalCE / Double(totalCESamples) : 0
            // ETA for the epoch based on running throughput
            let elapsedRun = Date().timeIntervalSince(epochStart)
            let thrRun = elapsedRun > 0 ? Double(seen) / elapsedRun : 0
            let remaining = max(0, trainSamples.count - seen)
            let etaSec = thrRun > 0 ? Double(remaining) / thrRun : 0
            let etaI = Int(etaSec.rounded())
            let etaH = etaI / 3600
            let etaM = (etaI % 3600) / 60
            let etaS = etaI % 60
            let etaStr = etaH > 0 ? String(format: "%dh %02dm %02ds", etaH, etaM, etaS) : String(format: "%dm %02ds", etaM, etaS)
            Logger.shared.info1(String(format: "epoch %d/%d — batch %d/%d done: lr=%.4g thr=%.1f/s lastA[mse=%.5f cos=%.5f] lastB[ce=%.5f] avg[mse=%.5f cos=%.5f ce=%.5f] A/B actual=%d/%d ETA=%@ (hints: cos↑→1, mse↓→0, ce↓→0)",
                                       epoch+1, args.epochs, batchIndex, totalBatches, trainer.optEncProj.lr, thrBatch,
                                       lastA_mse, lastA_cos, lastB_ce, avgMSErun, avgCOSrun, avgCErun, actualAChunks, actualBChunks, etaStr),
                                category: Logger.Category.training)
        }
        
        if seen > 0 {
            let avgMSE = totalMSE / Double(seen)
            let avgCos = totalCos / Double(seen)
            let elapsed = Date().timeIntervalSince(epochStart)
            let thr = elapsed > 0 ? Double(seen) / elapsed : 0
            if totalCESamples > 0 {
                let avgCE = totalCE / Double(totalCESamples)
                logger.info(String(format: "epoch %d summary: samples=%d avgMSE=%.6f avgCos=%.6f avgCE=%.6f thr=%.2f/s", epoch+1, seen, avgMSE, avgCos, avgCE, thr), category: Logger.Category.dataset)
            } else {
                logger.info(String(format: "epoch %d summary: samples=%d avgMSE=%.6f avgCos=%.6f thr=%.2f/s", epoch+1, seen, avgMSE, avgCos, thr), category: Logger.Category.dataset)
            }
        }
        
        // Validation
        if !valSamples.isEmpty {
            let (valMSE, valCos) = evaluate(enc: trainer.enc, samples: valSamples, batchSize: args.batchSize, microBatch: args.microBatch, maxLen: args.maxLength)
            var valMsg = String(format: "epoch %d VAL: mse=%.6f cos=%.6f", epoch+1, valMSE, valCos)
            if args.enableAB {
                let hasTokensVal = valSamples.first(where: { $0.inputIDs != nil && $0.attentionMask != nil }) != nil
                if hasTokensVal {
                    var vptr = 0
                    var ceSum: Double = 0
                    var ceCount: Int = 0
                    let valStart = Date()
                    let totalVal = valSamples.count
                    let stride = max(1, totalVal / 10)
                    while vptr < valSamples.count {
                        let vend = min(vptr + args.batchSize, valSamples.count)
                        let vslice = Array(valSamples[vptr..<vend])
                        vptr = vend
                        var ids: [[Int]] = []
                        ids.reserveCapacity(vslice.count)
                        var zData: [Float] = []
                        zData.reserveCapacity(vslice.count * encCfg.outputDim)
                        for s in vslice {
                            let row = s.inputIDs ?? []
                            if row.count > args.maxLength { ids.append(Array(row.prefix(args.maxLength))) }
                            else if row.count < args.maxLength { ids.append(row + Array(repeating: 0, count: args.maxLength - row.count)) }
                            else { ids.append(row) }
                            zData.append(contentsOf: s.target)
                        }
                        if ids.isEmpty { continue }
                        var targets: [[Int]] = []
                        targets.reserveCapacity(ids.count)
                        for r in ids { var t = Array(r.dropFirst()); t.append(r.first ?? 0); targets.append(t) }
                        let zt = Tensor(shape: [ids.count, encCfg.outputDim], data: zData)
                        let logits = trainer.decTrainer.decoder.forward(ids: ids, z: zt)
                        let ce = CrossEntropyLoss.meanLogits(logits: logits, targets: targets)
                        ceSum += Double(ce) * Double(ids.count)
                        ceCount += ids.count
                        // Periodic VAL progress log
                        if ceCount % stride == 0 || vptr >= valSamples.count {
                            let elapsed = Date().timeIntervalSince(valStart)
                            let thrVal = elapsed > 0 ? Double(ceCount) / elapsed : 0
                            let pct = Double(vptr) / Double(totalVal) * 100.0
                            let cePart = ceCount > 0 ? (ceSum / Double(ceCount)) : 0
                            Logger.shared.info1(String(format: "VAL progress: %d/%d (%.0f%%) ce=%.6f thr=%.1f/s", vptr, totalVal, pct, cePart, thrVal), category: Logger.Category.dataset)
                        }
                    }
                    if ceCount > 0 { valMsg += String(format: " ce=%.6f", ceSum / Double(ceCount)) }
                }
            }
            logger.info(valMsg, category: Logger.Category.dataset)
            if valCos > bestValCos {
                bestValCos = valCos
                if !args.saveCheckpoint.isEmpty {
                    let p = trainer.enc.getProjParams()
                    saveProjectionCheckpoint(path: args.saveCheckpoint, weight: p.weight, bias: p.bias)
                    logger.info("saved checkpoint: \(args.saveCheckpoint) (best cos=\(String(format: "%.6f", bestValCos)))", category: Logger.Category.dataset)
                }
                if !args.saveOptState.isEmpty {
                    let (projW, projB) = trainer.enc.getProjParams()
                    var counts: [Int] = [projW.count]
                    if let bT = projB { counts.append(bT.count) }
                    let ok = trainer.optEncProj.saveState(path: args.saveOptState, paramCounts: counts)
                    logger.info("saved opt state: \(ok) -> \(args.saveOptState)", category: Logger.Category.training)
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
