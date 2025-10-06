import Foundation
import EnergyFlow
import EFCore

struct TrainArgs {
    var dataPath: String
    var batchSize: Int = 16
    var maxLength: Int = 128
    var maxBatches: Int = 100
    var epochs: Int = 1
    var lr: Float = 3e-4
    var weightDecay: Float = 0.01
    var alphaCos: Float = 1.0   // weight for (1 - cosine)
    var betaMSE: Float = 1.0    // weight for MSE
    var microBatch: Int = 32
    var valFraction: Float = 0.1
    var saveCheckpoint: String = ""   // path to save best checkpoint
    var loadCheckpoint: String = ""   // path to load checkpoint
    var accumSteps: Int = 1
    // New training controls
    var warmupSteps: Int = 0
    var cosineDecaySteps: Int = 0
    var minLR: Float = 0.0
    var clipNorm: Float = 0.0
    var saveOptState: String = ""
    var loadOptState: String = ""
    // Unfreeze controls
    var unfreezeLastTCN: Bool = false
    // Central config
    var configPath: String = ""
}

func parseTrainArgs() -> TrainArgs? {
    // Defaults may be overridden by config file, then environment EFTRAIN_*, then CLI flags
    var a = TrainArgs(dataPath: "")
    let env = ProcessInfo.processInfo.environment
    func envInt(_ key: String, _ def: Int) -> Int { if let s = env[key], let v = Int(s) { return v } else { return def } }
    func envFloat(_ key: String, _ def: Float) -> Float { if let s = env[key], let v = Float(s) { return v } else { return def } }
    func envString(_ key: String, _ def: String = "") -> String { env[key] ?? def }
    // Pre-scan CLI for --config path (so we can load file before parsing overrides)
    let argv = Array(CommandLine.arguments.dropFirst())
    if let idx = argv.firstIndex(of: "--config"), idx + 1 < argv.count {
        let cpath = argv[idx + 1]
        a.configPath = cpath
        if let cfg = try? TrainConfig.load(path: cpath) {
            var tmp = cfg
            tmp.apply(to: &a)
        }
    }

    // Environment overrides (selected keys)
    if let s = env["EFTRAIN_UNFREEZE_LAST_TCN"] { a.unfreezeLastTCN = (s == "1" || s.lowercased() == "true" || s.lowercased() == "yes") }

    // Then override by CLI flags
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let key = it.next() {
        switch key {
        case "--config": _ = it.next().map { a.configPath = $0 }
        case "--data": if let v = it.next() { a.dataPath = v }
        case "--batch-size": if let v = it.next(), let n = Int(v) { a.batchSize = n }
        case "--max-length": if let v = it.next(), let n = Int(v) { a.maxLength = n }
        case "--max-batches": if let v = it.next(), let n = Int(v) { a.maxBatches = n }
        case "--epochs": if let v = it.next(), let n = Int(v) { a.epochs = n }
        case "--lr": if let v = it.next(), let f = Float(v) { a.lr = f }
        case "--weight-decay": if let v = it.next(), let f = Float(v) { a.weightDecay = f }
        case "--micro-batch": if let v = it.next(), let n = Int(v) { a.microBatch = n }
        case "--val-fraction": if let v = it.next(), let f = Float(v) { a.valFraction = f }
        case "--save-checkpoint": if let v = it.next() { a.saveCheckpoint = v }
        case "--load-checkpoint": if let v = it.next() { a.loadCheckpoint = v }
        case "--accum-steps": if let v = it.next(), let n = Int(v) { a.accumSteps = max(1, n) }
        case "--alpha-cos": if let v = it.next(), let f = Float(v) { a.alphaCos = f }
        case "--beta-mse": if let v = it.next(), let f = Float(v) { a.betaMSE = f }
        case "--unfreeze-last-tcn": a.unfreezeLastTCN = true
        case "--warmup-steps": if let v = it.next(), let n = Int(v) { a.warmupSteps = max(0, n) }
        case "--cosine-decay-steps": if let v = it.next(), let n = Int(v) { a.cosineDecaySteps = max(0, n) }
        case "--min-lr": if let v = it.next(), let f = Float(v) { a.minLR = max(0, f) }
        case "--clip-norm": if let v = it.next(), let f = Float(v) { a.clipNorm = max(0, f) }
        case "--save-opt-state": if let v = it.next() { a.saveOptState = v }
        case "--load-opt-state": if let v = it.next() { a.loadOptState = v }
        case "-h", "--help": return nil
        default: continue
        }
    }
    if a.dataPath.isEmpty { return nil }
    return a
}

func usage() {
    print("Usage: EFTrain [--config path.json] --data /path/to/data.jsonl|.efb [--batch-size N] [--max-length N] [--max-batches N] [--epochs N] [--lr F] [--weight-decay F] [--micro-batch N] [--alpha-cos F] [--beta-mse F] [--val-fraction F] [--save-checkpoint path] [--load-checkpoint path] [--accum-steps N] [--warmup-steps N] [--cosine-decay-steps N] [--min-lr F] [--clip-norm F] [--save-opt-state path] [--load-opt-state path] [--unfreeze-last-tcn]\nPrecedence: config file < environment EFTRAIN_* < CLI flags. Set EFTRAIN_* to avoid typing flags.")
}

func run() throws {
    guard let args = parseTrainArgs() else { usage(); return }
    var globalStep = 0
    let logger = Logger.shared
    logger.info("EFTrain start: data=\(args.dataPath) batchSize=\(args.batchSize) maxLen=\(args.maxLength) epochs=\(args.epochs) alphaCos=\(args.alphaCos) betaMSE=\(args.betaMSE)", category: Logger.Category.dataset)
    
    let ds = try SimpleJSONLDataset(path: args.dataPath)
    // Configure encoder; outputDim must match teacher dim
    let cfg = TextToCubeEncoderConfig(
        hiddenDim: 256,
        maxLength: args.maxLength,
        outputDim: ds.embeddingDim,
        useTanhOutput: false,
        tcnBlocks: 4,
        kernelSize: 5,
        dilationSchedule: [1,2,4,8],
        ffDim: 512,
        useGPUProjection: true,
        baseSeed: 42
    )
    let enc = TextToCubeEncoder(modelConfig: cfg)
    
    // Load checkpoint if provided
    if !args.loadCheckpoint.isEmpty, let (wLoaded, bLoaded) = loadProjectionCheckpoint(path: args.loadCheckpoint) {
        if wLoaded.shape == [cfg.outputDim, cfg.hiddenDim] {
            enc.setProjParams(weight: wLoaded, bias: bLoaded)
            enc.invalidateProjectionCache()
            logger.info("loaded checkpoint: \(args.loadCheckpoint) W=\(wLoaded.prettyShape) B=\(bLoaded != nil)", category: Logger.Category.dataset)
        } else {
            logger.warn("checkpoint shape mismatch: expected [\(cfg.outputDim), \(cfg.hiddenDim)] got \(wLoaded.prettyShape) — skipping load", category: Logger.Category.dataset)
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
        var batchIdx = 0
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
                    logger.info(String(format: "train chunk: b=%d d=%d MSE=%.6f Cos=%.6f", b, d, mse.mean, cos.mean), category: Logger.Category.dataset)
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
                        let maskFixed = args.unfreezeLastTCN ? maskFixedLocal : padOrTruncateMask(maskChunk, cfg.maxLength)
                        let dEnc = enc.maskedMeanBackward(dPooled: dXin, mask: maskFixed, seqLen: cfg.maxLength)
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
                            // Conv2 backward (GPU via GraphLinear on flattened [B*L, H])
                            let params = enc.getLastBlockParams()
                            let Bc = cache.h1a.shape[0], Lc = cache.h1a.shape[1], Hc = cache.h1a.shape[2]
                            let Dc = dOut.shape[2]
                            let Xf = cache.h1a.reshaped([Bc * Lc, Hc])
                            let dYf = dOut.reshaped([Bc * Lc, Dc])
                            var gl = GraphLinear(inFeatures: Hc, outFeatures: Dc, bias: params.b2 != nil, seed: 0)
                            // Set weights/bias to conv2 params
                            gl.weight = params.w2.reshaped([Dc, Hc])
                            gl.bias = params.b2
                            // Ensure GPU weight buffer exists (run cheap dummy forward)
                            _ = try? gl.forward(Tensor.zeros([1, Hc]))
                            // Gradients and input gradient
                            let (dW2lin, dB2) = try gl.gradientsGPU(X: Xf, dY: dYf)
                            let dX2f = try gl.inputGradientsGPU(dY: dYf)
                            let dX2 = dX2f.reshaped([Bc, Lc, Hc])
                            // GELU backward
                            let dH1 = dGELU(x: cache.h1, upstream: dX2)
                            // Conv1 backward
                            let conv1 = Conv1DGrad.backward(X: cache.norm, W: params.w1, dY: dH1, dilation: cfg.kernelSize == 1 ? 1 : (cfg.dilationSchedule.last ?? 1))
                            // LN backward
                            let Bf = cache.xIn.shape[0]; let Lf = cache.xIn.shape[1]; let Df = cache.xIn.shape[2]
                            let gNormFlat = conv1.dX.reshaped([Bf * Lf, Df])
                            let xFlat = cache.xIn.reshaped([Bf * Lf, Df])
                            let (dxFlat, dGamma, dBeta) = layerNormBackward(x: xFlat, upstream: gNormFlat, gamma: params.gamma)
                            _ = dxFlat // upstream to earlier blocks (ignored)
                            // Accumulate grads for last block (initialize accumulators lazily)
                            if accW == nil { /* projection acc created earlier */ }
                            // Use separate accumulators for last block
                            if accLastW1 == nil { accLastW1 = Tensor.zeros(params.w1.shape) }
                            if accLastB1 == nil { accLastB1 = params.b1 != nil ? Tensor.zeros([params.b1!.count]) : Tensor.zeros([0]) }
                            if accLastW2 == nil { accLastW2 = Tensor.zeros(params.w2.shape) }
                            if accLastB2 == nil { accLastB2 = params.b2 != nil ? Tensor.zeros([params.b2!.count]) : Tensor.zeros([0]) }
                            if accGamma == nil { accGamma = Tensor.zeros([params.gamma.count]) }
                            if accBeta == nil { accBeta = Tensor.zeros([params.beta.count]) }
                            for i2 in 0..<conv1.dW.count { accLastW1!.data[i2] += conv1.dW.data[i2] }
                            if params.b1 != nil { for i2 in 0..<conv1.dB.count { accLastB1!.data[i2] += conv1.dB.data[i2] } }
                            let dW2 = dW2lin.reshaped([Dc, Hc, 1])
                            for i2 in 0..<dW2.count { accLastW2!.data[i2] += dW2.data[i2] }
                            if params.b2 != nil { for i2 in 0..<dB2.count { accLastB2!.data[i2] += dB2.data[i2] } }
                            for i2 in 0..<dGamma.count { accGamma!.data[i2] += dGamma.data[i2] }
                            for i2 in 0..<dBeta.count { accBeta!.data[i2] += dBeta.data[i2] }
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
                        for i in 0..<gW.count { gW.data[i] *= scale }
                        for i in 0..<gB.count { gB.data[i] *= scale }
                        // Gradient clipping (global L2)
                        if args.clipNorm > 0 {
                            var gradList: [Tensor] = [gW]
                            if gB.count > 0 { gradList.append(gB) }
                            _ = GradClip.clipGlobalL2Norm(tensors: &gradList, maxNorm: args.clipNorm, eps: 1e-6)
                            gW = gradList[0]
                            if gradList.count > 1 { gB = gradList[1] }
                        }
                        // Param list and LR scheduling
                        let (projW, projB) = enc.getProjParams()
                        var params: [Tensor] = [projW]
                        var grads: [Tensor] = [gW]
                        if let bT = projB { params.append(bT); grads.append(gB) }
                        if args.unfreezeLastTCN {
                            // Average last-block grads and optionally clip
                            if let aW1 = accLastW1 { var g = aW1; for i in 0..<g.count { g.data[i] *= scale }; params.append(enc.getLastBlockParams().w1); grads.append(g) }
                            if let aB1 = accLastB1, aB1.count > 0 { var g = aB1; for i in 0..<g.count { g.data[i] *= scale }; if let b1p = enc.getLastBlockParams().b1 { params.append(b1p); grads.append(g) } }
                            if let aW2 = accLastW2 { var g = aW2; for i in 0..<g.count { g.data[i] *= scale }; params.append(enc.getLastBlockParams().w2); grads.append(g) }
                            if let aB2 = accLastB2, aB2.count > 0 { var g = aB2; for i in 0..<g.count { g.data[i] *= scale }; if let b2p = enc.getLastBlockParams().b2 { params.append(b2p); grads.append(g) } }
                            if let aG = accGamma { var g = aG; for i in 0..<g.count { g.data[i] *= scale }; params.append(enc.getLastBlockParams().gamma); grads.append(g) }
                            if let aBt = accBeta { var g = aBt; for i in 0..<g.count { g.data[i] *= scale }; params.append(enc.getLastBlockParams().beta); grads.append(g) }
                            if args.clipNorm > 0 {
                                var list = grads
                                _ = GradClip.clipGlobalL2Norm(tensors: &list, maxNorm: args.clipNorm, eps: 1e-6)
                                grads = list
                            }
                        }
                        // LR schedule (warmup + cosine)
                        let lrNow = LRSchedulers.warmupCosine(baseLR: args.lr, minLR: args.minLR, warmupSteps: args.warmupSteps, decaySteps: args.cosineDecaySteps, step: globalStep)
                        if lrNow != opt.lr { opt.lr = lrNow }
                        logger.info(String(format: "opt step=%d lr=%.6g", globalStep, opt.lr), category: Logger.Category.training)
                        var paramsCopy = params
                        opt.step(params: &paramsCopy, grads: grads)
                        globalStep += 1
                        let newW = paramsCopy[0]
                        let newB = (projB != nil) ? paramsCopy[1] : nil
                        enc.setProjParams(weight: newW, bias: newB)
                        enc.invalidateProjectionCache()
                        if args.unfreezeLastTCN {
                            // After step, reload last block params from paramsCopy tail in same order as added
                            var idxTail = 1 + (projB != nil ? 1 : 0)
                            if let _ = accLastW1 {
                                let w1New = paramsCopy[idxTail]; idxTail += 1
                                var b1New: Tensor? = nil
                                if let aB1 = accLastB1, aB1.count > 0 { b1New = paramsCopy[idxTail]; idxTail += 1 }
                                let w2New = paramsCopy[idxTail]; idxTail += 1
                                var b2New: Tensor? = nil
                                if let aB2 = accLastB2, aB2.count > 0 { b2New = paramsCopy[idxTail]; idxTail += 1 }
                                let gammaNew = paramsCopy[idxTail]; idxTail += 1
                                let betaNew = paramsCopy[idxTail]; idxTail += 1
                                enc.setLastBlockParams(w1: w1New, b1: b1New, w2: w2New, b2: b2New, gamma: gammaNew, beta: betaNew)
                                enc.invalidateLastBlockCaches()
                            }
                        }
                        // Post-update metrics using project-only
                        let outPost = enc.projectOnly(pooled)
                        let msePost = Losses.mseRowwise(outPost, t)
                        let cosPost = Losses.cosineSimilarityRowwise(outPost, t)
                        logger.info(String(format: "post-upd: b=%d d=%d MSE=%.6f Cos=%.6f", b, d, msePost.mean, cosPost.mean), category: Logger.Category.dataset)
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
                    logger.info(String(format: "train chunk (text): b=%d d=%d MSE=%.6f Cos=%.6f", b, d, mse.mean, cos.mean), category: Logger.Category.dataset)
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
                            let dEnc = enc.maskedMeanBackward(dPooled: dXin, mask: maskFixedText, seqLen: cfg.maxLength)
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
                            // Conv2 backward (GPU)
                            let params = enc.getLastBlockParams()
                            let Bc = cache.h1a.shape[0], Lc = cache.h1a.shape[1], Hc = cache.h1a.shape[2]
                            let Dc = dOut.shape[2]
                            let Xf = cache.h1a.reshaped([Bc * Lc, Hc])
                            let dYf = dOut.reshaped([Bc * Lc, Dc])
                            var gl = GraphLinear(inFeatures: Hc, outFeatures: Dc, bias: params.b2 != nil, seed: 0)
                            gl.weight = params.w2.reshaped([Dc, Hc])
                            gl.bias = params.b2
                            _ = try? gl.forward(Tensor.zeros([1, Hc]))
                            let (dW2lin, dB2t) = try gl.gradientsGPU(X: Xf, dY: dYf)
                            let dX2f = try gl.inputGradientsGPU(dY: dYf)
                            let dX2 = dX2f.reshaped([Bc, Lc, Hc])
                            let dH1 = dGELU(x: cache.h1, upstream: dX2)
                            let conv1 = Conv1DGrad.backward(X: cache.norm, W: params.w1, dY: dH1, dilation: cfg.kernelSize == 1 ? 1 : (cfg.dilationSchedule.last ?? 1))
                            let Bf = cache.xIn.shape[0]; let Lf = cache.xIn.shape[1]; let Df = cache.xIn.shape[2]
                            let gNormFlat = conv1.dX.reshaped([Bf * Lf, Df])
                            let xFlat = cache.xIn.reshaped([Bf * Lf, Df])
                            let (dxFlat, dGammaT, dBetaT) = layerNormBackward(x: xFlat, upstream: gNormFlat, gamma: params.gamma)
                            _ = dxFlat
                            // Accumulate (text mode uses same accumulators as token mode)
                            if accLastW1 == nil { accLastW1 = Tensor.zeros(params.w1.shape) }
                            if accLastB1 == nil { accLastB1 = params.b1 != nil ? Tensor.zeros([params.b1!.count]) : Tensor.zeros([0]) }
                            if accLastW2 == nil { accLastW2 = Tensor.zeros(params.w2.shape) }
                            if accLastB2 == nil { accLastB2 = params.b2 != nil ? Tensor.zeros([params.b2!.count]) : Tensor.zeros([0]) }
                            if accGamma == nil { accGamma = Tensor.zeros([params.gamma.count]) }
                            if accBeta == nil { accBeta = Tensor.zeros([params.beta.count]) }
                            for i2 in 0..<conv1.dW.count { accLastW1!.data[i2] += conv1.dW.data[i2] }
                            if params.b1 != nil { for i2 in 0..<conv1.dB.count { accLastB1!.data[i2] += conv1.dB.data[i2] } }
                            let dW2t = dW2lin.reshaped([Dc, Hc, 1])
                            for i2 in 0..<dW2t.count { accLastW2!.data[i2] += dW2t.data[i2] }
                            if params.b2 != nil { for i2 in 0..<dB2t.count { accLastB2!.data[i2] += dB2t.data[i2] } }
                            for i2 in 0..<dGammaT.count { accGamma!.data[i2] += dGammaT.data[i2] }
                            for i2 in 0..<dBetaT.count { accBeta!.data[i2] += dBetaT.data[i2] }
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
                        for i in 0..<gW.count { gW.data[i] *= scale }
                        for i in 0..<gB.count { gB.data[i] *= scale }
                        // Gradient clipping (global L2)
                        if args.clipNorm > 0 {
                            var gradList: [Tensor] = [gW]
                            if gB.count > 0 { gradList.append(gB) }
                            _ = GradClip.clipGlobalL2Norm(tensors: &gradList, maxNorm: args.clipNorm, eps: 1e-6)
                            gW = gradList[0]
                            if gradList.count > 1 { gB = gradList[1] }
                        }
                        let (projW, projB) = enc.getProjParams()
                        var params: [Tensor] = [projW]
                        var grads: [Tensor] = [gW]
                        if let bT = projB { params.append(bT); grads.append(gB) }
                        // LR schedule (warmup + cosine)
                        let lrNow = LRSchedulers.warmupCosine(baseLR: args.lr, minLR: args.minLR, warmupSteps: args.warmupSteps, decaySteps: args.cosineDecaySteps, step: globalStep)
                        if lrNow != opt.lr { opt.lr = lrNow }
                        logger.info(String(format: "opt step=%d lr=%.6g", globalStep, opt.lr), category: Logger.Category.training)
                        var paramsCopy = params
                        opt.step(params: &paramsCopy, grads: grads)
                        globalStep += 1
                        let newW = paramsCopy[0]
                        let newB = (projB != nil) ? paramsCopy[1] : nil
                        enc.setProjParams(weight: newW, bias: newB)
                        enc.invalidateProjectionCache()
                        let outPost = enc.projectOnly(pooled)
                        let msePost = Losses.mseRowwise(outPost, t)
                        let cosPost = Losses.cosineSimilarityRowwise(outPost, t)
                        logger.info(String(format: "post-upd: b=%d d=%d MSE=%.6f Cos=%.6f", b, d, msePost.mean, cosPost.mean), category: Logger.Category.dataset)
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
    
    // MARK: - Evaluation helper
    func evaluate(enc: TextToCubeEncoder, samples: [JSONLSample], batchSize: Int, microBatch: Int, maxLen: Int) -> (Double, Double) {
        var ptr = 0
        var totalMSE: Double = 0
        var totalCos: Double = 0
        var seen = 0
        while ptr < samples.count {
            let end = min(ptr + batchSize, samples.count)
            let slice = Array(samples[ptr..<end])
            ptr = end
            let hasTokens = slice.allSatisfy { $0.inputIDs != nil && $0.attentionMask != nil }
            if hasTokens {
                var ids: [[Int]] = []
                var mask: [[Int]] = []
                var tgts: [[Float]] = []
                for s in slice { ids.append(s.inputIDs!); mask.append(s.attentionMask!); tgts.append(s.target) }
                let B = ids.count
                let micro = microBatch > 0 ? microBatch : B
                var offset = 0
                let numChunks = (B + micro - 1) / micro
                for _ in 0..<numChunks {
                    let take = min(micro, B - offset)
                    let idsChunk = Array(ids[offset..<(offset+take)])
                    let maskChunk = Array(mask[offset..<(offset+take)])
                    let tgtChunk = Array(tgts[offset..<(offset+take)])
                    let out = enc.encodeTokens(inputIDs: idsChunk, attentionMask: maskChunk)
                    let b = out.shape[0]; let d = out.shape[1]
                    var pred = Array(repeating: Array(repeating: 0.0 as Float, count: d), count: b)
                    for bi in 0..<b { for di in 0..<d { pred[bi][di] = out.data[bi*d + di] } }
                    let m = batchMSE(pred, tgtChunk)
                    let c = batchCosine(pred, tgtChunk)
                    totalMSE += Double(m) * Double(b)
                    totalCos += Double(c) * Double(b)
                    seen += b
                    offset += take
                }
            } else {
                let texts = slice.map { $0.text ?? "" }
                let tgts = slice.map { $0.target }
                let B = texts.count
                let micro = microBatch > 0 ? microBatch : B
                var offset = 0
                let numChunks = (B + micro - 1) / micro
                for _ in 0..<numChunks {
                    let take = min(micro, B - offset)
                    let txtChunk = Array(texts[offset..<(offset+take)])
                    let tgtChunk = Array(tgts[offset..<(offset+take)])
                    let out = enc.encode(txtChunk)
                    let b = out.shape[0]; let d = out.shape[1]
                    var pred = Array(repeating: Array(repeating: 0.0 as Float, count: d), count: b)
                    for bi in 0..<b { for di in 0..<d { pred[bi][di] = out.data[bi*d + di] } }
                    let m = batchMSE(pred, tgtChunk)
                    let c = batchCosine(pred, tgtChunk)
                    totalMSE += Double(m) * Double(b)
                    totalCos += Double(c) * Double(b)
                    seen += b
                    offset += take
                }
            }
        }
        let avgMSE = seen > 0 ? totalMSE / Double(seen) : 0
        let avgCos = seen > 0 ? totalCos / Double(seen) : 0
        return (avgMSE, avgCos)
    }
    
    // reuse utilities from EFTextEval
    func batchMSE(_ y: [[Float]], _ t: [[Float]]) -> Float {
        precondition(y.count == t.count)
        var acc: Float = 0
        for i in 0..<y.count { acc += mse(y[i], t[i]) }
        return acc / Float(max(y.count, 1))
    }
    func mse(_ y: [Float], _ t: [Float]) -> Float {
        precondition(y.count == t.count)
        var acc: Float = 0
        for i in 0..<y.count { let d = y[i] - t[i]; acc += d*d }
        return acc / Float(y.count)
    }
    func cosine(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var dot: Float = 0
        var na: Float = 0
        var nb: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i]; na += a[i]*a[i]; nb += b[i]*b[i] }
        let denom = sqrt(max(na, 1e-12)) * sqrt(max(nb, 1e-12))
        return denom > 0 ? dot / denom : 0
    }
    func batchCosine(_ y: [[Float]], _ t: [[Float]]) -> Float {
        precondition(y.count == t.count)
        var acc: Float = 0
        for i in 0..<y.count { acc += cosine(y[i], t[i]) }
        return acc / Float(max(y.count, 1))
    }
    
    // MARK: - Checkpoint I/O (projection-only)
    func saveProjectionCheckpoint(path: String, weight: Tensor, bias: Tensor?) {
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var data = Data()
        // magic
        data.append(contentsOf: [0x45, 0x46, 0x43, 0x4B, 0x31]) // "EFCK1"
        func putU32(_ v: UInt32) { var le = v.littleEndian; withUnsafeBytes(of: &le) { data.append(contentsOf: $0) } }
        func putF32(_ v: Float) { var le = v.bitPattern.littleEndian; withUnsafeBytes(of: &le) { data.append(contentsOf: $0) } }
        putU32(UInt32(weight.shape[0]))
        putU32(UInt32(weight.shape[1]))
        putU32(UInt32(bias != nil ? 1 : 0))
        for v in weight.data { putF32(v) }
        if let b = bias { for v in b.data { putF32(v) } }
        try? data.write(to: url)
    }
    
    func loadProjectionCheckpoint(path: String) -> (Tensor, Tensor?)? {
        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url) else { return nil }
        var idx = 0
        func getU8() -> UInt8 { defer { idx += 1 }; return data[idx] }
        guard idx + 5 <= data.count else { return nil }
        // magic "EFCK1"
        if getU8() != 0x45 || getU8() != 0x46 || getU8() != 0x43 || getU8() != 0x4B || getU8() != 0x31 { return nil }
        func getU32() -> UInt32 { defer { idx += 4 }; return data[idx..<(idx+4)].withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian }
        func getF32() -> Float { defer { idx += 4 }; let u = data[idx..<(idx+4)].withUnsafeBytes { $0.load(as: UInt32.self) }.littleEndian; return Float(bitPattern: u) }
        let out = Int(getU32())
        let inf = Int(getU32())
        let hasB = getU32() != 0
        let wCount = out * inf
        if idx + 4 * wCount > data.count { return nil }
        var wHost = [Float](repeating: 0, count: wCount)
        for i in 0..<wCount { wHost[i] = getF32() }
        let weight = Tensor(shape: [out, inf], data: wHost)
        var bias: Tensor? = nil
        if hasB {
            if idx + 4 * out > data.count { return nil }
            var bHost = [Float](repeating: 0, count: out)
            for i in 0..<out { bHost[i] = getF32() }
            bias = Tensor(shape: [out], data: bHost)
        }
        return (weight, bias)
    }
}

do {
    try run()
} catch {
    Logger.shared.error("EFTrain failed: \(error)", category: Logger.Category.dataset)
    exit(1)
}
