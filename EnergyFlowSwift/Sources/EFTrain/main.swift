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
    var microBatch: Int = 8
}

func parseTrainArgs() -> TrainArgs? {
    var a = TrainArgs(dataPath: "")
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let key = it.next() {
        switch key {
        case "--data": if let v = it.next() { a.dataPath = v }
        case "--batch-size": if let v = it.next(), let n = Int(v) { a.batchSize = n }
        case "--max-length": if let v = it.next(), let n = Int(v) { a.maxLength = n }
        case "--max-batches": if let v = it.next(), let n = Int(v) { a.maxBatches = n }
        case "--epochs": if let v = it.next(), let n = Int(v) { a.epochs = n }
        case "--lr": if let v = it.next(), let f = Float(v) { a.lr = f }
        case "--weight-decay": if let v = it.next(), let f = Float(v) { a.weightDecay = f }
        case "--micro-batch": if let v = it.next(), let n = Int(v) { a.microBatch = n }
        case "--alpha-cos": if let v = it.next(), let f = Float(v) { a.alphaCos = f }
        case "--beta-mse": if let v = it.next(), let f = Float(v) { a.betaMSE = f }
        case "-h", "--help": return nil
        default: continue
        }
    }
    if a.dataPath.isEmpty { return nil }
    return a
}

func usage() {
    print("Usage: EFTrain --data /path/to/data.jsonl [--batch-size 16] [--max-length 128] [--max-batches 100] [--epochs 1] [--lr 3e-4] [--weight-decay 0.01] [--micro-batch 8] [--alpha-cos 1.0] [--beta-mse 1.0]")
}

func run() throws {
    guard let args = parseTrainArgs() else { usage(); return }
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

    // Optimizer: we update only projector weights/bias for now
    let opt = AdamW(lr: args.lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: args.weightDecay)

    for epoch in 0..<args.epochs {
        logger.info("epoch=\(epoch+1)/\(args.epochs)", category: Logger.Category.dataset)
        var batchIdx = 0
        var seen = 0
        var totalMSE: Double = 0
        var totalCos: Double = 0
        for batch in ds.batches(batchSize: args.batchSize, dropLast: false, padTokens: true) {
            batchIdx += 1
            if args.maxBatches > 0 && batchIdx > args.maxBatches { break }
            switch batch {
            case let .tokens(inputIDs, attentionMask, targets):
                let B = inputIDs.count
                let micro = args.microBatch > 0 ? args.microBatch : B
                let numChunks = (B + micro - 1) / micro
                var offset = 0
                for _ in 0..<numChunks {
                    let take = min(micro, B - offset)
                    let idsChunk = Array(inputIDs[offset..<(offset+take)])
                    let maskChunk = Array(attentionMask[offset..<(offset+take)])
                    let tgtChunk = Array(targets[offset..<(offset+take)])
                    let (pooled, out) = enc.forwardForTraining(inputIDs: idsChunk, attentionMask: maskChunk)
                    // Convert target to tensor
                    let b = out.shape[0]; let d = out.shape[1]
                    var tHost = [Float](repeating: 0, count: b*d)
                    for bi in 0..<b { for di in 0..<d { tHost[bi*d + di] = tgtChunk[bi][di] } }
                    let t = Tensor(shape: [b, d], data: tHost)
                    // Metrics
                    let mse = Losses.mseRowwise(out, t)
                    let cos = Losses.cosineSimilarityRowwise(out, t)
                    totalMSE += Double(mse.mean) * Double(b)
                    totalCos += Double(cos.mean) * Double(b)
                    seen += b
                    logger.info(String(format: "train chunk: b=%d d=%d MSE=%.6f Cos=%.6f", b, d, mse.mean, cos.mean), category: Logger.Category.dataset)
                    // Backward: combine MSE and cosine components
                    var dY = dY_MSEMean(y: out, target: t)  // [b, d]
                    if args.alphaCos != 0 {
                        let dYcos = dY_CosineMeanLoss(y: out, target: t) // for (1 - mean cosine)
                        // total grad: beta*MSE + alpha*Cos
                        let B = dY.shape[0]; let D = dY.shape[1]
                        for idx in 0..<(B*D) {
                            dY.data[idx] = args.betaMSE * dY.data[idx] + args.alphaCos * dYcos.data[idx]
                        }
                    } else {
                        // pure MSE
                        let scale = args.betaMSE
                        if scale != 1.0 {
                            let n = dY.count
                            for i in 0..<n { dY.data[i] *= scale }
                        }
                    }
                    let (projW, projB) = enc.getProjParams()
                    let (dW, dB) = gradsGraphLinear(X: pooled, dY: dY, outFeatures: d, inFeatures: pooled.shape[1])
                    // Optimizer step
                    var params: [Tensor] = [projW]
                    var grads: [Tensor] = [dW]
                    if let bT = projB { params.append(bT); grads.append(dB) }
                    var paramsCopy = params
                    opt.step(params: &paramsCopy, grads: grads)
                    // Reassign updated params back and invalidate cache
                    let newW = paramsCopy[0]
                    let newB = (projB != nil) ? paramsCopy[1] : nil
                    enc.setProjParams(weight: newW, bias: newB)
                    enc.invalidateProjectionCache()
                    // Post-update metrics using project-only on same pooled
                    let outPost = enc.projectOnly(pooled)
                    let msePost = Losses.mseRowwise(outPost, t)
                    let cosPost = Losses.cosineSimilarityRowwise(outPost, t)
                    logger.info(String(format: "post-upd: b=%d d=%d MSE=%.6f Cos=%.6f", b, d, msePost.mean, cosPost.mean), category: Logger.Category.dataset)
                    offset += take
                }
            case let .text(texts, targets):
                let B = texts.count
                let micro = args.microBatch > 0 ? args.microBatch : B
                let numChunks = (B + micro - 1) / micro
                var offset = 0
                for _ in 0..<numChunks {
                    let take = min(micro, B - offset)
                    let txtChunk = Array(texts[offset..<(offset+take)])
                    let tgtChunk = Array(targets[offset..<(offset+take)])
                    let (pooled, out) = enc.forwardForTraining(texts: txtChunk)
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
                    logger.info(String(format: "train chunk (text): b=%d d=%d MSE=%.6f Cos=%.6f", b, d, mse.mean, cos.mean), category: Logger.Category.dataset)
                    // Backward combine
                    var dY = dY_MSEMean(y: out, target: t)
                    if args.alphaCos != 0 {
                        let dYcos = dY_CosineMeanLoss(y: out, target: t)
                        let B = dY.shape[0]; let D = dY.shape[1]
                        for idx in 0..<(B*D) { dY.data[idx] = args.betaMSE * dY.data[idx] + args.alphaCos * dYcos.data[idx] }
                    } else {
                        let scale = args.betaMSE
                        if scale != 1.0 { for i in 0..<dY.count { dY.data[i] *= scale } }
                    }
                    let (projW, projB) = enc.getProjParams()
                    let (dW, dB) = gradsGraphLinear(X: pooled, dY: dY, outFeatures: d, inFeatures: pooled.shape[1])
                    var params: [Tensor] = [projW]
                    var grads: [Tensor] = [dW]
                    if let bT = projB { params.append(bT); grads.append(dB) }
                    var paramsCopy = params
                    opt.step(params: &paramsCopy, grads: grads)
                    let newW = paramsCopy[0]
                    let newB = (projB != nil) ? paramsCopy[1] : nil
                    enc.setProjParams(weight: newW, bias: newB)
                    enc.invalidateProjectionCache()
                    // Post-update metrics using project-only on same pooled
                    let outPost = enc.projectOnly(pooled)
                    let msePost = Losses.mseRowwise(outPost, t)
                    let cosPost = Losses.cosineSimilarityRowwise(outPost, t)
                    logger.info(String(format: "post-upd: b=%d d=%d MSE=%.6f Cos=%.6f", b, d, msePost.mean, cosPost.mean), category: Logger.Category.dataset)
                    offset += take
                }
            }
        }
        if seen > 0 {
            let avgMSE = totalMSE / Double(seen)
            let avgCos = totalCos / Double(seen)
            logger.info(String(format: "epoch %d summary: samples=%d avgMSE=%.6f avgCos=%.6f", epoch+1, seen, avgMSE, avgCos), category: Logger.Category.dataset)
        }
    }
}

do { try run() } catch {
    Logger.shared.error("EFTrain failed: \(error)", category: Logger.Category.dataset)
    exit(1)
}
