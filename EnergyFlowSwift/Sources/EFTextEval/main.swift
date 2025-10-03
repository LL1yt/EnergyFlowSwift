import Foundation
import EnergyFlow
import EFCore

// CLI: EFTextEval --data /path/to/dataset.(jsonl|efb) [--batch-size 16] [--layers 2] [--heads 8] [--max-length 128]
// Env for logs: EF_LOG_STDOUT=1 EF_LOG_LEVEL=debug

struct Args {
    var dataPath: String
    var batchSize: Int = 16
    var maxLength: Int = 128   // 0 means auto from dataset
    var lengthCap: Int = 256   // cap for auto length (keeps eval fast); 0 means no cap
    var maxBatches: Int = 0    // 0 means all batches
    var microBatch: Int = 32   // internal split for progress and latency; 0 => same as batchSize
}

func parseArgs() -> Args? {
    var a = Args(dataPath: "")
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let key = it.next() {
        switch key {
        case "--data":
            if let v = it.next() { a.dataPath = v }
        case "--batch-size":
            if let v = it.next(), let n = Int(v) { a.batchSize = n }
        case "--max-length":
            if let v = it.next(), let n = Int(v) { a.maxLength = n }
        case "--length-cap":
            if let v = it.next(), let n = Int(v) { a.lengthCap = n }
        case "--max-batches":
            if let v = it.next(), let n = Int(v) { a.maxBatches = n }
        case "--micro-batch":
            if let v = it.next(), let n = Int(v) { a.microBatch = n }
        case "-h", "--help":
            return nil
        default:
            // ignore unknown
            continue
        }
    }
    if a.dataPath.isEmpty { return nil }
    return a
}

func usage() {
    print("Usage: EFTextEval --data /path/to/data.jsonl|.efb [--batch-size 16] [--max-length 128|0(auto)] [--length-cap 256] [--max-batches 0(all)] [--micro-batch 32]")
}

func mse(_ y: [Float], _ t: [Float]) -> Float {
    precondition(y.count == t.count)
    var acc: Float = 0
    for i in 0..<y.count { let d = y[i] - t[i]; acc += d*d }
    return acc / Float(y.count)
}

func batchMSE(_ y: [[Float]], _ t: [[Float]]) -> Float {
    precondition(y.count == t.count)
    var acc: Float = 0
    for i in 0..<y.count { acc += mse(y[i], t[i]) }
    return acc / Float(max(y.count, 1))
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

func run() throws {
    guard let args = parseArgs() else { usage(); return }
    let logger = Logger.shared
    logger.info("EFTextEval start: data=\(args.dataPath) batchSize=\(args.batchSize) maxLen=\(args.maxLength)", category: Logger.Category.dataset)

    let ds = try SimpleJSONLDataset(path: args.dataPath)
    logger.info("dataset: count=\(ds.count()) targetDim=\(ds.embeddingDim) hasTokens=\(ds.hasTokens)", category: Logger.Category.dataset)

    // Auto-select maxLength if requested
    var maxLen = args.maxLength
    if maxLen <= 0 {
        if ds.hasTokens {
            var lengths: [Int] = []
            lengths.reserveCapacity(ds.count())
            for s in ds.samples {
                if let ids = s.inputIDs { lengths.append(ids.count) }
            }
            if lengths.isEmpty {
                maxLen = 128
                logger.warn("no token lengths available; fallback maxLen=\\(maxLen)", category: Logger.Category.dataset)
            } else {
                lengths.sort()
                func pct(_ p: Double) -> Int {
                    let idx = Int(Double(lengths.count - 1) * p)
                    return lengths[max(0, min(idx, lengths.count - 1))]
                }
                let minL = lengths.first ?? 0
                let p50 = pct(0.50), p95 = pct(0.95), p99 = pct(0.99), maxL = lengths.last ?? 0
                // Choose conservative cap at p95, then apply user-provided lengthCap (default 256), and global 512 safety cap
                var chosen = p95
                if args.lengthCap > 0 { chosen = min(chosen, args.lengthCap) }
                chosen = min(chosen, 512)
                maxLen = chosen
                logger.info("length stats: min=\(minL) p50=\(p50) p95=\(p95) p99=\(p99) max=\(maxL) -> chosen maxLen=\(maxLen) (cap=\(args.lengthCap))", category: Logger.Category.dataset)
            }
        } else {
            maxLen = 128
            logger.warn("dataset has no tokens; using default maxLen=\\(maxLen)", category: Logger.Category.dataset)
        }
    }

    // Configure encoder to output same dim as targets
    let cfg = TextToCubeEncoderConfig(
        hiddenDim: 256,
        maxLength: maxLen,
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

    // Progress helpers
    let t0 = Date()
    var totalMSE: Double = 0
    var seen: Int = 0
    var batchIdx = 0
    let totalSamples = ds.count()
    let expectedSamples: Int = {
        if args.maxBatches > 0 { return min(totalSamples, args.batchSize * args.maxBatches) }
        return totalSamples
    }()
    logger.info("eval plan: totalSamples=\(totalSamples) expected=\(expectedSamples) batchSize=\(args.batchSize) micro=\(args.microBatch > 0 ? args.microBatch : args.batchSize)", category: Logger.Category.dataset)

    func logProgress() {
        let dt = Date().timeIntervalSince(t0)
        let rate = dt > 0 ? Double(seen) / dt : 0
        logger.info(String(format: "progress: seen=%d/ %d (%.1f%%) elapsed=%.2fs rate=%.1f samples/s", seen, expectedSamples, 100.0 * (expectedSamples > 0 ? Double(seen) / Double(expectedSamples) : 0), dt, rate), category: Logger.Category.dataset)
    }

    for batch in ds.batches(batchSize: args.batchSize, dropLast: false, padTokens: true) {
        batchIdx += 1
        if args.maxBatches > 0 && batchIdx > args.maxBatches {
            logger.info("maxBatches reached (\(args.maxBatches)) — stopping early", category: Logger.Category.dataset)
            break
        }

        switch batch {
        case let .tokens(inputIDs, attentionMask, targets):
            let B = inputIDs.count
            let micro = args.microBatch > 0 ? args.microBatch : B
            let numChunks = (B + micro - 1) / micro
            logger.info("batch #\(batchIdx) tokens: B=\(B) chunks=\(numChunks) maxLen=\(cfg.maxLength)", category: Logger.Category.dataset)
            var offset = 0
            for c in 0..<numChunks {
                let take = min(micro, B - offset)
                let idsChunk = Array(inputIDs[offset..<(offset+take)])
                let maskChunk = Array(attentionMask[offset..<(offset+take)])
                let tgtChunk = Array(targets[offset..<(offset+take)])
                let out = enc.encodeTokens(inputIDs: idsChunk, attentionMask: maskChunk)
                let b = out.shape[0], d = out.shape[1]
                var pred: [[Float]] = Array(repeating: Array(repeating: 0, count: d), count: b)
                for bi in 0..<b { for di in 0..<d { pred[bi][di] = out.data[bi*d + di] } }
                let m = batchMSE(pred, tgtChunk)
                let cmean = batchCosine(pred, tgtChunk)
                totalMSE += Double(m) * Double(b)
                seen += b
                logger.info("  chunk #\(c+1)/\(numChunks): b=\(b) d=\(d) MSE=\(String(format: "%.6f", m)) Cos=\(String(format: "%.6f", cmean))", category: Logger.Category.dataset)
                logProgress()
                offset += take
            }
        case let .text(texts, targets):
            let B = texts.count
            let micro = args.microBatch > 0 ? args.microBatch : B
            let numChunks = (B + micro - 1) / micro
            logger.info("batch #\(batchIdx) text: B=\(B) chunks=\(numChunks) maxLen=\(cfg.maxLength)", category: Logger.Category.dataset)
            var offset = 0
            for c in 0..<numChunks {
                let take = min(micro, B - offset)
                let txtChunk = Array(texts[offset..<(offset+take)])
                let tgtChunk = Array(targets[offset..<(offset+take)])
                let out = enc.encode(txtChunk)
                let b = out.shape[0], d = out.shape[1]
                var pred: [[Float]] = Array(repeating: Array(repeating: 0, count: d), count: b)
                for bi in 0..<b { for di in 0..<d { pred[bi][di] = out.data[bi*d + di] } }
                let m = batchMSE(pred, tgtChunk)
                let cmean = batchCosine(pred, tgtChunk)
                totalMSE += Double(m) * Double(b)
                seen += b
                logger.info("  chunk #\(c+1)/\(numChunks): b=\(b) d=\(d) MSE=\(String(format: "%.6f", m)) Cos=\(String(format: "%.6f", cmean))", category: Logger.Category.dataset)
                logProgress()
                offset += take
            }
        }
    }

    let finalMSE = seen > 0 ? (totalMSE / Double(seen)) : 0
    // We don't track a running cosine across all chunks without storing all preds; compute per-chunk only.
    logger.info("FINAL: samples=\(seen) avgMSE=\(String(format: "%.6f", finalMSE))", category: Logger.Category.dataset)
}
do {
    try run()
} catch {
    Logger.shared.error("EFTextEval failed: \(error)", category: Logger.Category.dataset)
    exit(1)
}
