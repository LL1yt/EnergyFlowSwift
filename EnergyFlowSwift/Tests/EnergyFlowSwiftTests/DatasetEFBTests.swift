import XCTest
import EnergyFlow

final class DatasetEFBTests: XCTestCase {
    func testLoadEFBIfPresent() throws {
        // Resolve repo root from this file path, then AA parent and dataset path
        // This test is optional: it will be skipped if the dataset file is missing.
        let thisFile = URL(fileURLWithPath: #filePath)
        let pkgRoot = thisFile
            .deletingLastPathComponent() // .../EnergyFlowSwift/Tests/EnergyFlowSwiftTests
            .deletingLastPathComponent() // .../EnergyFlowSwift/Tests
            .deletingLastPathComponent() // .../EnergyFlowSwift
        let aaRoot = pkgRoot.deletingLastPathComponent() // .../AA
        let efbPath = aaRoot.appendingPathComponent("data/embeddings/15k_out.efb").path

        guard FileManager.default.fileExists(atPath: efbPath) else {
            throw XCTSkip("Dataset not found at \(efbPath). Skipping.")
        }

        let ds = try SimpleJSONLDataset(path: efbPath)
        XCTAssertTrue(ds.hasTokens, "EFB should provide tokens")
        XCTAssertGreaterThan(ds.embeddingDim, 0)
        XCTAssertGreaterThan(ds.count(), 0)

        let it = ds.batches(batchSize: 32, dropLast: false, padTokens: true, padTokenID: 0)
        guard let batch = it.next() else {
            XCTFail("No batch produced")
            return
        }
        switch batch {
        case .tokens(let ids, let mask, let targets):
            XCTAssertEqual(ids.count, mask.count)
            XCTAssertEqual(ids.count, targets.count)
            XCTAssertGreaterThan(ids.count, 0)
            // All sequences padded to equal length
            let seqLens = ids.map { $0.count }
            XCTAssertTrue(seqLens.allSatisfy { $0 == seqLens.first })
            // Target dims equal embeddingDim
            XCTAssertTrue(targets.allSatisfy { $0.count == ds.embeddingDim })
            // Mask is 0/1
            for row in mask {
                for v in row { XCTAssertTrue(v == 0 || v == 1) }
            }
        case .text:
            XCTFail("EFB loader should produce token batches, not text")
        }
    }
}