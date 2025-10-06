import XCTest
@testable import EFCore

final class OptimUtilsTests: XCTestCase {
    func testWarmupCosineScheduler() throws {
        let base: Float = 1e-3
        let minLR: Float = 1e-5
        let warm = 10
        let decay = 90
        // step 0 -> ~base/10
        let lr0 = LRSchedulers.warmupCosine(baseLR: base, minLR: minLR, warmupSteps: warm, decaySteps: decay, step: 0)
        XCTAssertGreaterThan(lr0, 0)
        XCTAssertLessThan(lr0, base)
        // step warm-1 -> ~base
        let lrW = LRSchedulers.warmupCosine(baseLR: base, minLR: minLR, warmupSteps: warm, decaySteps: decay, step: warm-1)
        XCTAssertGreaterThan(lrW, lr0)
        // end of decay -> minLR
        let lrEnd = LRSchedulers.warmupCosine(baseLR: base, minLR: minLR, warmupSteps: warm, decaySteps: decay, step: warm+decay)
        XCTAssertEqual(lrEnd, minLR, accuracy: 1e-7)
    }

    func testGradClipGlobalL2() throws {
        let t1 = Tensor(shape: [2], data: [3, 4]) // norm 5
        let t2 = Tensor(shape: [1], data: [0])
        var list = [t1, t2]
        let scale = GradClip.clipGlobalL2Norm(tensors: &list, maxNorm: 1)
        XCTAssertLessThan(scale, 1.0)
        let n = sqrt(list[0].data[0]*list[0].data[0] + list[0].data[1]*list[0].data[1] + list[1].data[0]*list[1].data[0])
        XCTAssertEqual(n, 1.0, accuracy: 1e-5)
    }
}