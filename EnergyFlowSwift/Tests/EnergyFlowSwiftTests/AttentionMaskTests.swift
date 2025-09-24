import XCTest
@testable import PyTorchSwift
@testable import EFCore

final class AttentionMaskTests: XCTestCase {
    // Build a simple input tensor [B,L,H] with deterministic values
    private func makeInput(b: Int, l: Int, h: Int) -> Tensor {
        var data: [Float] = []
        data.reserveCapacity(b*l*h)
        for bi in 0..<b {
            for li in 0..<l {
                for hi in 0..<h {
                    data.append(Float(bi*1000 + li*10 + hi))
                }
            }
        }
        return Tensor(shape: [b,l,h], data: data)
    }

    func testMaskedQueriesBecomeZero_SingleLayer() {
        let b = 2, l = 4, h = 8
        let x = makeInput(b: b, l: l, h: h)
        // Mask: first two positions valid, last two padded
        let mask: [[Int]] = Array(repeating: [1,1,0,0], count: b)
        // Encoder: 1 layer, heads=2, ff=16
        let layer = TransformerEncoderLayer(hidden: h, ffDim: 16, numHeads: 2, seed: 123)
        let out = layer.forward(x, mask: mask)
        XCTAssertEqual(out.shape, [b,l,h])
        for bi in 0..<b {
            for li in 0..<l {
                let isPadded = mask[bi][li] == 0
                let base = (bi*l + li) * h
                var allZero = true
                for d in 0..<h {
                    if out.data[base + d] != 0 { allZero = false; break }
                }
                if isPadded {
                    XCTAssertTrue(allZero, "Expected zero vector at padded query [\(bi),\(li)]")
                } else {
                    XCTAssertFalse(allZero, "Expected non-zero vector at valid query [\(bi),\(li)]")
                }
            }
        }
    }
}
