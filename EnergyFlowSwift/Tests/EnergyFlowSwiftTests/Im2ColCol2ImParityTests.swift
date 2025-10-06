import XCTest
@testable import EFCore

final class Im2ColCol2ImParityTests: XCTestCase {
    func testIm2ColCol2ImParitySmall() throws {
        let B = 2, L = 8, Cin = 3, K = 3, dil = 2
        // Random X
        var xData: [Float] = []
        for _ in 0..<(B*L*Cin) { xData.append(Float.random(in: -1...1)) }
        let X = Tensor(shape: [B, L, Cin], data: xData)
        // GPU im2col
        let XcolGPU = try Im2ColCol2ImGPU.im2col(X: X, B: B, L: L, Cin: Cin, K: K, dilation: dil)
        // CPU im2col
        let rows = B * L
        let colsX = Cin * K
        var XcolCPU = Tensor.zeros([rows, colsX])
        for b in 0..<B {
            for t in 0..<L {
                let r = b * L + t
                let rowBase = r * colsX
                for i in 0..<Cin {
                    for k in 0..<K {
                        let ti = t - k * dil
                        let dst = rowBase + i * K + k
                        if ti < 0 { XcolCPU.data[dst] = 0 } else { XcolCPU.data[dst] = X.data[(b * L + ti) * Cin + i] }
                    }
                }
            }
        }
        // Compare
        XCTAssertEqual(XcolCPU.shape, XcolGPU.shape)
        for i in 0..<(rows*colsX) {
            XCTAssertLessThan(abs(XcolCPU.data[i] - XcolGPU.data[i]), 1e-5)
        }
        // Now test col2im parity
        // Random dXcol
        var dxcData: [Float] = []
        for _ in 0..<(rows*colsX) { dxcData.append(Float.random(in: -1...1)) }
        let dXcol = Tensor(shape: [rows, colsX], data: dxcData)
        let dXGPU = try Im2ColCol2ImGPU.col2im(dXcol: dXcol, B: B, L: L, Cin: Cin, K: K, dilation: dil)
        var dXCPU = Tensor.zeros([B, L, Cin])
        for b in 0..<B {
            for t in 0..<L {
                let r = b * L + t
                let rowBase = r * colsX
                for i in 0..<Cin {
                    for k in 0..<K {
                        let ti = t - k * dil
                        if ti < 0 { continue }
                        let val = dXcol.data[rowBase + i * K + k]
                        dXCPU.data[(b * L + ti) * Cin + i] += val
                    }
                }
            }
        }
        XCTAssertEqual(dXCPU.shape, dXGPU.shape)
        for i in 0..<(B*L*Cin) {
            XCTAssertLessThan(abs(dXCPU.data[i] - dXGPU.data[i]), 1e-5)
        }
    }
}