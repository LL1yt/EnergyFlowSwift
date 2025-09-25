import XCTest
@testable import EnergyFlow

final class TextCubeTokenizerPhase1Tests: XCTestCase {
    func testRoundTripSimpleChains() throws {
        let tok = TextCubeTokenizer()
        let samples = [
            "(x<y, y<z)",
            "(a<=b, b<=c, a<=c)",
            "(u=v, v=w, u=w)",
            "(x!=y)",
            "(x>y, y>=z)"
        ]
        for s in samples {
            let enc = try tok.encode(s)
            let back = try tok.decode(enc)
            // Logs for non-experts: show example, ids head, mask header span
            let idsHead = enc.ids.prefix(64).map { Int($0) }
            let headerMaskCount = enc.mask.prefix(enc.len).filter { $0 == 1 }.count
            print("[LOG] Input=\(s)")
            print("[LOG] Enc.len=\(enc.len), headerMaskCount(within-len)=\(headerMaskCount)")
            print("[LOG] Enc.ids[0..64)=\(idsHead)")
            print("[LOG] Decoded=\(back)")
            XCTAssertEqual(s, back)
        }
    }

    func testNumbersAndSigns() throws {
        let tok = TextCubeTokenizer()
        let samples = [
            "(x<0)", "(x<-1)", "(x<+12)", "(a=001)" // note: leading zeros allowed lexically
        ]
        for s in samples {
            let enc = try tok.encode(s)
            let back = try tok.decode(enc)
            print("[LOG] NUM Input=\(s) -> Decoded=\(back)")
            XCTAssertEqual(s, back)
        }
    }

    func testScopeExamplesHelp() throws {
        // Showcase of supported grammar scope for onboarding
        // Supported:
        // - Block: (stmt, stmt, ...)
        // - Stmt: <term> <rel> <term>
        // - Term: identifier | signed integer number
        // - Rel: <, <=, =, !=, >, >=
        let tok = TextCubeTokenizer()
        let example = "(foo<bar, foo<=baz, a=b, a!=c, z>t, z>=t)"
        let enc = try tok.encode(example)
        let back = try tok.decode(enc)
        print("[LOG] Scope example: \n  input=\(example)\n  decoded=\(back)\n  note=Identifiers become V0..Vn in header mapping; body references them via IDX tokens; numbers use NUM_BOS/END with digits.")
        XCTAssertEqual(example, back)
    }
}