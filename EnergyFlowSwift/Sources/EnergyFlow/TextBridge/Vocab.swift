import Foundation

// Fixed IDs according to docs/textcube_tokenizer_plan.md (subset needed for Phase 1)
public enum Vocab: Int {
    case PAD = 0
    case BOS = 1
    case EOS = 2
    case SEP = 3

    case HDR_BOS = 4
    case HDR_END = 5

    case BODY_BOS = 6
    case BODY_END = 7

    case LP = 8
    case RP = 9
    case COMMA = 10

    case IMPL = 11 // reserved (not used in Phase 1)

    case LT = 12
    case LE = 13
    case EQ = 14
    case NE = 15
    case GT = 16
    case GE = 17

    case NUM_BOS = 18
    case NUM_END = 19
    case SIGN_MINUS = 20
    case SIGN_PLUS = 21

    // DIGIT_0 = 30 ... DIGIT_9 = 39
    // MAP_PAIR = 40
    // BYTES_BOS = 41, BYTES_END = 42
    // BYTE_0 = 43 ... BYTE_255 = 298
    // IDX_BOS = 299, IDX_END = 300
    // IDX_0 = 301 ... IDX_63 = 364
}

public enum VocabRanges {
    public static let DIGIT_0 = 30
    public static let DIGIT_9 = 39

    public static let MAP_PAIR = 40
    public static let BYTES_BOS = 41
    public static let BYTES_END = 42

    public static let BYTE_0 = 43
    public static let BYTE_255 = 298

    public static let IDX_BOS = 299
    public static let IDX_END = 300
    public static let IDX_0 = 301
    public static let IDX_63 = 364
}

public enum VocabUtil {
    public static func digitID(_ d: Int) -> Int {
        precondition((0...9).contains(d))
        return VocabRanges.DIGIT_0 + d
    }
    public static func idxID(_ i: Int) -> Int {
        precondition((0...63).contains(i))
        return VocabRanges.IDX_0 + i
    }
    public static func byteID(_ b: UInt8) -> Int {
        return VocabRanges.BYTE_0 + Int(b)
    }

    public static func idToDigit(_ id: Int) -> Int? {
        guard id >= VocabRanges.DIGIT_0 && id <= VocabRanges.DIGIT_9 else { return nil }
        return id - VocabRanges.DIGIT_0
    }
    public static func idToIdx(_ id: Int) -> Int? {
        guard id >= VocabRanges.IDX_0 && id <= VocabRanges.IDX_63 else { return nil }
        return id - VocabRanges.IDX_0
    }
    public static func idToByte(_ id: Int) -> UInt8? {
        guard id >= VocabRanges.BYTE_0 && id <= VocabRanges.BYTE_255 else { return nil }
        return UInt8(id - VocabRanges.BYTE_0)
    }
}