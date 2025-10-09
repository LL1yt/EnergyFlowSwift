import Foundation

enum EmbeddingMetalLibrary {
    static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void embedding_gather_f32(
        const device int*    ids   [[buffer(0)]],
        const device float*  W     [[buffer(1)]],
        device float*        out   [[buffer(2)]],
        constant int&        B     [[buffer(3)]],
        constant int&        L     [[buffer(4)]],
        constant int&        V     [[buffer(5)]],
        constant int&        D     [[buffer(6)]],
        uint gid [[thread_position_in_grid]])
    {
        int N = B * L * D;
        if ((int)gid >= N) return;
        int rem = (int)gid;
        int b = rem / (L * D);
        rem = rem % (L * D);
        int t = rem / D;
        int d = rem % D;
        int token = ids[b * L + t];
        if (token < 0 || token >= V) token = 0;
        out[gid] = W[token * D + d];
    }
    """
}
