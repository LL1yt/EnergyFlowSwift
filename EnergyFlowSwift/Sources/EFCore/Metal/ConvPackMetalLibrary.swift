import Foundation

enum ConvPackMetalLibrary {
    static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void pack_w_to_col_f32(
        const device float*  W     [[buffer(0)]],
        device float*        Wcol  [[buffer(1)]],
        constant int&        Cout  [[buffer(2)]],
        constant int&        Cin   [[buffer(3)]],
        constant int&        K     [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        int total = Cout * Cin * K;
        if ((int)gid >= total) return;
        int rem = (int)gid;
        int o = rem / (Cin * K);
        rem = rem % (Cin * K);
        int i = rem / K;
        int k = rem % K;
        int src = (o * Cin + i) * K + k;
        int dst = o * (Cin * K) + (i * K + k);
        Wcol[dst] = W[src];
    }

    kernel void unpack_dwcol_f32(
        const device float*  dWcol [[buffer(0)]],
        device float*        dW    [[buffer(1)]],
        constant int&        Cout  [[buffer(2)]],
        constant int&        Cin   [[buffer(3)]],
        constant int&        K     [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        int total = Cout * Cin * K;
        if ((int)gid >= total) return;
        int rem = (int)gid;
        int o = rem / (Cin * K);
        rem = rem % (Cin * K);
        int i = rem / K;
        int k = rem % K;
        int src = o * (Cin * K) + (i * K + k);
        int dst = (o * Cin + i) * K + k;
        dW[dst] = dWcol[src];
    }
    """
}
