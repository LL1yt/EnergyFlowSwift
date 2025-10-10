import Foundation

enum Im2ColCol2ImMetalLibrary {
    static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void im2col_1d_causal_f32(
        const device float* X          [[buffer(0)]],
        device float*       Xcol       [[buffer(1)]],
        constant int&       B          [[buffer(2)]],
        constant int&       L          [[buffer(3)]],
        constant int&       Cin        [[buffer(4)]],
        constant int&       K          [[buffer(5)]],
        constant int&       dilation   [[buffer(6)]],
        uint                gid        [[thread_position_in_grid]])
    {
        int rows = B * L;
        int colsX = Cin * K;
        int total = rows * colsX;
        if ((int)gid >= total) return;
        int row = gid / colsX;
        int col = gid % colsX;
        int i = col / K;
        int k = col % K;
        int b = row / L;
        int t = row % L;
        int dil = max(dilation, 1);
        int srcT = t - k * dil;
        float val = 0.0f;
        if (srcT >= 0) {
            int src = (b * L + srcT) * Cin + i;
            val = X[src];
        }
        Xcol[gid] = val;
    }

    kernel void col2im_1d_causal_f32(
        const device float* dXcol     [[buffer(0)]],
        device float*       dX        [[buffer(1)]],
        constant int&       B         [[buffer(2)]],
        constant int&       L         [[buffer(3)]],
        constant int&       Cin       [[buffer(4)]],
        constant int&       K         [[buffer(5)]],
        constant int&       dilation  [[buffer(6)]],
        uint                gid       [[thread_position_in_grid]])
    {
        int total = B * L * Cin;
        if ((int)gid >= total) return;
        int rem = (int)gid;
        int b = rem / (L * Cin);
        rem = rem % (L * Cin);
        int t = rem / Cin;
        int i = rem % Cin;
        int dil = max(dilation, 1);
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            int srcT = t + k * dil;
            if (srcT < L) {
                int row = b * L + srcT;
                int col = i * K + k;
                int idx = row * (Cin * K) + col;
                acc += dXcol[idx];
            }
        }
        dX[gid] = acc;
    }

    kernel void im2col_1d_causal_f32_to_f16(
        const device float* X          [[buffer(0)]],
        device half*        Xcol       [[buffer(1)]],
        constant int&       B          [[buffer(2)]],
        constant int&       L          [[buffer(3)]],
        constant int&       Cin        [[buffer(4)]],
        constant int&       K          [[buffer(5)]],
        constant int&       dilation   [[buffer(6)]],
        constant int&       rowStride  [[buffer(7)]],
        constant int&       colsTotal  [[buffer(8)]],
        uint                gid        [[thread_position_in_grid]])
    {
        int rows = B * L;
        int colsX = Cin * K;
        int total = rows * colsX;
        if ((int)gid >= total) return;
        int row = gid / colsX;
        int col = gid % colsX;
        int i = col / K;
        int k = col % K;
        int b = row / L;
        int t = row % L;
        int dil = max(dilation, 1);
        int srcT = t - k * dil;
        float val = 0.0f;
        if (srcT >= 0) {
            int src = (b * L + srcT) * Cin + i;
            val = X[src];
        }
        int dst = row * rowStride + col;
        Xcol[dst] = (half)val;
    }

    kernel void fill_bias_col_fp16(
        device half*        Xcol       [[buffer(0)]],
        constant int&       rows       [[buffer(1)]],
        constant int&       rowStride  [[buffer(2)]],
        constant int&       biasIndex  [[buffer(3)]],
        uint                gid        [[thread_position_in_grid]])
    {
        if ((int)gid >= rows) return;
        int row = (int)gid;
        int idx = row * rowStride + biasIndex;
        Xcol[idx] = (half)1.0f;
    }
    """
}
