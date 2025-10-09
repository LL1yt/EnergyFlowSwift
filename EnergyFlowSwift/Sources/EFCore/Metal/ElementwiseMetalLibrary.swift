import Foundation

enum ElementwiseMetalLibrary {
    static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    // y += x (elementwise)
    kernel void residual_add_f32(
        device float*       y   [[buffer(0)]],
        const device float* x   [[buffer(1)]],
        constant int&       N   [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        y[gid] += x[gid];
    }

    // y[b,t,d] += a[b,d]
    kernel void add_broadcast_2d_into_3d_f32(
        device float*        y    [[buffer(0)]],
        const device float*  a    [[buffer(1)]],
        constant int&        B    [[buffer(2)]],
        constant int&        L    [[buffer(3)]],
        constant int&        D    [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        int N = B * L * D;
        if ((int)gid >= N) return;
        int rem = (int)gid;
        int b = rem / (L * D);
        rem = rem % (L * D);
        int t = rem / D;
        int d = rem % D;
        (void)t; // unused, but kept for clarity
        y[gid] += a[b * D + d];
    }

    // Zero masked positions: y[b,t,:] = 0 if mask[b][t] == 0
    kernel void mask_zero_f32(
        device float*        y    [[buffer(0)]],
        const device int*    mask [[buffer(1)]],
        constant int&        B    [[buffer(2)]],
        constant int&        L    [[buffer(3)]],
        constant int&        D    [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        int N = B * L * D;
        if ((int)gid >= N) return;
        int rem = (int)gid;
        int b = rem / (L * D);
        rem = rem % (L * D);
        int t = rem / D;
        int m = mask[b * L + t];
        if (m == 0) {
            y[gid] = 0.0f;
        }
    }

    // Masked mean over L: y[b,h] = sum_t mask[b,t]*x[b,t,h] / max(sum mask[b,*], eps)
    kernel void masked_mean_f32(
        const device float*  x     [[buffer(0)]],
        const device int*    mask  [[buffer(1)]],
        device float*        y     [[buffer(2)]],
        constant int&        B     [[buffer(3)]],
        constant int&        L     [[buffer(4)]],
        constant int&        H     [[buffer(5)]],
        constant float&      eps   [[buffer(6)]],
        uint gid [[thread_position_in_grid]])
    {
        int total = B * H;
        if ((int)gid >= total) return;
        int b = (int)gid / H;
        int h = (int)gid % H;
        float sum = 0.0f;
        float denom = 0.0f;
        for (int t = 0; t < L; ++t) {
            int m = mask[b * L + t];
            if (m != 0) {
                int idx = (b * L + t) * H + h;
                sum += x[idx];
                denom += 1.0f;
            }
        }
        denom = max(denom, eps);
        y[b * H + h] = sum / denom;
    }

    // Backward: dx[b,t,h] = (mask[b,t]/denom_b) * dY[b,h]
    kernel void masked_mean_bwd_f32(
        const device float*  dY    [[buffer(0)]],
        const device int*    mask  [[buffer(1)]],
        device float*        dX    [[buffer(2)]],
        constant int&        B     [[buffer(3)]],
        constant int&        L     [[buffer(4)]],
        constant int&        H     [[buffer(5)]],
        constant float&      eps   [[buffer(6)]],
        uint gid [[thread_position_in_grid]])
    {
        int N = B * L * H;
        if ((int)gid >= N) return;
        int rem = (int)gid;
        int b = rem / (L * H);
        rem = rem % (L * H);
        int t = rem / H;
        int h = rem % H;
        int m = mask[b * L + t];
        float denom = 0.0f;
        for (int tt = 0; tt < L; ++tt) { denom += (mask[b * L + tt] != 0) ? 1.0f : 0.0f; }
        denom = max(denom, eps);
        int dyIndex = b * H + h;
        int dxIndex = (b * L + t) * H + h;
        float val = (m != 0) ? (dY[dyIndex] / denom) : 0.0f;
        dX[dxIndex] = val;
    }
    """
}
