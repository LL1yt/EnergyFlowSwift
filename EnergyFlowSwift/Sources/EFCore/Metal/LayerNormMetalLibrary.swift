import Foundation

enum LayerNormMetalLibrary {
    static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void ln_compute_stats_fp16(
        const device half*  x     [[buffer(0)]],
        device float*       mean  [[buffer(1)]],
        device float*       invst [[buffer(2)]],
        constant int&       N     [[buffer(3)]],
        constant int&       D     [[buffer(4)]],
        constant float&     eps   [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        int base = (int)gid * D;
        float s = 0.0f;
        float ss = 0.0f;
        for (int j = 0; j < D; ++j) {
            float v = (float)x[base + j];
            s  += v;
            ss += v * v;
        }
        float m = s / (float)D;
        float var = ss / (float)D - m * m;
        mean[gid] = m;
        invst[gid] = rsqrt(var + eps);
    }

    kernel void ln_normalize_affine_fp16(
        const device half*   x     [[buffer(0)]],
        device half*         y     [[buffer(1)]],
        const device float*  gamma [[buffer(2)]],
        const device float*  beta  [[buffer(3)]],
        const device float*  mean  [[buffer(4)]],
        const device float*  invst [[buffer(5)]],
        constant int&        N     [[buffer(6)]],
        constant int&        D     [[buffer(7)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        int base = (int)gid * D;
        float m = mean[gid];
        float is = invst[gid];
        for (int j = 0; j < D; ++j) {
            float vx = (float)x[base + j];
            float nrm = (vx - m) * is;
            float out = nrm * gamma[j] + beta[j];
            y[base + j] = (half)out;
        }
    }

    kernel void ln_bwd_row_sums_fp16(
        const device half*   x      [[buffer(0)]],
        const device half*   g      [[buffer(1)]],
        const device float*  gamma  [[buffer(2)]],
        device float*        mean    [[buffer(3)]],
        device float*        invst   [[buffer(4)]],
        device float*        sumG    [[buffer(5)]],
        device float*        sumGXh  [[buffer(6)]],
        constant int&        N       [[buffer(7)]],
        constant int&        D       [[buffer(8)]],
        constant float&      eps     [[buffer(9)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        int base = (int)gid * D;
        float s = 0.0f;
        float ss = 0.0f;
        for (int j = 0; j < D; ++j) {
            float v = (float)x[base + j];
            s  += v;
            ss += v * v;
        }
        float m = s / (float)D;
        float var = ss / (float)D - m * m;
        float is = rsqrt(var + eps);
        mean[gid] = m;
        invst[gid] = is;
        float sg = 0.0f;
        float sgx = 0.0f;
        for (int j = 0; j < D; ++j) {
            float v = (float)x[base + j];
            float xh = (v - m) * is;
            float gy = (float)g[base + j] * gamma[j];
            sg += gy;
            sgx += gy * xh;
        }
        sumG[gid] = sg;
        sumGXh[gid] = sgx;
    }

    kernel void ln_bwd_dx_fp16(
        const device half*   x      [[buffer(0)]],
        const device half*   g      [[buffer(1)]],
        const device float*  gamma  [[buffer(2)]],
        const device float*  mean   [[buffer(3)]],
        const device float*  invst  [[buffer(4)]],
        const device float*  sumG   [[buffer(5)]],
        const device float*  sumGXh [[buffer(6)]],
        device half*         dx     [[buffer(7)]],
        constant int&        N      [[buffer(8)]],
        constant int&        D      [[buffer(9)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        int base = (int)gid * D;
        float m = mean[gid];
        float is = invst[gid];
        float sg = sumG[gid];
        float sgx = sumGXh[gid];
        float invD = 1.0f / (float)D;
        for (int j = 0; j < D; ++j) {
            float v = (float)x[base + j];
            float xh = (v - m) * is;
            float gy = (float)g[base + j] * gamma[j];
            float term = (float)D * gy - sg - xh * sgx;
            float dxi = invD * is * term;
            dx[base + j] = (half)dxi;
        }
    }

    kernel void ln_bwd_dgamma_dbeta_f32(
        const device half*   x      [[buffer(0)]],
        const device half*   g      [[buffer(1)]],
        const device float*  mean   [[buffer(2)]],
        const device float*  invst  [[buffer(3)]],
        device float*        dGamma [[buffer(4)]],
        device float*        dBeta  [[buffer(5)]],
        constant int&        N      [[buffer(6)]],
        constant int&        D      [[buffer(7)]],
        uint gid [[thread_position_in_grid]])
    {
        int j = (int)gid;
        if (j >= D) return;
        float dgb = 0.0f;
        float db = 0.0f;
        for (int n = 0; n < N; ++n) {
            int idx = n * D + j;
            float v = (float)x[idx];
            float m = mean[n];
            float is = invst[n];
            float xh = (v - m) * is;
            float gy = (float)g[idx];
            dgb += gy * xh;
            db += gy;
        }
        dGamma[j] = dgb;
        dBeta[j] = db;
    }
    """
}
