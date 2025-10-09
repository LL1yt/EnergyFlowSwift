import Foundation

enum GELUMetalLibrary {
    static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void gelu_tanh_fp16(
        const device half*  x   [[buffer(0)]],
        device half*        y   [[buffer(1)]],
        constant int&       N   [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        float v = (float)x[gid];
        const float c = 0.7978845608028654f;
        float v3 = v * v * v;
        float u = c * (v + 0.044715f * v3);
        float t = tanh(u);
        float out = 0.5f * v * (1.0f + t);
        y[gid] = (half)out;
    }

    kernel void gelu_tanh_bwd_fp16(
        const device half*  x    [[buffer(0)]],
        const device half*  dy   [[buffer(1)]],
        device half*        dx   [[buffer(2)]],
        constant int&       N    [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= N) return;
        float v = (float)x[gid];
        float gy = (float)dy[gid];
        const float c = 0.7978845608028654f;
        const float a = 0.044715f;
        float v2 = v * v;
        float v3 = v2 * v;
        float u = c * (v + a * v3);
        float th = tanh(u);
        float sech2 = 1.0f - th * th;
        float du_dx = c * (1.0f + 3.0f * a * v2);
        float dgelu = 0.5f * (1.0f + th) + 0.5f * v * sech2 * du_dx;
        dx[gid] = (half)(gy * dgelu);
    }
    """
}
