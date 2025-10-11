import Foundation

enum MetricsMetalLibrary {
    static let source: String = """
    #include <metal_stdlib>
    #include <metal_atomic>
    using namespace metal;

    kernel void kd_metrics_reduce_f32(
        const device float* student [[buffer(0)]],
        const device float* teacher [[buffer(1)]],
        device atomic_float* accum  [[buffer(2)]],
        constant int&       B       [[buffer(3)]],
        constant int&       D       [[buffer(4)]],
        constant float&     eps     [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= B) {
            return;
        }
        int base = (int)gid * D;
        float mse = 0.0f;
        float dot = 0.0f;
        float ns = 0.0f;
        float nt = 0.0f;
        for (int i = 0; i < D; ++i) {
            float s = student[base + i];
            float t = teacher[base + i];
            float diff = s - t;
            mse += diff * diff;
            dot += s * t;
            ns += s * s;
            nt += t * t;
        }
        mse /= (float)D;
        float denom = sqrt(max(ns, eps)) * sqrt(max(nt, eps));
        float cosVal = (denom > 0.0f) ? (dot / denom) : 0.0f;
        atomic_fetch_add_explicit(&accum[0], mse, memory_order_relaxed);
        atomic_fetch_add_explicit(&accum[1], cosVal, memory_order_relaxed);
    }

    kernel void cross_entropy_mean_logits_f32(
        const device float* logits  [[buffer(0)]],
        const device int*   targets [[buffer(1)]],
        device atomic_float* accum  [[buffer(2)]],
        constant int&       samples [[buffer(3)]],
        constant int&       vocab   [[buffer(4)]],
        constant float&     eps     [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
    {
        if ((int)gid >= samples) {
            return;
        }
        int base = (int)gid * vocab;
        float maxLogit = -FLT_MAX;
        for (int v = 0; v < vocab; ++v) {
            float lv = logits[base + v];
            if (lv > maxLogit) { maxLogit = lv; }
        }
        float sumExp = 0.0f;
        for (int v = 0; v < vocab; ++v) {
            sumExp += exp(logits[base + v] - maxLogit);
        }
        int target = targets[gid];
        if (target < 0 || target >= vocab) {
            return;
        }
        float logZ = maxLogit + log(sumExp + eps);
        float logProb = logits[base + target] - logZ;
        atomic_fetch_add_explicit(&accum[0], -logProb, memory_order_relaxed);
        atomic_fetch_add_explicit(&accum[1], 1.0f, memory_order_relaxed);
    }
    """
}
