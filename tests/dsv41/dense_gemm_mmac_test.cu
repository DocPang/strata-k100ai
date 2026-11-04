// CPU-reference and timing test for the opt-in gfx928 F16 MMAC GEMM.

#include "../../src/prefill/wmma_gemm.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(expr)                                                                 \
    do {                                                                         \
        const hipError_t status_ = (expr);                                       \
        if (status_ != hipSuccess) {                                             \
            std::fprintf(stderr, "%s at %d\n", hipGetErrorString(status_),       \
                         __LINE__);                                               \
            std::exit(2);                                                        \
        }                                                                        \
    } while (0)

static uint16_t half_bits(float value) {
    const __half h = __float2half_rn(value);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static float half_value(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

template <typename T>
static T* device_copy(const std::vector<T>& source) {
    T* dst = nullptr;
    CK(hipMalloc(&dst, source.size() * sizeof(T)));
    CK(hipMemcpy(dst, source.data(), source.size() * sizeof(T), hipMemcpyHostToDevice));
    return dst;
}

template <typename T>
static std::vector<T> host_copy(const T* source, size_t count) {
    std::vector<T> dst(count);
    CK(hipMemcpy(dst.data(), source, count * sizeof(T), hipMemcpyDeviceToHost));
    return dst;
}

static bool check_shape(int T, int N, int K, int ldx, int ldy, bool beta_one,
                        std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> x((size_t) T * ldx), w((size_t) N * K);
    std::vector<float> y0((size_t) T * ldy), y((size_t) T * ldy);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < ldx; ++k)
            x[(size_t) t * ldx + k] = half_bits(k < K ? nd(rng) : 0.0f);
    for (auto& v : w) v = half_bits(0.05f * nd(rng));
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < ldy; ++c)
            y0[(size_t) t * ldy + c] = c < N ? 0.25f * nd(rng) : 17.0f;
    y = y0;

    uint16_t* dx = device_copy(x);
    uint16_t* dw = device_copy(w);
    float* dy = device_copy(y);
    const bool taken = strata_pf_gemm_f16_ld(dx, ldx, dw, K, dy, T, N, K, ldy,
                                             beta_one ? 1.0f : 0.0f);
    CK(hipDeviceSynchronize());
    if (!taken) {
        std::printf("shape T=%d N=%d K=%d was not taken\n", T, N, K);
        CK(hipFree(dx));
        CK(hipFree(dw));
        CK(hipFree(dy));
        return false;
    }
    y = host_copy(dy, y.size());

    double worst = 0.0, mag = 1e-30;
    int bad = 0;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < N; ++c) {
            double ref = beta_one ? y0[(size_t) t * ldy + c] : 0.0;
            for (int k = 0; k < K; ++k) {
                const float xv = half_value(x[(size_t) t * ldx + k]);
                const float wv = half_value(w[(size_t) c * K + k]);
                ref += (double) xv * wv;
            }
            const double got = y[(size_t) t * ldy + c];
            if (!std::isfinite(got)) ++bad;
            worst = std::max(worst, std::fabs(got - ref));
            mag = std::max(mag, std::fabs(ref));
        }
    for (int t = 0; t < T; ++t)
        for (int c = N; c < ldy; ++c)
            if (y[(size_t) t * ldy + c] != y0[(size_t) t * ldy + c]) ++bad;
    const bool ok = bad == 0 && worst <= 2e-3 * mag;
    std::printf("%s T=%d N=%d K=%d ldx=%d ldy=%d beta=%d: rel %.3e, bad %d\n",
                ok ? "ok  " : "FAIL", T, N, K, ldx, ldy, beta_one ? 1 : 0,
                worst / mag, bad);
    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(dy));
    return ok;
}

static void bench(int T, int N, int K) {
    std::vector<uint16_t> x((size_t) T * K, half_bits(1.0f));
    std::vector<uint16_t> w((size_t) N * K, half_bits(0.01f));
    uint16_t* dx = device_copy(x);
    uint16_t* dw = device_copy(w);
    float* dy = nullptr;
    CK(hipMalloc(&dy, (size_t) T * N * 4));
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    for (int i = 0; i < 3; ++i)
        (void) strata_pf_gemm_f16_ld(dx, K, dw, K, dy, T, N, K, N, 0.0f);
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(e0));
    constexpr int reps = 10;
    for (int i = 0; i < reps; ++i)
        (void) strata_pf_gemm_f16_ld(dx, K, dw, K, dy, T, N, K, N, 0.0f);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms = 0.0f;
    CK(hipEventElapsedTime(&ms, e0, e1));
    const double tmac = (double) T * N * K * reps / (ms * 1e-3) / 1e12;
    std::printf("bench T=%d N=%d K=%d: %.3f ms, %.2f TMAC/s\n", T, N, K,
                ms / reps, tmac);
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(dy));
}

int main() {
    std::mt19937 rng(1234);
    bool ok = true;
    ok &= check_shape(65, 193, 128, 136, 201, false, rng);
    ok &= check_shape(65, 193, 128, 136, 201, true, rng);
    ok &= check_shape(96, 320, 5120, 5120, 320, false, rng);
    bench(1024, 2304, 5120);
    bench(1024, 5120, 2304);
    return ok ? 0 : 1;
}
