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

__global__ void fill_pattern(uint16_t* values, size_t count, uint32_t salt) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t v = (uint32_t) i * 1664525u + salt * 1013904223u + 0x3f803f80u;
    v ^= v >> 16;
    // Keep the operands finite and of moderate magnitude without spending host
    // time constructing and uploading hundreds of MB of random FP16 data.
    values[i] = (uint16_t) (0x2c002c00u + (v & 0x03ff03ffu));
}

__global__ void rows_to_half(const float* __restrict__ x, int ldx, int rows, int cols,
                             uint16_t* __restrict__ out) {
    const int r = blockIdx.y;
    if (r >= rows) return;
    const float* src = x + (size_t) r * ldx;
    uint16_t* dst = out + (size_t) r * cols;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x)
        dst[c] = __half_as_ushort(__float2half_rn(src[c]));
}

static bool check_out16(int T, int N, int K) {
    uint16_t* dx = nullptr;
    uint16_t* dw = nullptr;
    float* yf = nullptr;
    uint16_t* converted = nullptr;
    uint16_t* direct = nullptr;
    CK(hipMalloc(&dx, (size_t) T * K * sizeof(uint16_t)));
    CK(hipMalloc(&dw, (size_t) N * K * sizeof(uint16_t)));
    CK(hipMalloc(&yf, (size_t) T * N * sizeof(float)));
    CK(hipMalloc(&converted, (size_t) T * N * sizeof(uint16_t)));
    CK(hipMalloc(&direct, (size_t) T * N * sizeof(uint16_t)));
    constexpr int threads = 256;
    fill_pattern<<<(unsigned) (((size_t) T * K + threads - 1) / threads), threads>>>(dx, (size_t) T * K, 11);
    fill_pattern<<<(unsigned) (((size_t) N * K + threads - 1) / threads), threads>>>(dw, (size_t) N * K, 12);
    const bool full_taken = strata_pf_gemm_f16_ld(dx, K, dw, K, yf, T, N, K, N, 0.0f);
    const dim3 grid(std::min(8, (N + 255) / 256), T);
    rows_to_half<<<grid, threads>>>(yf, N, T, N, converted);
    const bool half_taken = strata_pf_gemm_f16_ld_out16(dx, K, dw, K, direct, T, N, K, N);
    CK(hipDeviceSynchronize());

    std::vector<uint16_t> converted_host((size_t) T * N), direct_host((size_t) T * N);
    CK(hipMemcpy(converted_host.data(), converted, converted_host.size() * sizeof(uint16_t),
                 hipMemcpyDeviceToHost));
    CK(hipMemcpy(direct_host.data(), direct, direct_host.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const size_t first_diff = [&] {
        for (size_t i = 0; i < converted_host.size(); ++i)
            if (converted_host[i] != direct_host[i]) return i;
        return converted_host.size();
    }();
    const bool ok = full_taken && half_taken && first_diff == converted_host.size();
    std::printf("%s out16 T=%d N=%d K=%d: %s\n", ok ? "ok  " : "FAIL", T, N, K,
                ok ? "bit-identical to float+rows_f16" : "outputs differ");
    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(yf));
    CK(hipFree(converted));
    CK(hipFree(direct));
    return ok;
}

static void bench_out16(int T, int N, int K) {
    uint16_t* dx = nullptr;
    uint16_t* dw = nullptr;
    float* yf = nullptr;
    uint16_t* yh = nullptr;
    CK(hipMalloc(&dx, (size_t) T * K * sizeof(uint16_t)));
    CK(hipMalloc(&dw, (size_t) N * K * sizeof(uint16_t)));
    CK(hipMalloc(&yf, (size_t) T * N * sizeof(float)));
    CK(hipMalloc(&yh, (size_t) T * N * sizeof(uint16_t)));
    constexpr int threads = 256;
    fill_pattern<<<(unsigned) (((size_t) T * K + threads - 1) / threads), threads>>>(dx, (size_t) T * K, 21);
    fill_pattern<<<(unsigned) (((size_t) N * K + threads - 1) / threads), threads>>>(dw, (size_t) N * K, 22);
    const dim3 grid(std::min(8, (N + 255) / 256), T);
    auto split = [&] {
        (void) strata_pf_gemm_f16_ld(dx, K, dw, K, yf, T, N, K, N, 0.0f);
        rows_to_half<<<grid, threads>>>(yf, N, T, N, yh);
    };
    auto fused = [&] { (void) strata_pf_gemm_f16_ld_out16(dx, K, dw, K, yh, T, N, K, N); };
    for (int i = 0; i < 3; ++i) split(), fused();
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto time_ms = [&](auto&& fn) {
        CK(hipEventRecord(e0));
        constexpr int reps = 20;
        for (int i = 0; i < reps; ++i) fn();
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float elapsed = 0.0f;
        CK(hipEventElapsedTime(&elapsed, e0, e1));
        return elapsed / reps;
    };
    const float a = time_ms(split), b = time_ms(fused);
    std::printf("bench out16 T=%d N=%d K=%d: float+rows_f16 %.3f ms, fused %.3f ms, %+.2f%%\n",
                T, N, K, a, b, 100.0f * (b / a - 1.0f));
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(yf));
    CK(hipFree(yh));
}

static bool check_bench_group_out16_n(int T, int N, int K, int G, const char* tag) {
    const int ldx = G * K, group_x = K, group_y = N, ldy = G * N;
    uint16_t* dx = nullptr;
    uint16_t* dy0 = nullptr;
    uint16_t* dy1 = nullptr;
    const uint16_t* dw[8] = {};
    CK(hipMalloc(&dx, (size_t) T * ldx * sizeof(uint16_t)));
    CK(hipMalloc(&dy0, (size_t) T * ldy * sizeof(uint16_t)));
    CK(hipMalloc(&dy1, (size_t) T * ldy * sizeof(uint16_t)));
    constexpr int threads = 256;
    fill_pattern<<<(unsigned) (((size_t) T * ldx + threads - 1) / threads), threads>>>(dx, (size_t) T * ldx, 71);
    for (int g = 0; g < G; ++g) {
        uint16_t* w = nullptr;
        CK(hipMalloc(&w, (size_t) N * K * sizeof(uint16_t)));
        fill_pattern<<<(unsigned) (((size_t) N * K + threads - 1) / threads), threads>>>(
            w, (size_t) N * K, 80 + g);
        dw[g] = w;
    }
    auto individual = [&] {
        for (int g = 0; g < G; ++g)
            (void) strata_pf_gemm_f16_ld_out16(
                dx + (size_t) g * group_x, ldx, dw[g], K,
                dy0 + (size_t) g * group_y, T, N, K, ldy);
    };
    auto grouped = [&] {
        (void) strata_pf_gemm_f16_ld_out16_group8(
            dx, ldx, dw, K, dy1, T, N, K, ldy, group_x, group_y, nullptr, G);
    };
    individual();
    grouped();
    CK(hipDeviceSynchronize());
    std::vector<uint16_t> a((size_t) T * ldy), b((size_t) T * ldy);
    CK(hipMemcpy(a.data(), dy0, a.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    CK(hipMemcpy(b.data(), dy1, b.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    const bool same = std::memcmp(a.data(), b.data(), a.size() * sizeof(uint16_t)) == 0;
    std::printf("%s wo_a %s out16: %s\n", same ? "ok  " : "FAIL", tag,
                same ? "bit-identical to individual launches" : "outputs differ");

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto time_ms = [&](auto&& fn) {
        CK(hipEventRecord(e0));
        constexpr int reps = 20;
        for (int i = 0; i < reps; ++i) fn();
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float elapsed = 0.0f;
        CK(hipEventElapsedTime(&elapsed, e0, e1));
        return elapsed / reps;
    };
    const float off = time_ms(individual), on = time_ms(grouped);
    std::printf("bench wo_a %s out16: individual %.3f ms, grouped %.3f ms, %+.2f%%\n",
                tag, off, on, 100.0f * (on / off - 1.0f));
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipFree(dx));
    CK(hipFree(dy0));
    CK(hipFree(dy1));
    for (int g = 0; g < G; ++g) CK(hipFree((void*) dw[g]));
    return same;
}

static bool check_bench_project4() {
    constexpr int T = 1024, K = 5120;
    constexpr int N[4] = {1280, 512, 512, 512};
    uint16_t* dx = nullptr;
    const uint16_t* dw[4] = {};
    float* dy0[4] = {};
    float* dy1[4] = {};
    CK(hipMalloc(&dx, (size_t) T * K * sizeof(uint16_t)));
    constexpr int threads = 256;
    fill_pattern<<<(unsigned) (((size_t) T * K + threads - 1) / threads), threads>>>(
        dx, (size_t) T * K, 91);
    for (int g = 0; g < 4; ++g) {
        uint16_t* w = nullptr;
        CK(hipMalloc(&w, (size_t) N[g] * K * sizeof(uint16_t)));
        fill_pattern<<<(unsigned) (((size_t) N[g] * K + threads - 1) / threads), threads>>>(
            w, (size_t) N[g] * K, 100 + g);
        dw[g] = w;
        CK(hipMalloc(&dy0[g], (size_t) T * N[g] * sizeof(float)));
        CK(hipMalloc(&dy1[g], (size_t) T * N[g] * sizeof(float)));
    }
    auto individual = [&] {
        for (int g = 0; g < 4; ++g)
            (void) strata_pf_gemm_f16_ld(
                dx, K, dw[g], K, dy0[g], T, N[g], K, N[g], 0.0f);
    };
    auto grouped = [&] {
        (void) strata_pf_gemm_f16_ld_project4(
            dx, K, dw, dy1, N, T, K);
    };
    individual();
    grouped();
    CK(hipDeviceSynchronize());
    bool same = true;
    for (int g = 0; g < 4; ++g) {
        std::vector<float> a((size_t) T * N[g]), b((size_t) T * N[g]);
        CK(hipMemcpy(a.data(), dy0[g], a.size() * sizeof(float), hipMemcpyDeviceToHost));
        CK(hipMemcpy(b.data(), dy1[g], b.size() * sizeof(float), hipMemcpyDeviceToHost));
        same = same &&
               std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    }
    std::printf("%s projection4: %s\n", same ? "ok  " : "FAIL",
                same ? "bit-identical to four launches" : "outputs differ");

    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto time_ms = [&](auto&& fn) {
        CK(hipEventRecord(e0));
        constexpr int reps = 20;
        for (int i = 0; i < reps; ++i) fn();
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float elapsed = 0.0f;
        CK(hipEventElapsedTime(&elapsed, e0, e1));
        return elapsed / reps;
    };
    const float off = time_ms(individual), on = time_ms(grouped);
    std::printf("bench projection4: individual %.3f ms, grouped %.3f ms, %+.2f%%\n",
                off, on, 100.0f * (on / off - 1.0f));
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipFree(dx));
    for (int g = 0; g < 4; ++g) {
        CK(hipFree((void*) dw[g]));
        CK(hipFree(dy0[g]));
        CK(hipFree(dy1[g]));
    }
    return same;
}

static bool check_split_n128(int T, int N, int K, bool qb) {
    uint16_t* dx = nullptr;
    uint16_t* dw = nullptr;
    float* y64 = nullptr;
    float* ysplit = nullptr;
    CK(hipMalloc(&dx, (size_t) T * K * sizeof(uint16_t)));
    CK(hipMalloc(&dw, (size_t) N * K * sizeof(uint16_t)));
    CK(hipMalloc(&y64, (size_t) T * N * sizeof(float)));
    CK(hipMalloc(&ysplit, (size_t) T * N * sizeof(float)));
    constexpr int threads = 256;
    fill_pattern<<<(unsigned) (((size_t) T * K + threads - 1) / threads), threads>>>(dx, (size_t) T * K, 51);
    fill_pattern<<<(unsigned) (((size_t) N * K + threads - 1) / threads), threads>>>(dw, (size_t) N * K, 52);
    const bool n64_taken = strata_pf_gemm_f16_ld(dx, K, dw, K, y64, T, N, K, N, 0.0f);
    const bool split_taken =
        qb ? strata_pf_gemm_f16_ld_qb_split_n128(dx, K, dw, K, ysplit, T, N, K, N, 0.0f)
           : strata_pf_gemm_f16_ld_split_n128(dx, K, dw, K, ysplit, T, N, K, N, 0.0f);
    CK(hipDeviceSynchronize());

    std::vector<float> n64((size_t) T * N), split((size_t) T * N);
    CK(hipMemcpy(n64.data(), y64, n64.size() * sizeof(float), hipMemcpyDeviceToHost));
    CK(hipMemcpy(split.data(), ysplit, split.size() * sizeof(float), hipMemcpyDeviceToHost));
    const size_t first_diff = [&] {
        for (size_t i = 0; i < n64.size(); ++i) {
            uint32_t a, b;
            std::memcpy(&a, &n64[i], sizeof(a));
            std::memcpy(&b, &split[i], sizeof(b));
            if (a != b) return i;
        }
        return n64.size();
    }();
    const bool ok = n64_taken && split_taken && first_diff == n64.size();
    std::printf("%s %s split N128 T=%d N=%d K=%d: %s\n", ok ? "ok  " : "FAIL", qb ? "q_b" : "general", T, N, K,
                ok ? "bit-identical to N64" : "outputs differ");
    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(y64));
    CK(hipFree(ysplit));
    return ok;
}

static void bench_split_n128(int T, int N, int K, bool qb) {
    uint16_t* dx = nullptr;
    uint16_t* dw = nullptr;
    float* y = nullptr;
    CK(hipMalloc(&dx, (size_t) T * K * sizeof(uint16_t)));
    CK(hipMalloc(&dw, (size_t) N * K * sizeof(uint16_t)));
    CK(hipMalloc(&y, (size_t) T * N * sizeof(float)));
    constexpr int threads = 256;
    fill_pattern<<<(unsigned) (((size_t) T * K + threads - 1) / threads), threads>>>(dx, (size_t) T * K, 61);
    fill_pattern<<<(unsigned) (((size_t) N * K + threads - 1) / threads), threads>>>(dw, (size_t) N * K, 62);
    auto n64 = [&] { (void) strata_pf_gemm_f16_ld(dx, K, dw, K, y, T, N, K, N, 0.0f); };
    auto split = [&] {
        if (qb)
            (void) strata_pf_gemm_f16_ld_qb_split_n128(dx, K, dw, K, y, T, N, K, N, 0.0f);
        else
            (void) strata_pf_gemm_f16_ld_split_n128(dx, K, dw, K, y, T, N, K, N, 0.0f);
    };
    for (int i = 0; i < 2; ++i) n64(), split();
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto time_ms = [&](auto&& fn) {
        CK(hipEventRecord(e0));
        constexpr int reps = 10;
        for (int i = 0; i < reps; ++i) fn();
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float elapsed = 0.0f;
        CK(hipEventElapsedTime(&elapsed, e0, e1));
        return elapsed / reps;
    };
    std::vector<float> n64_us, split_us;
    for (int pair = 0; pair < 3; ++pair) {
        if ((pair & 1) == 0) {
            n64_us.push_back(1000.0f * time_ms(n64));
            split_us.push_back(1000.0f * time_ms(split));
        } else {
            split_us.push_back(1000.0f * time_ms(split));
            n64_us.push_back(1000.0f * time_ms(n64));
        }
    }
    auto median = [](std::vector<float> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    const float a = median(n64_us), b = median(split_us);
    std::printf("bench %s split N128 T=%d N=%d K=%d: N64 %.3f us, split %.3f us, %+.2f%%\n",
                qb ? "q_b" : "general", T, N, K, a, b, 100.0f * (b / a - 1.0f));
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(y));
}

static bool check_m128(int T, int N, int K) {
    uint16_t* dx = nullptr;
    uint16_t* dw = nullptr;
    float* y64 = nullptr;
    float* y128 = nullptr;
    CK(hipMalloc(&dx, (size_t) T * K * sizeof(uint16_t)));
    CK(hipMalloc(&dw, (size_t) N * K * sizeof(uint16_t)));
    CK(hipMalloc(&y64, (size_t) T * N * sizeof(float)));
    CK(hipMalloc(&y128, (size_t) T * N * sizeof(float)));
    constexpr int threads = 256;
    fill_pattern<<<(unsigned) (((size_t) T * K + threads - 1) / threads), threads>>>(dx, (size_t) T * K, 71);
    fill_pattern<<<(unsigned) (((size_t) N * K + threads - 1) / threads), threads>>>(dw, (size_t) N * K, 72);
    const bool m64_taken = strata_pf_gemm_f16_ld(dx, K, dw, K, y64, T, N, K, N, 0.0f);
    const bool m128_taken = strata_pf_gemm_f16_ld_m128(dx, K, dw, K, y128, T, N, K, N, 0.0f);
    CK(hipDeviceSynchronize());

    std::vector<float> m64((size_t) T * N), m128((size_t) T * N);
    CK(hipMemcpy(m64.data(), y64, m64.size() * sizeof(float), hipMemcpyDeviceToHost));
    CK(hipMemcpy(m128.data(), y128, m128.size() * sizeof(float), hipMemcpyDeviceToHost));
    const size_t first_diff = [&] {
        for (size_t i = 0; i < m64.size(); ++i) {
            uint32_t a, b;
            std::memcpy(&a, &m64[i], sizeof(a));
            std::memcpy(&b, &m128[i], sizeof(b));
            if (a != b) return i;
        }
        return m64.size();
    }();
    const bool ok = m64_taken && m128_taken && first_diff == m64.size();
    std::printf("%s M128 T=%d N=%d K=%d: %s\n", ok ? "ok  " : "FAIL", T, N, K,
                ok ? "bit-identical to M64 N64" : "outputs differ");
    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(y64));
    CK(hipFree(y128));
    return ok;
}

static void bench_m128(int T, int N, int K) {
    uint16_t* dx = nullptr;
    uint16_t* dw = nullptr;
    float* y = nullptr;
    CK(hipMalloc(&dx, (size_t) T * K * sizeof(uint16_t)));
    CK(hipMalloc(&dw, (size_t) N * K * sizeof(uint16_t)));
    CK(hipMalloc(&y, (size_t) T * N * sizeof(float)));
    constexpr int threads = 256;
    fill_pattern<<<(unsigned) (((size_t) T * K + threads - 1) / threads), threads>>>(dx, (size_t) T * K, 81);
    fill_pattern<<<(unsigned) (((size_t) N * K + threads - 1) / threads), threads>>>(dw, (size_t) N * K, 82);
    auto m64 = [&] { (void) strata_pf_gemm_f16_ld(dx, K, dw, K, y, T, N, K, N, 0.0f); };
    auto m128 = [&] { (void) strata_pf_gemm_f16_ld_m128(dx, K, dw, K, y, T, N, K, N, 0.0f); };
    for (int i = 0; i < 2; ++i) m64(), m128();
    CK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto time_ms = [&](auto&& fn) {
        CK(hipEventRecord(e0));
        constexpr int reps = 10;
        for (int i = 0; i < reps; ++i) fn();
        CK(hipEventRecord(e1));
        CK(hipEventSynchronize(e1));
        float elapsed = 0.0f;
        CK(hipEventElapsedTime(&elapsed, e0, e1));
        return elapsed / reps;
    };
    const float a = time_ms(m64), b = time_ms(m128);
    std::printf("bench M128 T=%d N=%d K=%d: M64 %.3f ms, M128 %.3f ms, %+.2f%%\n",
                T, N, K, a, b, 100.0f * (b / a - 1.0f));
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(y));
}

// The dense weight cache changes a sequence of N-column GEMM launches into one
// full-width launch.  The gfx928 kernel partitions both into identical 128x256
// output tiles, so every output must be bit-identical.  Exercise the shapes and
// slab width used by the real 4K/8K prompt path.
static bool check_partitioned(int T, int N, int K, int part) {
    uint16_t* dx = nullptr;
    uint16_t* dw = nullptr;
    float* y_full = nullptr;
    float* y_part = nullptr;
    CK(hipMalloc(&dx, (size_t) T * K * sizeof(uint16_t)));
    CK(hipMalloc(&dw, (size_t) N * K * sizeof(uint16_t)));
    CK(hipMalloc(&y_full, (size_t) T * N * sizeof(float)));
    CK(hipMalloc(&y_part, (size_t) T * N * sizeof(float)));
    constexpr int threads = 256;
    fill_pattern<<<(unsigned) (((size_t) T * K + threads - 1) / threads), threads>>>(dx, (size_t) T * K, 1);
    fill_pattern<<<(unsigned) (((size_t) N * K + threads - 1) / threads), threads>>>(dw, (size_t) N * K, 2);
    CK(hipGetLastError());

    const bool full_taken = strata_pf_gemm_f16_ld(dx, K, dw, K, y_full, T, N, K, N, 0.0f);
    bool part_taken = full_taken;
    for (int n0 = 0; n0 < N; n0 += part) {
        const int nc = std::min(part, N - n0);
        part_taken &= strata_pf_gemm_f16_ld(dx, K, dw + (size_t) n0 * K, K, y_part + n0, T, nc, K, N, 0.0f);
    }
    CK(hipDeviceSynchronize());

    std::vector<float> full((size_t) T * N), partitioned((size_t) T * N);
    CK(hipMemcpy(full.data(), y_full, full.size() * sizeof(float), hipMemcpyDeviceToHost));
    CK(hipMemcpy(partitioned.data(), y_part, partitioned.size() * sizeof(float), hipMemcpyDeviceToHost));
    const size_t first_diff = [&] {
        for (size_t i = 0; i < full.size(); ++i) {
            uint32_t a, b;
            std::memcpy(&a, &full[i], sizeof(a));
            std::memcpy(&b, &partitioned[i], sizeof(b));
            if (a != b) return i;
        }
        return full.size();
    }();
    const bool ok = full_taken && part_taken && first_diff == full.size();
    std::printf("%s partition T=%d N=%d K=%d part=%d: %s\n", ok ? "ok  " : "FAIL", T, N, K, part,
                ok ? "bit-identical" : "outputs differ");
    CK(hipFree(dx));
    CK(hipFree(dw));
    CK(hipFree(y_full));
    CK(hipFree(y_part));
    return ok;
}

int main() {
    std::mt19937 rng(1234);
    bool ok = true;
    ok &= check_shape(65, 193, 128, 136, 201, false, rng);
    ok &= check_shape(65, 193, 128, 136, 201, true, rng);
    ok &= check_shape(96, 320, 5120, 5120, 320, false, rng);
    ok &= check_partitioned(1024, 32768, 1280, 6400);  // wq_b: 8 MiB dequant slab
    ok &= check_partitioned(1024, 5120, 8192, 1024);   // wo_b: 8 MiB dequant slab
    ok &= check_partitioned(1024, 1024, 4096, 2048);   // one wo_a group, already full width
    ok &= check_out16(1024, 512, 4096);                 // one wo_a group in the chunked prompt
    ok &= check_bench_group_out16_n(1024, 512, 4096, 8, "group8 (nm=1)");  // all eight wo_a groups
    ok &= check_bench_group_out16_n(1024, 512, 4096, 4, "group4 (nm=2)");  // TP=2's four
    ok &= check_bench_group_out16_n(1024, 512, 4096, 2, "group2 (nm=4)");  // TP=4's two
    ok &= check_bench_project4();                       // q_a + wkv + compressor projections
    ok &= check_split_n128(1024, 32768, 1280, true);
    ok &= check_split_n128(1024, 8192, 1280, true);    // q_b, TP=4
    ok &= check_split_n128(1024, 1152, 1280, true);    // q_b, TP=8
    ok &= check_split_n128(1024, 1280, 5120, false);
    ok &= check_split_n128(1024, 512, 5120, false);
    ok &= check_split_n128(1024, 2304, 5120, false);
    ok &= check_split_n128(1024, 640, 5120, false);    // shared expert gate/up, TP=4 odd-block members
    ok &= check_split_n128(1024, 576, 5120, false);    // shared expert gate/up, TP=4
    ok &= check_split_n128(1024, 288, 5120, false);    // shared expert gate/up, TP=8
    ok &= check_split_n128(1024, 5120, 4096, false);
    ok &= check_split_n128(1024, 5120, 2304, false);
    ok &= check_split_n128(1024, 5120, 640, false);    // shared expert down, TP=4
    ok &= check_split_n128(1024, 5120, 576, false);    // shared expert down, TP=4
    ok &= check_split_n128(1024, 5120, 512, false);    // shared expert down, TP=4
    ok &= check_m128(1024, 32768, 1280);                // q_b, nm=1
    ok &= check_m128(1024, 8192, 1280);                 // q_b, nm=4
    ok &= check_m128(1024, 5120, 4096);                 // wo_b
    bench(1024, 2304, 5120);
    bench(1024, 5120, 2304);
    bench_out16(1024, 512, 4096);
    bench_split_n128(1024, 32768, 1280, true);
    bench_split_n128(1024, 1280, 5120, false);
    bench_split_n128(1024, 512, 5120, false);
    bench_split_n128(1024, 2304, 5120, false);
    bench_split_n128(1024, 640, 5120, false);
    bench_split_n128(1024, 5120, 4096, false);
    bench_split_n128(1024, 5120, 2304, false);
    bench_split_n128(1024, 5120, 640, false);
    bench_split_n128(1024, 5120, 512, false);
    bench_m128(1024, 32768, 1280);
    bench_m128(1024, 8192, 1280);
    bench_m128(1024, 5120, 4096);
    return ok ? 0 : 1;
}
