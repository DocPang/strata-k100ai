// Opt-in F16 dense GEMM for K100AI/gfx928.
//
// Y[T,N] += beta * Y[T,N] + X[T,K] * W[N,K]^T.  A CTA owns 64
// rows x 64 columns; its four waves each own a 16-row tile and all four
// 16-column MMAC tiles.  The path is selected by STRATA_DS4_DENSE_MMAC928=1.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace pfg928 {

typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
typedef float floatx4 __attribute__((ext_vector_type(4)));

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int SH = BK + 8;
constexpr int WAVES = BM / 16;
constexpr int THREADS = WAVES * 64;
constexpr int NT = BN / 16;

__device__ __forceinline__ floatx4 mmac(f16x4 a, f16x4 b, floatx4 c) {
#if defined(__gfx928__)
    return __builtin_amdgcn_mmac_f32_16x16x16f16(a, b, c);
#else
    (void) a;
    (void) b;
    return c;
#endif
}

__global__ void __launch_bounds__(THREADS) gemm_kernel(
    const uint16_t* __restrict__ X, int ldx,
    const uint16_t* __restrict__ W, int ldw,
    float* __restrict__ Y, int T, int N, int K, int ldy, int beta) {
    __shared__ __align__(16) uint16_t sx[BM][SH];
    __shared__ __align__(16) uint16_t sw[BN][SH];

    const int tid = (int) threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    const int m0 = (int) blockIdx.y * BM;
    const int n0 = (int) blockIdx.x * BN;

    floatx4 acc[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt] = floatx4{0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 512 aligned 16-byte loads: two per thread for each operand.
        for (int item = tid; item < BM * 8; item += THREADS) {
            const int r = item >> 3, col = (item & 7) * 8;
            uint4 value = {};
            if (m0 + r < T)
                value = *reinterpret_cast<const uint4*>(
                    X + (size_t) (m0 + r) * ldx + k0 + col);
            *reinterpret_cast<uint4*>(&sx[r][col]) = value;
        }
        for (int item = tid; item < BN * 8; item += THREADS) {
            const int r = item >> 3, col = (item & 7) * 8;
            uint4 value = {};
            if (n0 + r < N)
                value = *reinterpret_cast<const uint4*>(
                    W + (size_t) (n0 + r) * ldw + k0 + col);
            *reinterpret_cast<uint4*>(&sw[r][col]) = value;
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const f16x4 a = *reinterpret_cast<const f16x4*>(
                &sx[wave * 16 + rr][ks + 4 * gg]);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const f16x4 b = *reinterpret_cast<const f16x4*>(
                    &sw[nt * 16 + rr][ks + 4 * gg]);
                acc[nt] = mmac(a, b, acc[nt]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = m0 + wave * 16 + rr;
            const int col = n0 + nt * 16 + gg + 4 * j;
            if (row < T && col < N) {
                float* dst = Y + (size_t) row * ldy + col;
                const float v = beta != 0 ? *dst : 0.0f;
                *dst = v + acc[nt][j];
            }
        }
    }
}

inline bool device_ok() {
    static std::atomic<int> cache[64];
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) return false;
    int value = cache[dev].load(std::memory_order_relaxed);
    if (value == 0) {
        hipDeviceProp_t prop{};
        value = hipGetDeviceProperties(&prop, dev) == hipSuccess &&
                        std::strncmp(prop.gcnArchName, "gfx928", 6) == 0
                    ? 2
                    : 1;
        cache[dev].store(value, std::memory_order_relaxed);
    }
    return value == 2;
}

inline bool enabled() {
    static const bool on = [] {
        const char* value = std::getenv("STRATA_DS4_DENSE_MMAC928");
        return value != nullptr && value[0] == '1';
    }();
    return on;
}

inline bool launch(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                   float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                   float beta, hipStream_t stream) {
    if (!enabled() || !device_ok()) return false;
    if (T < 1 || T > INT32_MAX || N < 1 || N > INT32_MAX || K < BK ||
        K > INT32_MAX || K % BK != 0)
        return false;
    if (ldx < K || ldw < K || ldx % 8 != 0 || ldw % 8 != 0 ||
        ldx > INT32_MAX || ldw > INT32_MAX ||
        ldy < N || ldy > INT32_MAX)
        return false;
    if (beta != 0.0f && beta != 1.0f) return false;
    const dim3 grid((unsigned) ((N + BN - 1) / BN), (unsigned) ((T + BM - 1) / BM));
    gemm_kernel<<<grid, THREADS, 0, stream>>>(
        X, (int) ldx, W, (int) ldw, Y, (int) T, (int) N, (int) K, (int) ldy,
        beta != 0.0f ? 1 : 0);
    return hipGetLastError() == hipSuccess;
}

}  // namespace pfg928
