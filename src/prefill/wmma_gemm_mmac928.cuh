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
constexpr int BM128 = 128;
constexpr int BN = 64;
constexpr int BN128 = 128;
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

template <bool OUT16>
__global__ void __launch_bounds__(THREADS) gemm_kernel(
    const uint16_t* __restrict__ X, int ldx,
    const uint16_t* __restrict__ W, int ldw,
    void* __restrict__ Y, int T, int N, int K, int ldy, int beta,
    int swizzle) {
    __shared__ __align__(16) uint16_t sx[BM][SH];
    __shared__ __align__(16) uint16_t sw[BN][SH];

    const int tid = (int) threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    int mt = (int) blockIdx.y;
    int nt = (int) blockIdx.x;
    if (swizzle) {
        const int linear = (int) blockIdx.x + (int) gridDim.x * (int) blockIdx.y;
        nt = linear / (int) gridDim.y;
        mt = linear % (int) gridDim.y;
    }
    const int m0 = mt * BM;
    const int n0 = nt * BN;

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
                if constexpr (OUT16) {
                    static_cast<uint16_t*>(Y)[(size_t) row * ldy + col] =
                        __half_as_ushort(__float2half_rn(acc[nt][j]));
                } else {
                    float* dst = static_cast<float*>(Y) + (size_t) row * ldy + col;
                    const float v = beta != 0 ? *dst : 0.0f;
                    *dst = v + acc[nt][j];
                }
            }
        }
    }
}

struct F16Group8 {
    const uint16_t* w[8];
};

struct F16Project4 {
    const uint16_t* w[4];
    float* y[4];
    int n[4];
};

__global__ void __launch_bounds__(THREADS) gemm_project4_kernel(
    const uint16_t* __restrict__ X, int ldx, F16Project4 jobs, int ldw,
    int T, int K) {
    __shared__ __align__(16) uint16_t sx[BM][SH];
    __shared__ __align__(16) uint16_t sw[BN][SH];

    const int g = (int) blockIdx.z;
    const int N = jobs.n[g];
    const int n0 = (int) blockIdx.x * BN;
    if (N <= 0 || n0 >= N) return;
    const uint16_t* W = jobs.w[g];
    float* Y = jobs.y[g];
    const int tid = (int) threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    const int m0 = (int) blockIdx.y * BM;

    floatx4 acc[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt] = floatx4{0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < K; k0 += BK) {
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
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = m0 + wave * 16 + rr;
            const int col = n0 + nt * 16 + gg + 4 * j;
            if (row < T && col < N) Y[(size_t) row * N + col] = acc[nt][j];
        }
}

__global__ void __launch_bounds__(THREADS) gemm_out16_group8_kernel(
    const uint16_t* __restrict__ X, int ldx, F16Group8 weights, int ldw,
    uint16_t* __restrict__ Y, int T, int N, int K, int ldy,
    size_t x_group_stride, size_t y_group_stride) {
    __shared__ __align__(16) uint16_t sx[BM][SH];
    __shared__ __align__(16) uint16_t sw[BN][SH];

    const int g = (int) blockIdx.z;
    const uint16_t* W = weights.w[g];
    X += (size_t) g * x_group_stride;
    Y += (size_t) g * y_group_stride;
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
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = m0 + wave * 16 + rr;
            const int col = n0 + nt * 16 + gg + 4 * j;
            if (row < T && col < N)
                Y[(size_t) row * ldy + col] =
                    __half_as_ushort(__float2half_rn(acc[nt][j]));
        }
}

// The grouped wo_a launch with a 128-row CTA.  It keeps the same N64 tile and
// K accumulation order as gemm_out16_group8_kernel, but each W tile is loaded
// once for twice as many activation rows.
__global__ void __launch_bounds__(512) gemm_out16_group8_m128_kernel(
    const uint16_t* __restrict__ X, int ldx, F16Group8 weights, int ldw,
    uint16_t* __restrict__ Y, int T, int N, int K, int ldy,
    size_t x_group_stride, size_t y_group_stride) {
    __shared__ __align__(16) uint16_t sx[BM128][SH];
    __shared__ __align__(16) uint16_t sw[BN][SH];

    const int g = (int) blockIdx.z;
    const uint16_t* W = weights.w[g];
    X += (size_t) g * x_group_stride;
    Y += (size_t) g * y_group_stride;
    const int tid = (int) threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    const int m0 = (int) blockIdx.y * BM128;
    const int n0 = (int) blockIdx.x * BN;

    floatx4 acc[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt] = floatx4{0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int item = tid; item < BM128 * 8; item += 512) {
            const int r = item >> 3, col = (item & 7) * 8;
            uint4 value = {};
            if (m0 + r < T)
                value = *reinterpret_cast<const uint4*>(
                    X + (size_t) (m0 + r) * ldx + k0 + col);
            *reinterpret_cast<uint4*>(&sx[r][col]) = value;
        }
        for (int item = tid; item < BN * 8; item += 512) {
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
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int row = m0 + wave * 16 + rr;
            const int col = n0 + nt * 16 + gg + 4 * j;
            if (row < T && col < N)
                Y[(size_t) row * ldy + col] =
                    __half_as_ushort(__float2half_rn(acc[nt][j]));
        }
}

// A 64x128 tile with two 4-wave row groups.  Both groups reuse the same A
// tile; each thread keeps the stock four accumulators instead of the eight a
// single 4-wave N128 kernel needs.
__global__ void __launch_bounds__(512) gemm_kernel_n128_split(
    const uint16_t* __restrict__ X, int ldx,
    const uint16_t* __restrict__ W, int ldw,
    float* __restrict__ Y, int T, int N, int K, int ldy, int beta,
    int swizzle) {
    __shared__ __align__(16) uint16_t sx[BM][SH];
    __shared__ __align__(16) uint16_t sw[BN128][SH];

    const int tid = (int) threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    const int wm = wave & 3;
    const int wn = wave >> 2;
    int mt = (int) blockIdx.y;
    int nt = (int) blockIdx.x;
    if (swizzle) {
        const int linear = (int) blockIdx.x + (int) gridDim.x * (int) blockIdx.y;
        nt = linear / (int) gridDim.y;
        mt = linear % (int) gridDim.y;
    }
    const int m0 = mt * BM;
    const int n0 = nt * BN128;
    const int nbase = n0 + 64 * wn;

    floatx4 acc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j] = floatx4{0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int item = tid; item < BM * 8; item += 512) {
            const int r = item >> 3, col = (item & 7) * 8;
            uint4 value = {};
            if (m0 + r < T)
                value = *reinterpret_cast<const uint4*>(X + (size_t) (m0 + r) * ldx + k0 + col);
            *reinterpret_cast<uint4*>(&sx[r][col]) = value;
        }
        for (int item = tid; item < BN128 * 8; item += 512) {
            const int r = item >> 3, col = (item & 7) * 8;
            uint4 value = {};
            if (n0 + r < N)
                value = *reinterpret_cast<const uint4*>(W + (size_t) (n0 + r) * ldw + k0 + col);
            *reinterpret_cast<uint4*>(&sw[r][col]) = value;
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const f16x4 a = *reinterpret_cast<const f16x4*>(&sx[wm * 16 + rr][ks + 4 * gg]);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const f16x4 b = *reinterpret_cast<const f16x4*>(&sw[64 * wn + j * 16 + rr][ks + 4 * gg]);
                acc[j] = mmac(a, b, acc[j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = m0 + wm * 16 + rr;
            const int col = nbase + j * 16 + gg + 4 * e;
            if (row < T && col < N) {
                float* dst = Y + (size_t) row * ldy + col;
                const float v = beta != 0 ? *dst : 0.0f;
                *dst = v + acc[j][e];
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

template <bool OUT16>
inline bool launch_impl(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                        void* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
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
    static const bool swizzle_env = [] {
        const char* value =
            std::getenv("STRATA_DS4_DENSE_MMAC928_SWIZZLE");
        return value != nullptr && value[0] == '1';
    }();
    const bool swizzle = swizzle_env && N == 32768 && K == 1280;
    const dim3 grid((unsigned) ((N + BN - 1) / BN), (unsigned) ((T + BM - 1) / BM));
    gemm_kernel<OUT16><<<grid, THREADS, 0, stream>>>(
        X, (int) ldx, W, (int) ldw, Y, (int) T, (int) N, (int) K, (int) ldy,
        beta != 0.0f ? 1 : 0, swizzle ? 1 : 0);
    return hipGetLastError() == hipSuccess;
}

inline bool launch_split_n128(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                              float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                              float beta, hipStream_t stream);
inline bool launch_m128(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                        float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                        float beta, hipStream_t stream);

inline bool launch(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                   float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                   float beta, hipStream_t stream) {
    static const bool m128_env = [] {
        const char* value = std::getenv("STRATA_DS4_DENSE_MMAC928_M128");
        return value != nullptr && value[0] == '1';
    }();
    // q_b is weight-heavy.  The 128-row CTA halves the number of times its
    // N-column W tile is loaded, while each output keeps the stock N64 K
    // accumulation order.  wo_b showed no measurable gain and stays stock.
    if (m128_env && T >= 64 && K == 1280 && N % BN == 0)
        return launch_m128(X, ldx, W, ldw, Y, T, N, K, ldy, beta, stream);
    static const bool qb_split_env = [] {
        const char* value = std::getenv("STRATA_DS4_DENSE_MMAC928_QB_SPLIT_N128");
        return value != nullptr && value[0] == '1';
    }();
    // q_b's N splits over the tensor-parallel members (32768 / nm: 8192 with
    // nm=4); every multiple of BN128 keeps the same 128-wide tile the 32768
    // case uses.
    if (qb_split_env && K == 1280 && N % BN128 == 0 && N >= BN128) {
        if (!enabled() || !device_ok()) return false;
        if (T < 1 || T > INT32_MAX || ldx < K || ldw < K || ldx % 8 != 0 || ldw % 8 != 0 ||
            ldy < N || beta != 0.0f && beta != 1.0f)
            return false;
        static const bool swizzle_env = [] {
            const char* value = std::getenv("STRATA_DS4_DENSE_MMAC928_SWIZZLE");
            return value != nullptr && value[0] == '1';
        }();
        const dim3 grid((unsigned) ((N + BN128 - 1) / BN128), (unsigned) ((T + BM - 1) / BM));
        gemm_kernel_n128_split<<<grid, 512, 0, stream>>>(
            X, (int) ldx, W, (int) ldw, Y, (int) T, (int) N, (int) K, (int) ldy,
            beta != 0.0f ? 1 : 0, swizzle_env ? 1 : 0);
        return hipGetLastError() == hipSuccess;
    }
    static const bool split_shapes_env = [] {
        const char* value = std::getenv("STRATA_DS4_DENSE_MMAC928_SPLIT_N128");
        return value != nullptr && value[0] == '1';
    }();
    // The nm=1 shapes (Round 42/58/59) generalized to the tensor-parallel
    // members: only the shared expert's gate/up (N split from 2304) and down
    // (K split from 2304) split; q_a and wkv keep their full N.  split128
    // rotates the 18 128-blocks, so nm=4 members are 640 or 512 wide (never
    // 576).  The direct launch takes all three measured shapes, but the
    // automatic whitelist leaves the TP4-only 640/640/512 shapes out:
    // Split-N128 measured +1.50%/+0.02%/+0.72% there, against -0.36% at the
    // even TP2 width.  Enumerated so the whitelist stays benchmark-backed.
    const bool split_shape =
        split_shapes_env && K % BK == 0 &&
        ((K == 5120 && (N == 1280 || N == 512 || N == 2304 || N == 1152 ||
                        N == 576)) ||
         (N == 5120 && (K == 2304 || K == 1152 || K == 576)));
    if (split_shape) return launch_split_n128(X, ldx, W, ldw, Y, T, N, K, ldy, beta, stream);
    return launch_impl<false>(X, ldx, W, ldw, Y, T, N, K, ldy, beta, stream);
}

// A 128x64 tile with eight row waves.  Each wave computes its original
// 16x64 outputs, so widening M only changes CTA assignment: the K sequence
// and the per-output accumulation are those of the 64x64 kernel.
__global__ void __launch_bounds__(512) gemm_kernel_m128(
    const uint16_t* __restrict__ X, int ldx,
    const uint16_t* __restrict__ W, int ldw,
    float* __restrict__ Y, int T, int N, int K, int ldy, int beta,
    int swizzle) {
    __shared__ __align__(16) uint16_t sx[BM128][SH];
    __shared__ __align__(16) uint16_t sw[BN][SH];

    const int tid = (int) threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    int mt = (int) blockIdx.y;
    int nt = (int) blockIdx.x;
    if (swizzle) {
        const int linear = (int) blockIdx.x + (int) gridDim.x * (int) blockIdx.y;
        nt = linear / (int) gridDim.y;
        mt = linear % (int) gridDim.y;
    }
    const int m0 = mt * BM128;
    const int n0 = nt * BN;

    floatx4 acc[NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[j] = floatx4{0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int item = tid; item < BM128 * 8; item += 512) {
            const int r = item >> 3, col = (item & 7) * 8;
            uint4 value = {};
            if (m0 + r < T)
                value = *reinterpret_cast<const uint4*>(X + (size_t) (m0 + r) * ldx + k0 + col);
            *reinterpret_cast<uint4*>(&sx[r][col]) = value;
        }
        for (int item = tid; item < BN * 8; item += 512) {
            const int r = item >> 3, col = (item & 7) * 8;
            uint4 value = {};
            if (n0 + r < N)
                value = *reinterpret_cast<const uint4*>(W + (size_t) (n0 + r) * ldw + k0 + col);
            *reinterpret_cast<uint4*>(&sw[r][col]) = value;
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const f16x4 a = *reinterpret_cast<const f16x4*>(
                &sx[wave * 16 + rr][ks + 4 * gg]);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const f16x4 b = *reinterpret_cast<const f16x4*>(
                    &sw[j * 16 + rr][ks + 4 * gg]);
                acc[j] = mmac(a, b, acc[j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = m0 + wave * 16 + rr;
            const int col = n0 + j * 16 + gg + 4 * e;
            if (row < T && col < N) {
                float* dst = Y + (size_t) row * ldy + col;
                const float v = beta != 0 ? *dst : 0.0f;
                *dst = v + acc[j][e];
            }
        }
}

inline bool launch_m128(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                        float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                        float beta, hipStream_t stream) {
    if (!enabled() || !device_ok()) return false;
    if (T < 1 || T > INT32_MAX || N < BN || N > INT32_MAX ||
        K < BK || K > INT32_MAX || K % BK != 0 || ldx < K || ldw < K ||
        ldx % 8 != 0 || ldw % 8 != 0 || ldy < N ||
        beta != 0.0f && beta != 1.0f)
        return false;
    static const bool swizzle_env = [] {
        const char* value = std::getenv("STRATA_DS4_DENSE_MMAC928_SWIZZLE");
        return value != nullptr && value[0] == '1';
    }();
    const dim3 grid((unsigned) ((N + BN - 1) / BN), (unsigned) ((T + BM128 - 1) / BM128));
    gemm_kernel_m128<<<grid, 512, 0, stream>>>(
        X, (int) ldx, W, (int) ldw, Y, (int) T, (int) N, (int) K, (int) ldy,
        beta != 0.0f ? 1 : 0, swizzle_env && K == 1280 ? 1 : 0);
    return hipGetLastError() == hipSuccess;
}

inline bool launch_split_n128(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                              float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                              float beta, hipStream_t stream) {
    if (!enabled() || !device_ok()) return false;
    if (T < 1 || T > INT32_MAX || N < BN128 || N > INT32_MAX ||
        K < BK || K > INT32_MAX || K % BK != 0 || ldx < K || ldw < K ||
        ldx % 8 != 0 || ldw % 8 != 0 || ldy < N || beta != 0.0f && beta != 1.0f)
        return false;
    static const bool swizzle_env = [] {
        const char* value = std::getenv("STRATA_DS4_DENSE_MMAC928_SWIZZLE");
        return value != nullptr && value[0] == '1';
    }();
    const dim3 grid((unsigned) ((N + BN128 - 1) / BN128), (unsigned) ((T + BM - 1) / BM));
    gemm_kernel_n128_split<<<grid, 512, 0, stream>>>(
        X, (int) ldx, W, (int) ldw, Y, (int) T, (int) N, (int) K, (int) ldy,
        beta != 0.0f ? 1 : 0, swizzle_env ? 1 : 0);
    return hipGetLastError() == hipSuccess;
}

inline bool launch_qb_split_n128(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                                 float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                                 float beta, hipStream_t stream) {
    if (K != 1280 || N < BN128 || N % BN128 != 0) return false;
    return launch_split_n128(X, ldx, W, ldw, Y, T, N, K, ldy, beta, stream);
}

inline bool launch_out16(const uint16_t* X, int64_t ldx, const uint16_t* W, int64_t ldw,
                         uint16_t* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                         hipStream_t stream) {
    return launch_impl<true>(X, ldx, W, ldw, Y, T, N, K, ldy, 0.0f, stream);
}

inline bool launch_out16_group8(const uint16_t* X, int64_t ldx,
                                const uint16_t* const* W, int64_t ldw,
                                uint16_t* Y, int64_t T, int64_t N, int64_t K,
                                int64_t ldy, int64_t x_group_stride,
                                int64_t y_group_stride, hipStream_t stream,
                                int groups = 8) {
    if (!enabled() || !device_ok() || !X || !W || !Y || T < 1 ||
        T > INT32_MAX || N < BN || N > INT32_MAX || K < BK ||
        K > INT32_MAX || K % BK != 0 || ldx < K || ldw < K ||
        ldx % 8 != 0 || ldw % 8 != 0 || ldy < N ||
        x_group_stride < K || y_group_stride < N)
        return false;
    if (groups <= 0 || groups > 8) return false;
    F16Group8 weights{};
    for (int g = 0; g < groups; ++g) {
        if (W[g] == nullptr) return false;
        weights.w[g] = W[g];
    }
    const dim3 grid((unsigned) ((N + BN - 1) / BN),
                    (unsigned) ((T + BM - 1) / BM), (unsigned) groups);
    gemm_out16_group8_kernel<<<grid, THREADS, 0, stream>>>(
        X, (int) ldx, weights, (int) ldw, Y, (int) T, (int) N, (int) K,
        (int) ldy, (size_t) x_group_stride, (size_t) y_group_stride);
    return hipGetLastError() == hipSuccess;
}

inline bool launch_out16_group8_m128(
    const uint16_t* X, int64_t ldx, const uint16_t* const* W, int64_t ldw,
    uint16_t* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
    int64_t x_group_stride, int64_t y_group_stride, hipStream_t stream,
    int groups = 8) {
    if (!enabled() || !device_ok() || !X || !W || !Y || T < 1 ||
        T > INT32_MAX || N < BN || N > INT32_MAX || K < BK ||
        K > INT32_MAX || K % BK != 0 || ldx < K || ldw < K ||
        ldx % 8 != 0 || ldw % 8 != 0 || ldy < N ||
        x_group_stride < K || y_group_stride < N)
        return false;
    if (groups <= 0 || groups > 8) return false;
    F16Group8 weights{};
    for (int g = 0; g < groups; ++g) {
        if (W[g] == nullptr) return false;
        weights.w[g] = W[g];
    }
    const dim3 grid((unsigned) ((N + BN - 1) / BN),
                    (unsigned) ((T + BM128 - 1) / BM128), (unsigned) groups);
    gemm_out16_group8_m128_kernel<<<grid, 512, 0, stream>>>(
        X, (int) ldx, weights, (int) ldw, Y, (int) T, (int) N, (int) K,
        (int) ldy, (size_t) x_group_stride, (size_t) y_group_stride);
    return hipGetLastError() == hipSuccess;
}

inline bool launch_project4(const uint16_t* X, int64_t ldx,
                            const uint16_t* const* W, float* const* Y,
                            const int* N, int64_t T, int64_t K,
                            hipStream_t stream) {
    if (!enabled() || !device_ok() || !X || !W || !Y || !N ||
        T < 1 || T > INT32_MAX || K < BK || K > INT32_MAX ||
        K % BK != 0 || ldx < K || ldx % 8 != 0)
        return false;
    F16Project4 jobs{};
    int max_n = 0;
    for (int g = 0; g < 4; ++g) {
        if (N[g] <= 0) continue;
        if (W[g] == nullptr || Y[g] == nullptr || N[g] > INT32_MAX)
            return false;
        jobs.w[g] = W[g];
        jobs.y[g] = Y[g];
        jobs.n[g] = N[g];
        max_n = max_n > N[g] ? max_n : N[g];
    }
    if (max_n < BN) return false;
    const dim3 grid((unsigned) ((max_n + BN - 1) / BN),
                    (unsigned) ((T + BM - 1) / BM), 4);
    gemm_project4_kernel<<<grid, THREADS, 0, stream>>>(
        X, (int) ldx, jobs, (int) K, (int) T, (int) K);
    return hipGetLastError() == hipSuccess;
}

}  // namespace pfg928
