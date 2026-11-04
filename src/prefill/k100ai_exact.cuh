// K100AI gfx928 exact expert products.
//
// These kernels keep the model's canonical quantized weights and the production
// Q8_1 activation layout.  They organize the same integer products on the HCU's
// 16x16x32 int8 MMAC instruction, with the same FP32 accumulation order as the
// production dp4a kernels.  They are compiled only into the gfx928 backend.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace strata::prefill::mmq::k100ai_exact {

struct alignas(16) Q8D4 {
    float d[4];
    int8_t q[128];
};
static_assert(sizeof(Q8D4) == 144, "K100AI Q8_1 D4 layout");

using I4 = int __attribute__((vector_size(16)));

__device__ __forceinline__ uint16_t out_hf(float f) {
    return __half_as_ushort(
        __float2half_rn(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f)));
}

// IQ3_XXS block (98 bytes): f16 scale, 64 grid selects, 32 sign scales.
__global__ void expand_iq3xxs(const uint8_t* __restrict__ w, int8_t* __restrict__ codes,
                              float* __restrict__ scales, int blocks_per_expert, int experts,
                              int rows, int cols) {
    const int linear = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= experts * blocks_per_expert * 32) return;

    const int gid = linear / 32;
    const int unit = linear & 31;
    const int expert = gid / blocks_per_expert;
    const int block = gid % blocks_per_expert;
    const int blocks_per_row = cols / 256;
    const int row = block / blocks_per_row;
    int8_t* out =
        codes + ((size_t) expert * rows + row) * cols + (size_t) (block % blocks_per_row) * 256;
    const uint8_t* src = w + (size_t) gid * 98;
    const float d = __half2float(*reinterpret_cast<const __half*>(src));
    const uint8_t* qs = src + 2;
    const uint8_t* signs = qs + 64;
    float* out_scales =
        scales + ((size_t) expert * rows + row) * (cols / 32) + (size_t) (block % blocks_per_row) * 8;

    const int ib = unit >> 2;
    const int lane = unit & 3;
    uint32_t aux;
    __builtin_memcpy(&aux, signs + 4 * ib, sizeof(aux));
    const uint8_t sign = ksigns_iq2xs[(aux >> (7 * lane)) & 127];
    const uint8_t* grid0 =
        reinterpret_cast<const uint8_t*>(iq3xxs_grid + qs[8 * ib + 2 * lane]);
    const uint8_t* grid1 =
        reinterpret_cast<const uint8_t*>(iq3xxs_grid + qs[8 * ib + 2 * lane + 1]);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo |= (uint32_t) (uint8_t) (grid0[j] * ((sign & (1u << j)) ? -1 : 1)) << (8 * j);
        hi |= (uint32_t) (uint8_t) (grid1[j] * ((sign & (1u << (j + 4))) ? -1 : 1)) << (8 * j);
    }
    *reinterpret_cast<uint32_t*>(out + ib * 32 + lane * 8) = lo;
    *reinterpret_cast<uint32_t*>(out + ib * 32 + lane * 8 + 4) = hi;
    if (lane == 0) out_scales[ib] = d * (0.5f + (float) (aux >> 28)) * 0.5f;
}

// IQ4_NL block (18 bytes): f16 scale and 16 packed nibbles.
__global__ void expand_iq4nl(const uint8_t* __restrict__ w, int8_t* __restrict__ codes,
                             float* __restrict__ scales, int blocks, int experts,
                             const int32_t* __restrict__ bounds) {
    const int expert = blockIdx.z;
    if (bounds[expert] == bounds[expert + 1]) return;
    const int blocks_per_expert = blocks / experts;
    const int local_word = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (local_word >= blocks_per_expert * 4) return;
    const int word = expert * blocks_per_expert * 4 + local_word;
    const int block = word / 4;
    const int segment = word & 3;
    const uint8_t* src = w + (size_t) block * 18;
    if (segment == 0)
        scales[block] = __half2float(*reinterpret_cast<const __half*>(src));
    uint64_t value = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned q = src[2 + (segment & 1) * 8 + i];
        const unsigned index = (q >> (segment < 2 ? 0 : 4)) & 15;
        value |= (uint64_t) (uint8_t) kvalues_iq4nl[index] << (i * 8);
    }
    reinterpret_cast<uint64_t*>(codes)[word] = value;
}

// One thread per 32-value unit for the remaining gate/up i-quants.  The
// codebooks are the pinned ggml tables already included by common.cuh.
__global__ void expand_iq2xxs(const uint8_t* __restrict__ w, int8_t* __restrict__ codes,
                              float* __restrict__ scales, int blocks_per_expert, int experts,
                              int rows, int cols) {
    const int linear = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= experts * blocks_per_expert * 32) return;
    const int gid = linear / 32, unit = linear & 31;
    const int expert = gid / blocks_per_expert;
    const int block = gid % blocks_per_expert;
    const int blocks_per_row = cols / 256;
    const int row = block / blocks_per_row;
    int8_t* out =
        codes + ((size_t) expert * rows + row) * cols + (size_t) (block % blocks_per_row) * 256;
    const uint8_t* src = w + (size_t) gid * 66;
    const float d = __half2float(*reinterpret_cast<const __half*>(src));
    const uint16_t* q2 = reinterpret_cast<const uint16_t*>(src + 2);
    const int ib = unit >> 2, lane = unit & 3;
    const uint32_t aux = (uint32_t) q2[4 * ib + 2] | ((uint32_t) q2[4 * ib + 3] << 16);
    const uint8_t* grid =
        reinterpret_cast<const uint8_t*>(iq2xxs_grid + (q2[4 * ib + lane] & 0xff));
    const uint8_t sign = ksigns_iq2xs[(aux >> (7 * lane)) & 127];
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo |= (uint32_t) (uint8_t) (grid[j] * ((sign & (1u << j)) ? -1 : 1)) << (8 * j);
        hi |= (uint32_t) (uint8_t) (grid[4 + j] * ((sign & (1u << (j + 4))) ? -1 : 1))
              << (8 * j);
    }
    *reinterpret_cast<uint32_t*>(out + ib * 32 + lane * 8) = lo;
    *reinterpret_cast<uint32_t*>(out + ib * 32 + lane * 8 + 4) = hi;
    if (lane == 0)
        scales[((size_t) expert * rows + row) * (cols / 32) +
               (size_t) (block % blocks_per_row) * 8 + ib] =
            d * (0.5f + (float) (aux >> 28)) * 0.25f;
}

__global__ void expand_iq2xs(const uint8_t* __restrict__ w, int8_t* __restrict__ codes,
                             float* __restrict__ scales, int blocks_per_expert, int experts,
                             int rows, int cols) {
    const int linear = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= experts * blocks_per_expert * 32) return;
    const int gid = linear / 32, unit = linear & 31;
    const int expert = gid / blocks_per_expert;
    const int block = gid % blocks_per_expert;
    const int blocks_per_row = cols / 256;
    const int row = block / blocks_per_row;
    int8_t* out =
        codes + ((size_t) expert * rows + row) * cols + (size_t) (block % blocks_per_row) * 256;
    const uint8_t* src = w + (size_t) gid * 74;
    const float d = __half2float(*reinterpret_cast<const __half*>(src));
    const uint16_t* q2 = reinterpret_cast<const uint16_t*>(src + 2);
    const uint8_t* scale_bytes = src + 66;
    const int ib = unit >> 2, lane = unit & 3;
    const uint16_t q = q2[4 * ib + lane];
    const uint8_t* grid = reinterpret_cast<const uint8_t*>(iq2xs_grid + (q & 511));
    const uint8_t sign = ksigns_iq2xs[q >> 9];
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo |= (uint32_t) (uint8_t) (grid[j] * ((sign & (1u << j)) ? -1 : 1)) << (8 * j);
        hi |= (uint32_t) (uint8_t) (grid[4 + j] * ((sign & (1u << (j + 4))) ? -1 : 1))
              << (8 * j);
    }
    *reinterpret_cast<uint32_t*>(out + ib * 32 + lane * 8) = lo;
    *reinterpret_cast<uint32_t*>(out + ib * 32 + lane * 8 + 4) = hi;
    float* out_scales =
        scales + ((size_t) expert * rows + row) * (cols / 16) +
        (size_t) (block % blocks_per_row) * 16;
    if (lane == 0)
        out_scales[2 * ib] =
            d * (0.5f + (float) (scale_bytes[ib] & 0x0f)) * 0.25f;
    if (lane == 2)
        out_scales[2 * ib + 1] =
            d * (0.5f + (float) (scale_bytes[ib] >> 4)) * 0.25f;
}

__global__ void expand_iq3s(const uint8_t* __restrict__ w, int8_t* __restrict__ codes,
                            float* __restrict__ scales, int blocks_per_expert, int experts,
                            int rows, int cols) {
    const int linear = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= experts * blocks_per_expert * 32) return;
    const int gid = linear / 32, unit = linear & 31;
    const int expert = gid / blocks_per_expert;
    const int block = gid % blocks_per_expert;
    const int blocks_per_row = cols / 256;
    const int row = block / blocks_per_row;
    int8_t* out =
        codes + ((size_t) expert * rows + row) * cols + (size_t) (block % blocks_per_row) * 256;
    const uint8_t* src = w + (size_t) gid * 110;
    const float d = __half2float(*reinterpret_cast<const __half*>(src));
    const uint8_t* qs = src + 2;
    const uint8_t* qh = qs + 64;
    const uint8_t* signs = qh + 8;
    const uint8_t* scale_bytes = signs + 32;
    const int it = unit >> 3, half = (unit >> 2) & 1, lane = unit & 3;
    const uint8_t qh_byte = qh[2 * it + half];
    const int qbase = 16 * it + 8 * half, sbase = 8 * it + 4 * half;
    const uint8_t sign = signs[sbase + lane];
    const uint8_t* grid0 = reinterpret_cast<const uint8_t*>(
        iq3s_grid + (qs[qbase + 2 * lane] |
                     (((unsigned) qh_byte << (8 - 2 * lane)) & 256u)));
    const uint8_t* grid1 = reinterpret_cast<const uint8_t*>(
        iq3s_grid + (qs[qbase + 2 * lane + 1] |
                     (((unsigned) qh_byte << (7 - 2 * lane)) & 256u)));
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo |= (uint32_t) (uint8_t) (grid0[j] * ((sign & (1u << j)) ? -1 : 1)) << (8 * j);
        hi |= (uint32_t) (uint8_t) (grid1[j] * ((sign & (1u << (j + 4))) ? -1 : 1))
              << (8 * j);
    }
    *reinterpret_cast<uint32_t*>(out + it * 64 + half * 32 + lane * 8) = lo;
    *reinterpret_cast<uint32_t*>(out + it * 64 + half * 32 + lane * 8 + 4) = hi;
    if (lane == 0) {
        const uint8_t s = scale_bytes[it];
        scales[((size_t) expert * rows + row) * (cols / 32) +
               (size_t) (block % blocks_per_row) * 8 + 2 * it + half] =
            d * (1.0f + 2.0f * (float) (half ? (s >> 4) : (s & 0x0f)));
    }
}

__global__ void expand_iq4xs(const uint8_t* __restrict__ w, int8_t* __restrict__ codes,
                             float* __restrict__ scales, int blocks_per_expert, int experts,
                             int rows, int cols) {
    const int linear = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= experts * blocks_per_expert * 32) return;
    const int gid = linear / 32, unit = linear & 31;
    const int expert = gid / blocks_per_expert;
    const int block = gid % blocks_per_expert;
    const int blocks_per_row = cols / 256;
    const int row = block / blocks_per_row;
    int8_t* out =
        codes + ((size_t) expert * rows + row) * cols + (size_t) (block % blocks_per_row) * 256;
    const uint8_t* src = w + (size_t) gid * 136;
    const float d = __half2float(*reinterpret_cast<const __half*>(src));
    uint16_t scale_high;
    __builtin_memcpy(&scale_high, src + 2, sizeof(scale_high));
    const uint8_t* scale_low = src + 4;
    const uint8_t* qs = src + 8;
    const int ib = unit >> 2, lane = unit & 3;
    const bool high = lane >= 2;
    const uint8_t* q8 = qs + 16 * ib + ((lane & 1) << 3);
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint8_t q0 = q8[j], q1 = q8[4 + j];
        lo |= (uint32_t) (uint8_t) kvalues_iq4nl[high ? (q0 >> 4) : (q0 & 0x0f)] << (8 * j);
        hi |= (uint32_t) (uint8_t) kvalues_iq4nl[high ? (q1 >> 4) : (q1 & 0x0f)] << (8 * j);
    }
    *reinterpret_cast<uint32_t*>(out + ib * 32 + lane * 8) = lo;
    *reinterpret_cast<uint32_t*>(out + ib * 32 + lane * 8 + 4) = hi;
    if (lane == 0) {
        const int ls = ((scale_low[ib / 2] >> (4 * (ib & 1))) & 0x0f) |
                       (((int) ((scale_high >> (2 * ib)) & 3)) << 4);
        scales[((size_t) expert * rows + row) * (cols / 32) +
               (size_t) (block % blocks_per_row) * 8 + ib] = d * (float) (ls - 32);
    }
}

__global__ void expand_q2_0(const uint8_t* __restrict__ w, int8_t* __restrict__ codes,
                            float* __restrict__ scales, int blocks, int experts,
                            const int32_t* __restrict__ bounds) {
    const int expert = blockIdx.z;
    if (bounds[expert] == bounds[expert + 1]) return;
    const int blocks_per_expert = blocks / experts;
    const int local_word = (int) blockIdx.x * blockDim.x + threadIdx.x;
    if (local_word >= blocks_per_expert * 8) return;
    const int word = expert * blocks_per_expert * 8 + local_word;
    const int block = word / 8;
    const int segment = word & 7;
    const uint8_t* src = w + (size_t) block * 18;
    if (segment == 0) {
        const float d = __half2float(*reinterpret_cast<const __half*>(src));
        scales[block * 2] = d;
        scales[block * 2 + 1] = d;
    }
    uint64_t value = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned q = src[2 + 2 * segment + (i >> 2)];
        const unsigned index = (q >> ((i & 3) * 2)) & 3;
        value |= (uint64_t) (uint8_t) ((int) index - 1) << (i * 8);
    }
    reinterpret_cast<uint64_t*>(codes)[word] = value;
}

// IQ3_XXS gate/up: 128 rows x 128 columns per CTA, eight 64-lane
// wavefronts.  The contraction remains ascending over K and accumulates in
// FP32, matching the production kernel's output bits.
__global__ void __launch_bounds__(512) gateup_iq3xxs(
    const int8_t* __restrict__ codes, const float* __restrict__ scales,
    const Q8D4* __restrict__ x, const int32_t* __restrict__ bounds,
    float* __restrict__ y, int rows, int cols, int total_rows) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int KS = 64;
    constexpr int NT = 8;
    __shared__ __align__(16) int8_t s_x[BM][KS + 16];
    __shared__ __align__(16) int8_t s_w[BN][KS + 8];
    __shared__ float s_scales[BN][KS / 32];

    const int t = threadIdx.x;
    const int lane = t & 63;
    const int wave = t >> 6;
    const int lr = lane & 15;
    const int lg = lane >> 4;
    const int expert = blockIdx.z;
    const int start = bounds[expert], end = bounds[expert + 1];
    if (start >= end) return;
    const int row0 = start + blockIdx.x * BM;
    const int col0 = blockIdx.y * BN;
    if (row0 >= end) return;

    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[nt][j] = 0.0f;

    for (int k0 = 0; k0 < cols; k0 += KS) {
        const int block128 = k0 / 128;
        const int offset = k0 % 128;
        for (int i = t; i < BM * (KS / 16); i += 512) {
            const int r = i / (KS / 16), v = i % (KS / 16);
            uint4 value = make_uint4(0, 0, 0, 0);
            if (row0 + r < end)
                value = *reinterpret_cast<const uint4*>(
                    &x[(size_t) block128 * total_rows + row0 + r].q[offset + v * 16]);
            *reinterpret_cast<uint4*>(&s_x[r][v * 16]) = value;
        }
        for (int i = t; i < BN * (KS / 16); i += 512) {
            const int c = i / (KS / 16), v = i % (KS / 16);
            uint4 value = make_uint4(0, 0, 0, 0);
            if (col0 + c < rows)
                value = *reinterpret_cast<const uint4*>(
                    &codes[((size_t) expert * rows + col0 + c) * cols + k0 + v * 16]);
            *reinterpret_cast<uint4*>(&s_w[c][v * 16]) = value;
        }
        for (int i = t; i < BN * (KS / 32); i += 512) {
            const int c = i / (KS / 32), ks = i % (KS / 32);
            s_scales[c][ks] =
                col0 + c < rows
                    ? scales[((size_t) expert * rows + col0 + c) * (cols / 32) + k0 / 32 + ks]
                    : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int ks = 0; ks < KS / 32; ++ks) {
            const int sub = offset / 32 + ks;
            const int r = wave * 16 + lr;
            const uint64_t a = *reinterpret_cast<const uint64_t*>(&s_x[r][ks * 32 + lg * 8]);
            const float activation_scale =
                row0 + r < end
                    ? x[(size_t) (k0 / 128) * total_rows + row0 + r].d[sub]
                    : 0.0f;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint64_t b =
                    *reinterpret_cast<const uint64_t*>(&s_w[nt * 16 + lr][ks * 32 + lg * 8]);
                I4 product = {0, 0, 0, 0};
                product =
                    __builtin_amdgcn_mmac_i32_16x16x32i8((long) a, (long) b, product);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[nt][j] =
                        fmaf((float) product[j],
                             activation_scale * s_scales[nt * 16 + lg + 4 * j][ks], acc[nt][j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = blockIdx.x * BM + wave * 16 + lr;
            const int c = col0 + nt * 16 + lg + 4 * j;
            if (row0 + (wave * 16 + lr) < end && c < rows)
                y[(size_t) (start + r) * rows + c] = acc[nt][j];
        }
}

// Gate/up formats with a scale per 16 K values.  A 32-deep MMAC is split
// into the two exact 16-deep halves; the activation rows and integer products
// are unchanged, only the masked fragment makes each accumulation exact.
__global__ void __launch_bounds__(256) gateup_wsg16(
    const int8_t* __restrict__ codes, const float* __restrict__ scales,
    const Q8D4* __restrict__ x, const int32_t* __restrict__ bounds,
    float* __restrict__ y, int rows, int cols, int total_rows) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int KS = 64;
    constexpr int NT = 4;
    __shared__ __align__(16) int8_t s_x[BM][KS + 16];
    __shared__ __align__(16) int8_t s_w[BN][KS + 8];
    __shared__ float s_scales[BN][KS / 16];
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const int lr = lane & 15, lg = lane >> 4;
    const int expert = blockIdx.z;
    const int start = bounds[expert], end = bounds[expert + 1];
    if (start >= end) return;
    const int row0 = start + blockIdx.x * BM;
    const int col0 = blockIdx.y * BN;
    if (row0 >= end) return;
    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[nt][j] = 0.0f;

    for (int k0 = 0; k0 < cols; k0 += KS) {
        const int block128 = k0 / 128;
        const int offset = k0 % 128;
        for (int i = t; i < BM * (KS / 16); i += 256) {
            const int r = i / (KS / 16), v = i % (KS / 16);
            uint4 value = make_uint4(0, 0, 0, 0);
            if (row0 + r < end)
                value = *reinterpret_cast<const uint4*>(
                    &x[(size_t) block128 * total_rows + row0 + r].q[offset + v * 16]);
            *reinterpret_cast<uint4*>(&s_x[r][v * 16]) = value;
        }
        for (int i = t; i < BN * (KS / 16); i += 256) {
            const int c = i / (KS / 16), v = i % (KS / 16);
            uint4 value = make_uint4(0, 0, 0, 0);
            if (col0 + c < rows)
                value = *reinterpret_cast<const uint4*>(
                    &codes[((size_t) expert * rows + col0 + c) * cols + k0 + v * 16]);
            *reinterpret_cast<uint4*>(&s_w[c][v * 16]) = value;
        }
        for (int i = t; i < BN * (KS / 16); i += 256) {
            const int c = i / (KS / 16), ks = i % (KS / 16);
            s_scales[c][ks] =
                col0 + c < rows
                    ? scales[((size_t) expert * rows + col0 + c) * (cols / 16) + k0 / 16 + ks]
                    : 0.0f;
        }
        __syncthreads();
        const uint64_t low_mask = lg < 2 ? ~0ull : 0ull;
#pragma unroll
        for (int ks = 0; ks < KS / 32; ++ks) {
            const int sub = offset / 32 + ks;
            const int r = wave * 16 + lr;
            const uint64_t a = *reinterpret_cast<const uint64_t*>(&s_x[r][ks * 32 + lg * 8]);
            const float activation_scale =
                row0 + r < end
                    ? x[(size_t) block128 * total_rows + row0 + r].d[sub]
                    : 0.0f;
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint64_t b =
                    *reinterpret_cast<const uint64_t*>(&s_w[nt * 16 + lr][ks * 32 + lg * 8]);
                I4 product0 = {0, 0, 0, 0}, product1 = {0, 0, 0, 0};
                product0 = __builtin_amdgcn_mmac_i32_16x16x32i8(
                    (long) (a & low_mask), (long) b, product0);
                product1 = __builtin_amdgcn_mmac_i32_16x16x32i8(
                    (long) (a & ~low_mask), (long) b, product1);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[nt][j] =
                        fmaf((float) product0[j],
                             activation_scale * s_scales[nt * 16 + lg + 4 * j][ks * 2],
                             acc[nt][j]);
                    acc[nt][j] =
                        fmaf((float) product1[j],
                             activation_scale * s_scales[nt * 16 + lg + 4 * j][ks * 2 + 1],
                             acc[nt][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = blockIdx.x * BM + wave * 16 + lr;
            const int c = col0 + nt * 16 + lg + 4 * j;
            if (row0 + (wave * 16 + lr) < end && c < rows)
                y[(size_t) (start + r) * rows + c] = acc[nt][j];
        }
}

// IQ4_NL down: weight rows are operand A and activation rows operand B, so
// the C fragment writes four rows by sixteen contiguous columns.  This is the
// exact restructure measured by the earlier K100AI release; the integer dot
// and FP32 accumulation order are unchanged.
__global__ void __launch_bounds__(256) down_iq4nl(
    const int8_t* __restrict__ codes, const float* __restrict__ scales,
    const Q8D4* __restrict__ x, const int32_t* __restrict__ bounds,
    float* __restrict__ y, int rows, int cols, int total_rows) {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int KS = 64;
    constexpr int NT = 8;
    constexpr int NS = KS / 32;
    constexpr int AC = (BM * (KS / 16) + 255) / 256;
    constexpr int BC = (BN * (KS / 16) + 255) / 256;
    constexpr int SC = (BN * NS + 255) / 256;
    constexpr int DC = (BM * NS + 255) / 256;
    __shared__ __align__(16) int8_t s_x[2][BM][KS + 16];
    __shared__ __align__(16) int8_t s_w[2][BN][KS + 8];
    __shared__ float s_ws[2][BN][NS];
    __shared__ float s_xs[2][BM][NS];

    const int t = threadIdx.x;
    const int lane = t & 63;
    const int wave = t >> 6;
    const int lr = lane & 15;
    const int lg = lane >> 4;
    const int expert = blockIdx.z;
    const int start = bounds[expert], end = bounds[expert + 1];
    if (start >= end) return;
    const int row0 = start + blockIdx.x * BM;
    const int col0 = blockIdx.y * BN;
    if (row0 >= end) return;

    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[nt][j] = 0.0f;

    uint4 ra[AC], rb[BC];
    float rs[SC], rd[DC];
    constexpr int AX = KS / 16;
    constexpr int BX = KS / 16;
#pragma unroll
    for (int i = 0; i < AC; ++i) {
        const int index = t + i * 256;
        const int r = index / AX, v = index % AX;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (index < BM * AX && row0 + r < end)
            value = *reinterpret_cast<const uint4*>(&x[row0 + r].q[v * 16]);
        ra[i] = value;
    }
#pragma unroll
    for (int i = 0; i < BC; ++i) {
        const int index = t + i * 256;
        const int c = index / BX, v = index % BX;
        uint4 value = make_uint4(0, 0, 0, 0);
        if (index < BN * BX && col0 + c < rows)
            value = *reinterpret_cast<const uint4*>(
                &codes[((size_t) expert * rows + col0 + c) * cols + v * 16]);
        rb[i] = value;
    }
#pragma unroll
    for (int i = 0; i < SC; ++i) {
        const int index = t + i * 256;
        const int c = index / NS, ks = index % NS;
        rs[i] = index < BN * NS && col0 + c < rows
                    ? scales[((size_t) expert * rows + col0 + c) * (cols / 32) + ks]
                    : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < DC; ++i) {
        const int index = t + i * 256;
        const int r = index / NS, ks = index % NS;
        rd[i] = index < BM * NS && row0 + r < end ? x[row0 + r].d[ks] : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < AC; ++i) {
        const int index = t + i * 256, r = index / AX, v = index % AX;
        if (index < BM * AX) *reinterpret_cast<uint4*>(&s_x[0][r][v * 16]) = ra[i];
    }
#pragma unroll
    for (int i = 0; i < BC; ++i) {
        const int index = t + i * 256, c = index / BX, v = index % BX;
        if (index < BN * BX) *reinterpret_cast<uint4*>(&s_w[0][c][v * 16]) = rb[i];
    }
#pragma unroll
    for (int i = 0; i < SC; ++i) {
        const int index = t + i * 256, c = index / NS, ks = index % NS;
        if (index < BN * NS) s_ws[0][c][ks] = rs[i];
    }
#pragma unroll
    for (int i = 0; i < DC; ++i) {
        const int index = t + i * 256, r = index / NS, ks = index % NS;
        if (index < BM * NS) s_xs[0][r][ks] = rd[i];
    }
    __syncthreads();

    for (int k0 = 0; k0 < cols; k0 += KS) {
        const int buffer = (k0 / KS) & 1;
        const int next = k0 + KS;
        if (next < cols) {
            const int block128 = next / 128;
            const int offset = next % 128;
#pragma unroll
            for (int i = 0; i < AC; ++i) {
                const int index = t + i * 256, r = index / AX, v = index % AX;
                uint4 value = make_uint4(0, 0, 0, 0);
                if (index < BM * AX && row0 + r < end)
                    value = *reinterpret_cast<const uint4*>(
                        &x[(size_t) block128 * total_rows + row0 + r].q[offset + v * 16]);
                ra[i] = value;
            }
#pragma unroll
            for (int i = 0; i < BC; ++i) {
                const int index = t + i * 256, c = index / BX, v = index % BX;
                uint4 value = make_uint4(0, 0, 0, 0);
                if (index < BN * BX && col0 + c < rows)
                    value = *reinterpret_cast<const uint4*>(
                        &codes[((size_t) expert * rows + col0 + c) * cols + next + v * 16]);
                rb[i] = value;
            }
#pragma unroll
            for (int i = 0; i < SC; ++i) {
                const int index = t + i * 256, c = index / NS, ks = index % NS;
                rs[i] = index < BN * NS && col0 + c < rows
                            ? scales[((size_t) expert * rows + col0 + c) * (cols / 32) +
                                     next / 32 + ks]
                            : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < DC; ++i) {
                const int index = t + i * 256, r = index / NS, ks = index % NS;
                rd[i] = index < BM * NS && row0 + r < end
                            ? x[(size_t) block128 * total_rows + row0 + r].d[offset / 32 + ks]
                            : 0.0f;
            }
        }

#pragma unroll
        for (int ks = 0; ks < NS; ++ks) {
            const int r = wave * 16 + lr;
            const uint64_t a =
                *reinterpret_cast<const uint64_t*>(&s_x[buffer][r][ks * 32 + lg * 8]);
            float a_scales[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                a_scales[j] = s_xs[buffer][wave * 16 + lg + 4 * j][ks];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint64_t b =
                    *reinterpret_cast<const uint64_t*>(&s_w[buffer][nt * 16 + lr][ks * 32 + lg * 8]);
                const float weight_scale = s_ws[buffer][nt * 16 + lr][ks];
                I4 product = {0, 0, 0, 0};
                product =
                    __builtin_amdgcn_mmac_i32_16x16x32i8((long) b, (long) a, product);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[nt][j] =
                        fmaf((float) product[j], a_scales[j] * weight_scale, acc[nt][j]);
            }
        }
        if (next < cols) {
            const int other = buffer ^ 1;
#pragma unroll
            for (int i = 0; i < AC; ++i) {
                const int index = t + i * 256, r = index / AX, v = index % AX;
                if (index < BM * AX)
                    *reinterpret_cast<uint4*>(&s_x[other][r][v * 16]) = ra[i];
            }
#pragma unroll
            for (int i = 0; i < BC; ++i) {
                const int index = t + i * 256, c = index / BX, v = index % BX;
                if (index < BN * BX)
                    *reinterpret_cast<uint4*>(&s_w[other][c][v * 16]) = rb[i];
            }
#pragma unroll
            for (int i = 0; i < SC; ++i) {
                const int index = t + i * 256, c = index / NS, ks = index % NS;
                if (index < BN * NS) s_ws[other][c][ks] = rs[i];
            }
#pragma unroll
            for (int i = 0; i < DC; ++i) {
                const int index = t + i * 256, r = index / NS, ks = index % NS;
                if (index < BM * NS) s_xs[other][r][ks] = rd[i];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = row0 + wave * 16 + lg + 4 * j;
            const int c = col0 + nt * 16 + lr;
            if (r < end && c < rows) y[(size_t) r * rows + c] = acc[nt][j];
        }
}

}  // namespace strata::prefill::mmq::k100ai_exact
