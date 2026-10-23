// kgu_wsg16.hpp -- kgu-20261014: the gate/up IQ2_S (WSG=16) kernel delivered by the kernel-gu-20261014
// isolated experiment.  Included by moe_mmq.cu (which is already recompiled by build.py), so no new
// compilation unit is needed.
//
// The shipped kernel is gateup_exact::gemm_moe_wsg16_lean<MT,NT,64,LINEAR>.  IQ2_S carries one weight
// scale per 16 k-values and gfx928 has no 16-deep int8 MMA, so each 32-wide k window costs TWO MMACs --
// that 2x is structural and is kept.  Three changes on the supply side, all bit-identical:
//
//  1. SWAP the two MMAC operands (weights -> A, activations -> B).  The C fragment puts lane (lr,lg) on
//     C[row=lr][col=lg+4j], so with the weight row on lr the CONTIGUOUS y dimension is the fast one: the
//     shipped orientation stored 16 rows x 4 columns (16 B) = 16 HALF-filled 32 B sectors per store,
//     the swapped one stores 4 rows x 16 consecutive columns (64 B) = 8 FULL sectors.  The 16-deep split
//     then masks the B (activation) fragment instead of A: same exact 16-deep dot, because the int8 MMAC
//     sum is exact and zeroing either operand's k>=16 removes exactly the same products.
//  2. The per-16 weight scales are staged into LDS as uint4 (4 consecutive scales per row per k0) and read
//     back as ONE float2 per (nt,j) instead of two scalar global reads / two scalar LDS reads.
//  3. The per-32 activation scales (Q8D4::d) are staged into LDS once per k0 instead of being re-read
//     from global inside the ks loop (that read cost 16 uncoalesced sectors per instruction).
//  4. KS=128 (one whole Q8D4 block per k0 step), halving the barrier count vs the shipped KS=64.
//
// Measured on the engine's 500 real gu16 (IQ2_S) per-launch shapes (HCU4, 1400 MHz):
//   shipped dispatch (NT=8 for max_rows>=5500 else NT=4): 1,170.2 ms / 17.14 TMAC/s
//   this kernel, same NT rule                            :   979.3 ms / 20.48 TMAC/s   (-190.9 ms)
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace kgu_wsg16 {

struct Q8D4 { float d[4]; int8_t q[128]; };
static_assert(sizeof(Q8D4) == 144, "Q8 D4 layout");
using I4 = int __attribute__((vector_size(16)));

// ---------------------------------------------------------------- v1/v2: swapped operands + staged scales
template <int MT, int NT, int KS, int ABL = 0>
__global__ void __launch_bounds__(256) w16_kernel(
    const int8_t* __restrict__ codes, const float* __restrict__ wsc, const Q8D4* __restrict__ x,
    const int* __restrict__ bounds, float* __restrict__ y, int N, int K, int rows) {
    constexpr int BM = 16 * MT * 4, BN = 16 * NT, NS = KS / 32, SST = KS / 16, NB = KS / 128;
    static_assert(KS % 128 == 0, "needs KS multiple of 128");
    const int KSG = K / 16;
    constexpr int AV = KS / 16, BV = KS / 16;
    constexpr int AC = (BM * AV + 255) / 256, BC = (BN * BV + 255) / 256;
    constexpr int SC = (BN * (SST / 4) + 255) / 256, DC = (BM * NS + 255) / 256;
    __shared__ __align__(16) int8_t sX[1][BM][KS + 16];   // activations (B operand)
    __shared__ __align__(16) int8_t sW[1][BN][KS + 8];    // weights     (A operand)
    __shared__ float sS[1][BN][SST];                      // per-16 weight scales
    __shared__ float sD[1][BM][NS];                       // per-32 activation scales
    const int t = threadIdx.x, lane = t & 63, wave = t >> 6;
    const int lr = lane & 15, lg = lane >> 4;
    const int e = blockIdx.z;
    const int start = bounds[e], end = bounds[e + 1];
    if (start >= end) return;
    const int m0 = start + blockIdx.x * BM, n0 = blockIdx.y * BN;
    if (m0 >= end) return;
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[i][j][q] = 0.0f;

    uint4 ra[AC], rb[BC], rs[SC]; float rd[DC];
#define KGU_LOAD(k0v, bufv)                                                                    \
    if (ABL & 4) {                                                                               \
        _Pragma("unroll") for (int i = 0; i < AC; i++) ra[i] = make_uint4(0x01010101,0x01010101,0x01010101,0x01010101); \
        _Pragma("unroll") for (int i = 0; i < BC; i++) rb[i] = make_uint4(0x01010101,0x01010101,0x01010101,0x01010101); \
        _Pragma("unroll") for (int i = 0; i < SC; i++) rs[i] = make_uint4(0x3f000000,0x3f000000,0x3f000000,0x3f000000); \
        _Pragma("unroll") for (int i = 0; i < DC; i++) rd[i] = 0.5f;                              \
    } else {                                                                                          \
        const int b128 = (k0v) / 128, off = (k0v) % 128;                                        \
        _Pragma("unroll") for (int i = 0; i < AC; i++) {                                        \
            const int idx = t + i * 256, r = idx / AV, v = idx % AV;                            \
            uint4 val = make_uint4(0, 0, 0, 0);                                                 \
            if (idx < BM * AV && m0 + r < end)                                                  \
                val = *reinterpret_cast<const uint4*>(&x[((size_t)b128 + v / 8) * rows + (m0 + r)].q[(v % 8) * 16]); \
            ra[i] = val;                                                                        \
        }                                                                                       \
        _Pragma("unroll") for (int i = 0; i < BC; i++) {                                        \
            const int idx = t + i * 256, nn = idx / BV, v = idx % BV;                           \
            uint4 val = make_uint4(0, 0, 0, 0);                                                 \
            if (idx < BN * BV && n0 + nn < N)                                                   \
                val = *reinterpret_cast<const uint4*>(&codes[((size_t)e * N + n0 + nn) * K + (k0v) + v * 16]); \
            rb[i] = val;                                                                        \
        }                                                                                       \
        _Pragma("unroll") for (int i = 0; i < SC; i++) {                                        \
            const int idx = t + i * 256, nn = idx / (SST / 4), v = idx % (SST / 4);              \
            uint4 val = make_uint4(0, 0, 0, 0);                                                 \
            if (idx < BN * (SST / 4) && n0 + nn < N)                                            \
                val = *reinterpret_cast<const uint4*>(&wsc[((size_t)e * N + n0 + nn) * KSG + (k0v) / 16 + v * 4]); \
            rs[i] = val;                                                                        \
        }                                                                                       \
        _Pragma("unroll") for (int i = 0; i < DC; i++) {                                        \
            const int idx = t + i * 256, r = idx / NS, s2 = idx % NS;                            \
            rd[i] = (idx < BM * NS && m0 + r < end)                                             \
                        ? x[((size_t)b128 + s2 / 4) * rows + (m0 + r)].d[s2 % 4] : 0.0f;         \
        }                                                                                       \
    }
#define KGU_STORE(bufv)                                                                        \
    {                                                                                          \
        _Pragma("unroll") for (int i = 0; i < AC; i++) { const int idx = t + i * 256, r = idx / AV, v = idx % AV; \
            if (idx < BM * AV) *reinterpret_cast<uint4*>(&sX[bufv][r][v * 16]) = ra[i]; }        \
        _Pragma("unroll") for (int i = 0; i < BC; i++) { const int idx = t + i * 256, nn = idx / BV, v = idx % BV; \
            if (idx < BN * BV) *reinterpret_cast<uint4*>(&sW[bufv][nn][v * 16]) = rb[i]; }       \
        _Pragma("unroll") for (int i = 0; i < SC; i++) { const int idx = t + i * 256, nn = idx / (SST / 4), v = idx % (SST / 4); \
            if (idx < BN * (SST / 4)) *reinterpret_cast<uint4*>(&sS[bufv][nn][v * 4]) = rs[i]; } \
        _Pragma("unroll") for (int i = 0; i < DC; i++) { const int idx = t + i * 256, r = idx / NS, s2 = idx % NS; \
            if (idx < BM * NS) sD[bufv][r][s2] = rd[i]; }                                        \
    }
#define KGU_COMPUTE(bufv)                                                                       \
    {                                                                                          \
        const uint64_t mlo = (lg < 2) ? ~0ull : 0ull;                                            \
        _Pragma("unroll") for (int ks = 0; ks < NS; ks++) {                                      \
            uint64_t xa[MT]; float sax[MT][4];                                                   \
            _Pragma("unroll") for (int mt = 0; mt < MT; mt++) {                                   \
                const int rr = (wave * MT + mt) * 16 + lr;                                       \
                xa[mt] = (ABL & 32) ? 0x0101010101010101ull : *reinterpret_cast<const uint64_t*>(&sX[bufv][rr][ks * 32 + lg * 8]); \
                const int b0 = (wave * MT + mt) * 16;                                            \
                _Pragma("unroll") for (int j = 0; j < 4; j++) sax[mt][j] = (ABL & 16) ? 1.0f : sD[bufv][b0 + lg + 4 * j][ks]; \
            }                                                                                    \
            _Pragma("unroll") for (int nt = 0; nt < NT; nt++) {                                   \
                const uint64_t wa = (ABL & 32) ? 0x0101010101010101ull : *reinterpret_cast<const uint64_t*>(&sW[bufv][nt * 16 + lr][ks * 32 + lg * 8]); \
                const float2 w01 = (ABL & 32) ? make_float2(0.5f,0.5f) : *reinterpret_cast<const float2*>(&sS[bufv][nt * 16 + lr][ks * 2]); \
                _Pragma("unroll") for (int mt = 0; mt < MT; mt++) {                               \
                    const uint64_t blo = xa[mt] & mlo, bhi = xa[mt] & ~mlo;                       \
                    I4 r0 = {0, 0, 0, 0}, r1 = {0, 0, 0, 0};                                      \
                    if (!(ABL & 8)) { r0 = __builtin_amdgcn_mmac_i32_16x16x32i8((long)wa, (long)blo, r0); \
                                      r1 = __builtin_amdgcn_mmac_i32_16x16x32i8((long)wa, (long)bhi, r1); } \
                    _Pragma("unroll") for (int j = 0; j < 4; j++) {                               \
                        if (ABL & 2) { acc[mt][nt][j] += (float)r0[j] + (float)r1[j]; continue; }   \
                        const float s0 = (ABL & 1) ? w01.x : sax[mt][j] * w01.x;                   \
                        const float s1 = (ABL & 1) ? w01.y : sax[mt][j] * w01.y;                   \
                        acc[mt][nt][j] = fmaf((float)r0[j], s0, acc[mt][nt][j]);                   \
                        acc[mt][nt][j] = fmaf((float)r1[j], s1, acc[mt][nt][j]);                   \
                    }                                                                             \
                }                                                                                 \
            }                                                                                     \
        }                                                                                         \
    }

    for (int k0 = 0; k0 < K; k0 += KS) {
        KGU_LOAD(k0, 0) KGU_STORE(0)
        if (!(ABL & 64)) __syncthreads();
        KGU_COMPUTE(0)
        if (!(ABL & 64)) __syncthreads();
    }
#undef KGU_LOAD
#undef KGU_STORE
#undef KGU_COMPUTE
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ar = m0 + (wave * MT + mt) * 16 + lg + 4 * j;   // activation row -> y's row
                const int wr = n0 + nt * 16 + lr;                         // weight row    -> y's column
                if (!(ABL & 128) && ar < end && wr < N) y[(size_t)ar * N + wr] = acc[mt][nt][j];
            }
}

template <int MT, int NT, int KS>
static void launch(const int8_t* codes, const float* wsc, const Q8D4* x, const int* bounds, float* y,
                   int N, int K, int total_rows, int max_rows, int n, hipStream_t s) {
    constexpr int BM = 16 * MT * 4, BN = 16 * NT;
    dim3 g((unsigned)((max_rows + BM - 1) / BM), (unsigned)((N + BN - 1) / BN), (unsigned)n);
    w16_kernel<MT, NT, KS, 0><<<g, 256, 0, s>>>(codes, wsc, x, bounds, y, N, K, total_rows);
}

}  // namespace kgu_wsg16
