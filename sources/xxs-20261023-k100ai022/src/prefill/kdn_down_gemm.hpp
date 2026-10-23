// kdn_down_gemm.hpp -- kdn-20261014 / kgu-20261014: the down GEMM delivered by the kernel-down-20261014
// isolated experiment.  Included by iq4_preexpand_exact.hpp (which moe_mmq.cu includes), so no new
// compilation unit is needed.
//
// The shipped MMAC and the shipped per-output fmaf chain, with three changes that keep the result
// BIT-IDENTICAL (verified vs the shipped kernel: 1,147,586,560 output elements over 35 real per-launch
// shapes byte-for-byte, and two independent 64-bit per-case hashes over all 1,283 shapes: ndiff=0):
//
//  1. The two MMAC operands are SWAPPED: weight rows feed the A operand and activation rows feed the B
//     operand, so the C fragment (lane (lr,lg) -> C[row=lr][col=lg+4j]) puts the CONTIGUOUS y dimension
//     (weight row == y's column) on lr.  The shipped orientation wrote, per store, 16 rows of 4 columns
//     each -- 16 partially filled 32-byte sectors 5120 B apart -- and removing that store measured ~58%
//     of the down GEMM.  Swapping makes it 4 rows x 16 consecutive columns = 4 FULL sectors.
//     Both operands use the identical per-lane k coverage (k = lg*8 .. lg*8+7), the int8 MMAC sum is
//     exact, and the scales are applied in the same order (activation scale * weight scale, then fmaf).
//  2. The per-32 activation scales (Q8D4::d) are staged into LDS once per k0 instead of being re-read
//     from global inside the ks loop.
//  3. v11 (MT=1 only) software double-buffers sX/sW/sS/sD: each k0's global loads are issued into
//     registers before the previous k0's compute, then written to the alternate LDS buffer.
//
// Measured on the 1,283 real per-launch shapes of the engine's T=30,602 prefill (HCU4, 1400 MHz):
//   shipped down_gemm_launch, STRATA_DOWN_MT=1 : 1,273.4 ms / 18.89 TMAC/s
//   this kernel,            STRATA_DOWN_MT=1 :   971.4 ms / 24.78 TMAC/s   (1.311x)
//   MT=2: v8 1,019.8 ms (v11's double buffer would spill)
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace kdn_down {

struct Q8D4 { float d[4]; int8_t q[128]; };
static_assert(sizeof(Q8D4) == 144, "Q8 D4 layout");
using I4 = int __attribute__((vector_size(16)));

// identical to gateup_exact::out_hf
__device__ __forceinline__ uint16_t out_hf(float f) {
    return __half_as_ushort(__float2half_rn(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f)));
}

// ---- v8: v7 + the activation scales staged in LDS.  Used for MT=2, where v11's double buffer
// ----     overflows the register budget and spills (measured 2,802.8 ms vs v8's 1,019.8 ms).
template <int MT, int NT, int KS, int ABL = 0, bool F32 = false>
__global__ void __launch_bounds__(256) kdn_v8_kernel(
    const int8_t* __restrict__ codes, const float* __restrict__ wsc,
    const Q8D4* __restrict__ x, const int* __restrict__ bounds,
    void* __restrict__ y, int N, int K, int rows) {
    constexpr int BM = 16 * MT * 4, BN = 16 * NT, NS = KS / 32;
    const int KSG = K / 32;
    __shared__ __align__(16) int8_t sX[BM][KS + 16];
    __shared__ __align__(16) int8_t sW[BN][KS + 8];
    __shared__ float sS[BN][NS];
    __shared__ float sD[BM][NS];      // activation scales (d[off/32+ks]) staged once per k0
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
    for (int k0 = 0; k0 < K; k0 += KS) {
        const int b128 = k0 / 128, off = k0 % 128;
        {
            constexpr int AV = KS / 16;
            for (int i = t; i < BM * AV; i += 256) {
                const int r = i / AV, v = i % AV;
                uint4 val = make_uint4(0, 0, 0, 0);
                if (m0 + r < end) val = *reinterpret_cast<const uint4*>(&x[(size_t)b128 * rows + (m0 + r)].q[off + v * 16]);
                *reinterpret_cast<uint4*>(&sX[r][v * 16]) = val;
            }
            for (int i = t; i < BM * NS; i += 256) {
                const int r = i / NS, s2 = i % NS;
                sD[r][s2] = (m0 + r < end) ? x[(size_t)b128 * rows + (m0 + r)].d[off / 32 + s2] : 0.0f;
            }
            constexpr int BV = KS / 16;
            for (int i = t; i < BN * BV; i += 256) {
                const int nn = i / BV, v = i % BV;
                uint4 val = make_uint4(0, 0, 0, 0);
                if (n0 + nn < N) val = *reinterpret_cast<const uint4*>(&codes[((size_t)e * N + n0 + nn) * K + k0 + v * 16]);
                *reinterpret_cast<uint4*>(&sW[nn][v * 16]) = val;
            }
            for (int i = t; i < BN * NS; i += 256) {
                const int nn = i / NS, ks2 = i % NS;
                sS[nn][ks2] = (n0 + nn < N) ? wsc[((size_t)e * N + n0 + nn) * KSG + k0 / 32 + ks2] : 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < NS; ks++) {
            uint64_t xa[MT];
            float sax[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                const int rr = (wave * MT + mt) * 16 + lr;
                xa[mt] = *reinterpret_cast<const uint64_t*>(&sX[rr][ks * 32 + lg * 8]);
                const int b = (wave * MT + mt) * 16;
#pragma unroll
                for (int j = 0; j < 4; j++) sax[mt][j] = sD[b + lg + 4 * j][ks];
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const uint64_t wa = *reinterpret_cast<const uint64_t*>(&sW[nt * 16 + lr][ks * 32 + lg * 8]);
                const float sw = sS[nt * 16 + lr][ks];
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    I4 r = {0, 0, 0, 0};
                    r = __builtin_amdgcn_mmac_i32_16x16x32i8((long)wa, (long)xa[mt], r);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        acc[mt][nt][j] = fmaf((float)r[j], sax[mt][j] * sw, acc[mt][nt][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ar = m0 + (wave * MT + mt) * 16 + lg + 4 * j;
                const int wr = n0 + nt * 16 + lr;
                if (ar < end && wr < N) {
                    if constexpr (F32) ((float*)y)[(size_t)ar * N + wr] = acc[mt][nt][j];
                    else ((uint16_t*)y)[(size_t)ar * N + wr] = out_hf(acc[mt][nt][j]);
                }
            }
}

template <int MT, int NT, int KS, int ABL = 0, bool F32 = false>
__global__ void __launch_bounds__(256) kdn_v11_kernel(
    const int8_t* __restrict__ codes, const float* __restrict__ wsc,
    const Q8D4* __restrict__ x, const int* __restrict__ bounds,
    void* __restrict__ y, int N, int K, int rows) {
    constexpr int BM = 16 * MT * 4, BN = 16 * NT, NS = KS / 32;
    const int KSG = K / 32;
    constexpr int AV = KS / 16, BV = KS / 16;
    constexpr int AC = (BM * AV + 255) / 256, BC = (BN * BV + 255) / 256;
    constexpr int SC = (BN * NS + 255) / 256, DC = (BM * NS + 255) / 256;
    __shared__ __align__(16) int8_t sX[2][BM][KS + 16];
    __shared__ __align__(16) int8_t sW[2][BN][KS + 8];
    __shared__ float sS[2][BN][NS];
    __shared__ float sD[2][BM][NS];
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
    uint4 ra[AC]; uint4 rb[BC]; float rs[SC]; float rd[DC];
    {   // prologue: k0 = 0
        const int b128 = 0, off = 0;
#pragma unroll
        for (int i = 0; i < AC; i++) { const int idx = t + i * 256, r = idx / AV, v = idx % AV;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (idx < BM * AV && m0 + r < end) val = *reinterpret_cast<const uint4*>(&x[(size_t)b128 * rows + (m0 + r)].q[off + v * 16]);
            ra[i] = val; }
#pragma unroll
        for (int i = 0; i < BC; i++) { const int idx = t + i * 256, nn = idx / BV, v = idx % BV;
            uint4 val = make_uint4(0, 0, 0, 0);
            if (idx < BN * BV && n0 + nn < N) val = *reinterpret_cast<const uint4*>(&codes[((size_t)e * N + n0 + nn) * K + 0 + v * 16]);
            rb[i] = val; }
#pragma unroll
        for (int i = 0; i < SC; i++) { const int idx = t + i * 256, nn = idx / NS, ks2 = idx % NS;
            rs[i] = (idx < BN * NS && n0 + nn < N) ? wsc[((size_t)e * N + n0 + nn) * KSG + ks2] : 0.0f; }
#pragma unroll
        for (int i = 0; i < DC; i++) { const int idx = t + i * 256, r = idx / NS, s2 = idx % NS;
            rd[i] = (idx < BM * NS && m0 + r < end) ? x[(size_t)0 * rows + (m0 + r)].d[s2] : 0.0f; }
#pragma unroll
        for (int i = 0; i < AC; i++) { const int idx = t + i * 256, r = idx / AV, v = idx % AV; if (idx < BM * AV) *reinterpret_cast<uint4*>(&sX[0][r][v * 16]) = ra[i]; }
#pragma unroll
        for (int i = 0; i < BC; i++) { const int idx = t + i * 256, nn = idx / BV, v = idx % BV; if (idx < BN * BV) *reinterpret_cast<uint4*>(&sW[0][nn][v * 16]) = rb[i]; }
#pragma unroll
        for (int i = 0; i < SC; i++) { const int idx = t + i * 256, nn = idx / NS, ks2 = idx % NS; if (idx < BN * NS) sS[0][nn][ks2] = rs[i]; }
#pragma unroll
        for (int i = 0; i < DC; i++) { const int idx = t + i * 256, r = idx / NS, s2 = idx % NS; if (idx < BM * NS) sD[0][r][s2] = rd[i]; }
    }
    __syncthreads();
    for (int k0 = 0; k0 < K; k0 += KS) {
        const int buf = (k0 / KS) & 1, k1 = k0 + KS;
        if (k1 < K) {   // issue the next k0's global loads early; they retire during the compute below
            const int b128 = k1 / 128, off = k1 % 128;
#pragma unroll
            for (int i = 0; i < AC; i++) { const int idx = t + i * 256, r = idx / AV, v = idx % AV;
                uint4 val = make_uint4(0, 0, 0, 0);
                if (idx < BM * AV && m0 + r < end) val = *reinterpret_cast<const uint4*>(&x[(size_t)b128 * rows + (m0 + r)].q[off + v * 16]);
                ra[i] = val; }
#pragma unroll
            for (int i = 0; i < BC; i++) { const int idx = t + i * 256, nn = idx / BV, v = idx % BV;
                uint4 val = make_uint4(0, 0, 0, 0);
                if (idx < BN * BV && n0 + nn < N) val = *reinterpret_cast<const uint4*>(&codes[((size_t)e * N + n0 + nn) * K + k1 + v * 16]);
                rb[i] = val; }
#pragma unroll
            for (int i = 0; i < SC; i++) { const int idx = t + i * 256, nn = idx / NS, ks2 = idx % NS;
                rs[i] = (idx < BN * NS && n0 + nn < N) ? wsc[((size_t)e * N + n0 + nn) * KSG + k1 / 32 + ks2] : 0.0f; }
#pragma unroll
            for (int i = 0; i < DC; i++) { const int idx = t + i * 256, r = idx / NS, s2 = idx % NS;
                rd[i] = (idx < BM * NS && m0 + r < end) ? x[(size_t)b128 * rows + (m0 + r)].d[off / 32 + s2] : 0.0f; }
        }
#pragma unroll
        for (int ks = 0; ks < NS; ks++) {
            uint64_t xa[MT];
            float sax[MT][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                const int rr = (wave * MT + mt) * 16 + lr;
                xa[mt] = *reinterpret_cast<const uint64_t*>(&sX[buf][rr][ks * 32 + lg * 8]);
                const int b = (wave * MT + mt) * 16;
#pragma unroll
                for (int j = 0; j < 4; j++) sax[mt][j] = sD[buf][b + lg + 4 * j][ks];
            }
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                const uint64_t wa = *reinterpret_cast<const uint64_t*>(&sW[buf][nt * 16 + lr][ks * 32 + lg * 8]);
                const float sw = sS[buf][nt * 16 + lr][ks];
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    I4 r = {0, 0, 0, 0};
                    r = __builtin_amdgcn_mmac_i32_16x16x32i8((long)wa, (long)xa[mt], r);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        acc[mt][nt][j] = fmaf((float)r[j], sax[mt][j] * sw, acc[mt][nt][j]);
                }
            }
        }
        if (k1 < K) {
            const int nb = buf ^ 1;
#pragma unroll
            for (int i = 0; i < AC; i++) { const int idx = t + i * 256, r = idx / AV, v = idx % AV; if (idx < BM * AV) *reinterpret_cast<uint4*>(&sX[nb][r][v * 16]) = ra[i]; }
#pragma unroll
            for (int i = 0; i < BC; i++) { const int idx = t + i * 256, nn = idx / BV, v = idx % BV; if (idx < BN * BV) *reinterpret_cast<uint4*>(&sW[nb][nn][v * 16]) = rb[i]; }
#pragma unroll
            for (int i = 0; i < SC; i++) { const int idx = t + i * 256, nn = idx / NS, ks2 = idx % NS; if (idx < BN * NS) sS[nb][nn][ks2] = rs[i]; }
#pragma unroll
            for (int i = 0; i < DC; i++) { const int idx = t + i * 256, r = idx / NS, s2 = idx % NS; if (idx < BM * NS) sD[nb][r][s2] = rd[i]; }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ar = m0 + (wave * MT + mt) * 16 + lg + 4 * j;
                const int wr = n0 + nt * 16 + lr;
                if (ar < end && wr < N) {
                    if constexpr (F32) ((float*)y)[(size_t)ar * N + wr] = acc[mt][nt][j];
                    else ((uint16_t*)y)[(size_t)ar * N + wr] = out_hf(acc[mt][nt][j]);
                }
            }
}

// MT==1 -> the double-buffered v11 (971.4 ms); MT==2 -> the single-buffered v8 (1,019.8 ms).
template <int MT, int NT, int KS, int ABL = 0, bool F32 = false>
static void launch(const int8_t* codes, const float* wsc, const Q8D4* x, const int* bounds, void* y,
                   int N, int K, int total_rows, int max_rows, int n, hipStream_t s) {
    constexpr int BM = 16 * MT * 4, BN = 16 * NT;
    dim3 g((unsigned)((max_rows + BM - 1) / BM), (unsigned)((N + BN - 1) / BN), (unsigned)n);
    if (MT == 1) kdn_v11_kernel<MT, NT, KS, ABL, F32><<<g, 256, 0, s>>>(codes, wsc, x, bounds, y, N, K, total_rows);
    else         kdn_v8_kernel <MT, NT, KS, ABL, F32><<<g, 256, 0, s>>>(codes, wsc, x, bounds, y, N, K, total_rows);
}

}  // namespace kdn_down
