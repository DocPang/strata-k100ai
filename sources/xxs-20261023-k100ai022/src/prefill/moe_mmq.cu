// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "strata/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "gateup_exact.hpp"   // k100ai: the exact IQ3/IQ2 gate-up int8-MMAC path
#include "kgu_wsg16.hpp"      // k100ai: its WSG=16 variant (IQ2_S / IQ2_XS)
#if defined(STRATA_MMQ_TAB) && !defined(__HIPCC__)
#include "moe_mmq_tab.hpp"   // declarations only: the tab kernels live in the per-type mmq-tab-*.cu TUs
#endif

#include "strata/prefill/mmq_kq_abi.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "strata/prefill/kernels.hpp"   // f16-20261014 ("f16 Dm"): kernels::sums_to_f16 (the fallback's conversion)

namespace strata::prefill::mmq {
// k100ai: the pre-expanded down route calls this before its GEMM (unused here).
void (*iq4_down_split_mark)(void* stream) = nullptr;
#include "iq4_preexpand_exact.hpp"   // k100ai: the pre-expanded int8 down route

namespace {
// The Gyro trellis types (ids 144-146) are outside the type table of an upstream ggml (indexed by the id): their block
// geometry is known here (128 weights; 34 / 26 / 30 bytes), everything else asks ggml.
inline bool is_tq(int t) { return t >= 144 && t <= 146; }
inline int64_t type_blck(int t) { return is_tq(t) ? 128 : (int64_t) ggml_blck_size((ggml_type) t); }
inline size_t type_bytes(int t) { return t == 144 ? 34 : t == 145 ? 26 : t == 146 ? 30 : ggml_type_size((ggml_type) t); }
inline const char* type_label(int t) {
    return t == 144 ? "tq2_t" : t == 145 ? "tqk6" : t == 146 ? "tqk7" : ggml_type_name((ggml_type) t);
}
}  // namespace
#if defined(STRATA_KQ_DSO)
namespace kq {
int abi_args_size() { return (int) sizeof(mmq_args); }
}  // namespace kq
#endif
namespace {
#if defined(STRATA_MMQ_KQUANTS)
// MODEL-UPDATE Phase 1: the K-quant products through MMQ are opt-in at run time (STRATA_MMQ_KQ=1); on CUDA their
// kernels are the strata_kq library (mmq_kq_loader.cpp), opened only then
bool kq_ok() {
#if defined(STRATA_KQ_DSO)
    return kq::available();
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MMQ_KQ");
        return v != nullptr && v[0] == '1';
    }();
    return on;
#endif
}
#endif

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

// ... plus, in the same launch, the expert's 16-byte NVFP4 tail (tail -> tail_dst) and the zeroed MMQ tails after
// the group's last slot (nz uint4 after ab_dst's and c_dst's copies): one launch per expert instead of a kernel,
// a 16-byte copy and, per group, two memsets.
__global__ void copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                              uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc, uint4* __restrict__ c_dst,
                              const uint4* __restrict__ tail, uint4* __restrict__ tail_dst, int64_t nz) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = na + nb + nc;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < n) c_dst[i - na - nb] = c[i - na - nb];
    else if (i < n + nz) ab_dst[na + nb + (i - n)] = make_uint4(0, 0, 0, 0);
    else if (i < n + 2 * nz) c_dst[nc + (i - n - nz)] = make_uint4(0, 0, 0, 0);
    else if (i == n + 2 * nz && tail != nullptr) *tail_dst = *tail;
}
// copy16_kernel for an MMQ group: blockIdx.y is the expert (first + y); the last one (n - 1) also zeroes nz uint4
// after its gu and d slots, and each copies its NVFP4 tail (tail_off, in uint4; tail_dst null = none)
struct GroupArgs {
    const uint8_t* blob[kGatherGroupMax];
    int64_t up_off, down_off, gu_stride, d_stride, tail_off;   // in uint4
};
__global__ void copy16_group_kernel(GroupArgs ga, int first, int n, int64_t na, int64_t nc, int64_t nz,
                                    uint4* __restrict__ gu_dst, uint4* __restrict__ d_dst, uint4* __restrict__ tail_dst) {
    const int q = first + (int) blockIdx.y;
    const uint4* src = (const uint4*) ga.blob[q];
    uint4* ab = gu_dst + (int64_t) q * ga.gu_stride;
    uint4* cd = d_dst + (int64_t) q * ga.d_stride;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t m = 2 * na + nc;
    if (i < na) ab[i] = src[i];
    else if (i < 2 * na) ab[i] = src[ga.up_off + (i - na)];
    else if (i < m) cd[i - 2 * na] = src[ga.down_off + (i - 2 * na)];
    else if (i < m + nz) { if (q == n - 1) ab[ga.gu_stride + (i - m)] = make_uint4(0, 0, 0, 0); }
    else if (i < m + 2 * nz) { if (q == n - 1) cd[ga.d_stride + (i - m - nz)] = make_uint4(0, 0, 0, 0); }
    else if (i == m + 2 * nz && tail_dst != nullptr) tail_dst[q] = src[ga.tail_off];
}
__global__ void copy1_kernel(const uint8_t* __restrict__ a, int64_t na, const uint8_t* __restrict__ b, int64_t nb,
                             uint8_t* __restrict__ ab_dst, const uint8_t* __restrict__ c, int64_t nc, uint8_t* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales
// [2560][10] f16 (the layout of prefill/kernels.cu's blob_dequant_kernel).  A GGUF Q2_0 block is {f16 d; 16 code
// bytes} with the same 2-bit codes in the same order, so a block is a scale and a 16-byte run of codes.
__global__ void strata_q2_kernel(const uint8_t* __restrict__ blob, uint16_t* __restrict__ gu, uint16_t* __restrict__ dn) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;   // one GGUF block
    const int64_t n_gu = 1280LL * 40, n_d = 2560LL * 10;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / 40, b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t*) (blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 10, b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t*) (blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const uint4 q = *(const uint4*) codes;
    const uint16_t* qh = (const uint16_t*) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

__global__ void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                              bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

// swiglu_kernel on the MMQ gate/up output with each expert's gate and up scales applied as it is read: the products
// scale_gu_rows stored, without the pass that stored them
__global__ void swiglu_scaled_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                                     const int32_t* __restrict__ bounds, int n, const float* __restrict__ tails,
                                     int64_t row0) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff, ra = row0 + r;
    int q = 0;
    while (q + 1 < n && ra >= bounds[q + 1]) ++q;
    const float* row = gu + r * 2 * n_ff;
    const float g = row[k] * tails[4 * q], u = row[n_ff + k] * tails[4 * q + 1];
    h[i] = g / (1.0f + __expf(-g)) * u;
}
__global__ void down_row_scales_kernel(float* __restrict__ sd, const int32_t* __restrict__ bounds, int n,
                                       const float* __restrict__ tails, int64_t nrows) {
    const int64_t r = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    sd[r] = tails[4 * q + 2];
}

// k100ai/STRATA_PF_FUSED_SQ: swiglu + its q8_1 quantisation in ONE launch (rc.7's kernel, verbatim).
template <mmq_q8_1_ds_layout ds_layout>
__global__ void swiglu_quantize_kernel(const float* __restrict__ gu, void* __restrict__ vy, int64_t ne00,
                                       int64_t ne0, int64_t ne1, int64_t n_ff, bool interleaved) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    // the quantizer's own thread mapping: one thread = four consecutive columns of one row
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;

    if (i0 >= ne0) {
        return;
    }

    // quantize() is called with ids == nullptr, so its source row is blockIdx.x and its row stride is ld == ne00;
    // swiglu_kernel's row r starts at gu + r * 2 * n_ff.
    const float* row = gu + (int64_t) blockIdx.x * 2 * n_ff;

    float4 xi;
    if (i0 >= ne00) {
        // the row's 512-value padding (ne0 = pad512(ne00)): quantize() reads zeros there
        xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else if (interleaved) {
        // the Strata pack: gate at 2k, up at 2k+1 (swiglu_kernel's `interleaved` branch)
        const float g0 = row[2 * (i0 + 0)], u0 = row[2 * (i0 + 0) + 1];
        const float g1 = row[2 * (i0 + 1)], u1 = row[2 * (i0 + 1) + 1];
        const float g2 = row[2 * (i0 + 2)], u2 = row[2 * (i0 + 2) + 1];
        const float g3 = row[2 * (i0 + 3)], u3 = row[2 * (i0 + 3) + 1];
        xi.x = g0 / (1.0f + __expf(-g0)) * u0;
        xi.y = g1 / (1.0f + __expf(-g1)) * u1;
        xi.z = g2 / (1.0f + __expf(-g2)) * u2;
        xi.w = g3 / (1.0f + __expf(-g3)) * u3;
    } else {
        // the GGUF pack: gate at k, up at n_ff + k (swiglu_kernel's default branch)
        const float g0 = row[i0 + 0], u0 = row[n_ff + i0 + 0];
        const float g1 = row[i0 + 1], u1 = row[n_ff + i0 + 1];
        const float g2 = row[i0 + 2], u2 = row[n_ff + i0 + 2];
        const float g3 = row[i0 + 3], u3 = row[n_ff + i0 + 3];
        xi.x = g0 / (1.0f + __expf(-g0)) * u0;
        xi.y = g1 / (1.0f + __expf(-g1)) * u1;
        xi.z = g2 / (1.0f + __expf(-g2)) * u2;
        xi.w = g3 / (1.0f + __expf(-g3)) * u3;
    }

    // ---- from here on: quantize_mmq_q8_1<ds_layout, /*scatter=*/false>, expression for expression ----

    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;

    block_q8_1_mmq* y = (block_q8_1_mmq*) vy;

    const int64_t k_block = i0 / QK8_1_MMQ;   // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ;   // quant index in block

    // scatter == false, so one block per (k_block, row); ne2 == ne3 == 1 => blockIdx.z == 0
    const int64_t ib0 = blockIdx.z * ((int64_t) gridDim.x * gridDim.y * blockDim.x / QK8_1);
    const int64_t ib = ib0 + k_block * ne1 + blockIdx.x;

    char4* yqs4 = (char4*) y[ib].qs;
    yqs4[iqs / 4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) {
                y[ib].d2s6[iqs / 64] = d;
            }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}


__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

#if defined(__HIPCC__)   // #820: only the HIP dense-MMQ path (Gemm::native_mmq) uses these two
__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(__ushort_as_half(x[i]));
}

__global__ void set_bounds_kernel(int32_t* d, int32_t rows) {
    d[0] = 0;
    d[1] = rows;
}

#endif

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

#if defined(STRATA_MMQ_TAB) && !defined(__HIPCC__)
// STRATA_MMQ_DIRECT: a group's guards (y < g.n, byte stores) and its two zero tails (y == g.n: gate/up, y == g.n + 1:
// down; uint4 stores when `tail16`, which the host checked: both pointers and the size 16-byte aligned).  Only in a
// STRATA_MMQ_TAB build: the engine loads every kernel at start (EAGER), so a build without it keeps today's image.
__global__ void prep_group_kernel(GuardTab g, int64_t guard, uint8_t* tail_gu, uint8_t* tail_d, int64_t tail, bool tail16) {
    const int y = (int) blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (y < g.n) {
        if (i < guard) g.p[y][i] = 0;
        return;
    }
    uint8_t* t = y == g.n ? tail_gu : tail_d;
    if (t == nullptr) return;
    if (tail16) {
        if (i < tail / 16) ((uint4*) t)[i] = make_uint4(0, 0, 0, 0);
    } else if (i < tail) {
        t[i] = 0;
    }
}

// the pointer-table MMQ per type: the native packs' nine types (CMake's _strata_mmq_tab_types, all of them checked
// bit for bit by prefill_mmq_tab_test).  false: no instance - the optional Q5_0 and K-quant experts keep the gather
bool run_tab(ggml_backend_cuda_context* ctx, const mmq_args* a, const mmq_x_tab* tab, cudaStream_t s) {
    const ggml_type t = a ? a->type_x : GGML_TYPE_COUNT;
#define STRATA_TAB_CASE(T) case T: return ctx == nullptr || mul_mat_q_tab_case<T>(*ctx, *a, *tab, s)
    switch (t) {
        STRATA_TAB_CASE(GGML_TYPE_Q2_0);
        STRATA_TAB_CASE(GGML_TYPE_IQ2_XXS);
        STRATA_TAB_CASE(GGML_TYPE_IQ2_XS);
        STRATA_TAB_CASE(GGML_TYPE_IQ2_S);
        STRATA_TAB_CASE(GGML_TYPE_IQ3_XXS);
        STRATA_TAB_CASE(GGML_TYPE_IQ3_S);
        STRATA_TAB_CASE(GGML_TYPE_IQ4_NL);
        STRATA_TAB_CASE(GGML_TYPE_IQ4_XS);
        STRATA_TAB_CASE(GGML_TYPE_Q8_0);
        default: return false;
    }
#undef STRATA_TAB_CASE
}
// whether run_tab has an instance for type t (no launch)
bool tab_has_type(int t) {
    mmq_args a{};
    a.type_x = (ggml_type) t;
    return run_tab(nullptr, &a, nullptr, nullptr);
}
#else
bool run_tab(ggml_backend_cuda_context*, const mmq_args*, const void*, cudaStream_t) {
    std::fprintf(stderr, "prefill mmq: this build has no pointer-table MMQ (STRATA_MMQ_TAB)\n");
    std::exit(1);
}
bool tab_has_type(int) { return false; }
#endif

}  // namespace

// mmq_nvfp4_w4a8.cu: mmq.cuh's int8 NVFP4 path, compiled for this GPU with Blackwell's FP4 MMA hidden
void run_nvfp4_w4a8(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s);
// mmq_nvfp4_w4a4.cu (the one unit built for 12xa): FP4 x FP4 and its activation quantizer, if this card runs it
bool w4a4_available();
void run_nvfp4_w4a4(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s);
void quantize_nvfp4_w4a4(const float* x, const int32_t* ids, void* xq, float* yscale, bool aligned, int64_t cols,
                         int64_t ld, int64_t rows, int64_t padded, cudaStream_t s);

bool built() { return true; }

Nvfp4Mode nvfp4_mode() {
    static const Nvfp4Mode m = [] {
        const char* e = std::getenv("STRATA_PREFILL_NVFP4");
        if (e == nullptr || std::strcmp(e, "w4a8") == 0) return Nvfp4Mode::W4A8;
        if (std::strcmp(e, "w4a4") == 0) return Nvfp4Mode::W4A4;
        if (std::strcmp(e, "fp16") == 0) return Nvfp4Mode::FP16;
        std::fprintf(stderr, "STRATA_PREFILL_NVFP4=%s: expected w4a8, w4a4 or fp16\n", e);
        std::exit(1);
    }();
    return m;
}

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0:
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0:   // #296 (Q4_K and Q5_1: STRATA_MMQ_KQUANTS)
#endif
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:   // the draft layer's dense matrices (E-9)
#ifdef STRATA_MMQ_TQ
        case GGML_TYPE_TQ2_T: case GGML_TYPE_TQK6: case GGML_TYPE_TQK7:   // APR experts (agention fork's MMQ)
#endif
            return true;
#ifdef STRATA_MMQ_KQUANTS
        // Unsloth's UD-Q4/Q5/Q6_K_XL experts; on HIP also the dense GGUF projections (STRATA_DENSE_MMQ)
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q5_1: case GGML_TYPE_Q6_K:
            return kq_ok();
#endif
#if !defined(GGML_USE_HIP)
        case GGML_TYPE_NVFP4:
            return nvfp4_mode() != Nvfp4Mode::FP16;
#endif
        default:
            return false;
    }
}

bool fits(int t, int64_t w_rows) {
    if (!supported(t)) return false;
    // mul_mat_q_case's choice: the "fallback" configs when the rows are not a multiple of 128; then
    // mul_mat_q_switch_J's loop - a tile size whose config exists for this card and fits its shared memory
    const bool fallback = w_rows % 128 != 0;
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, fallback, cc);
            any = c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo;
        }
        if (!any) {
            static bool said[GGML_TYPE_COUNT] = {};
            if (t >= 0 && t < GGML_TYPE_COUNT && !said[t]) {
                said[t] = true;
                std::fprintf(stderr, "strata: prompt kernels: llama.cpp's MMQ has no tile for %s (%lld rows) on GPU %d "
                                     "(cc %d, %zu bytes of shared memory per block): that product takes the non-MMQ path "
                                     "(#420)\n", type_label((int) t), (long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / type_blck((int) t)) * type_bytes((int) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

namespace {
__global__ void scale_gu_rows_kernel(float* __restrict__ gu, int64_t ld, int64_t n_ff, const int32_t* __restrict__ bounds,
                                     int n, const float* __restrict__ tails, int64_t row0) {
    const int64_t r = row0 + blockIdx.x;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    const float sg = tails[4 * q], su = tails[4 * q + 1];
    float* row = gu + r * ld;
    for (int64_t k = threadIdx.x; k < 2 * n_ff; k += blockDim.x) row[k] *= k < n_ff ? sg : su;
}
__global__ void scale_down_rows_kernel(float* __restrict__ d, int64_t ld, const int32_t* __restrict__ bounds, int n,
                                       const float* __restrict__ tails) {
    const int64_t r = blockIdx.x;
    int q = 0;
    while (q + 1 < n && r >= bounds[q + 1]) ++q;
    const float sd = tails[4 * q + 2];
    float* row = d + r * ld;
    for (int64_t k = threadIdx.x; k < ld; k += blockDim.x) row[k] *= sd;
}
}  // namespace

bool fp4_activations(int t) {
#if defined(GGML_USE_HIP)   // no NVFP4 MMQ on HIP (and no W4A4 unit)
    (void) t;
    return false;
#else
    if (t != GGML_TYPE_NVFP4 || nvfp4_mode() != Nvfp4Mode::W4A4) return false;
    static const bool on = [] {
        const bool ok = w4a4_available();
        if (!ok) std::fprintf(stderr, "prefill mmq: STRATA_PREFILL_NVFP4=w4a4 needs an sm_12x card in a 12x build; W4A8\n");
        return ok;
    }();
    return on;
#endif
}

void scale_gu_rows(float* gu, int64_t ld, int64_t n_ff, const int32_t* bounds, int n, const float* tails,
                   int64_t row0, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    scale_gu_rows_kernel<<<(unsigned) nrows, 256, 0, (cudaStream_t) stream>>>(gu, ld, n_ff, bounds, n, tails, row0);
    ck(cudaGetLastError(), "scale_gu_rows");
}

void scale_down_rows(float* d, int64_t ld, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    scale_down_rows_kernel<<<(unsigned) nrows, 256, 0, (cudaStream_t) stream>>>(d, ld, bounds, n, tails);
    ck(cudaGetLastError(), "scale_down_rows");
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream,
              float* yscale) {
    if (rows <= 0) return;
#if !defined(GGML_USE_HIP)
    if (fp4_activations(t)) {
        if (yscale == nullptr) { std::fprintf(stderr, "prefill mmq: NVFP4 activations need a scale buffer\n"); std::exit(1); }
        const bool aligned = ((uintptr_t) x % 32 == 0) && ((size_t) ld * sizeof(float)) % 32 == 0;
        quantize_nvfp4_w4a4(x, ids, xq, yscale, aligned, cols, ld, rows, pad512(cols), (cudaStream_t) stream);
        ck(cudaGetLastError(), "quantize");
        return;
    }
#endif
    (void) yscale;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

void quantize_scatter(const float* x, const int32_t* slot, const int32_t* src, void* xq, int t, int64_t cols, int64_t ld,
                      int64_t tokens, int k_used, void* stream) {
    if (tokens <= 0 || k_used <= 0) return;
    static const bool gather = [] { const char* e = std::getenv("STRATA_QUANT_GATHER"); return e != nullptr && std::atoi(e) != 0; }();
    if (gather) {
        quantize(x, src, xq, t, cols, ld, tokens * k_used, stream);
        return;
    }
    quantize_scatter_mmq_q8_1_cuda(x, slot, xq, (ggml_type) t, cols, ld, pad512(cols), tokens, tokens * k_used, k_used,
                                   (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize_scatter");
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

// f16-20261014 ("f16 Dm", rc.7): the generic dp4a mul_mat_q_case writes FP32.  A down product now carries dst16
// instead of dst, so the fallback needs an FP32 staging buffer of total_rows*w_rows floats, converted right after
// (grow on demand; fail closed inside a capture, where no allocation is allowed).
float* g_dst_scr = nullptr;
size_t g_dst_scr_cap = 0;
bool dst_scr_ready(size_t n, hipStream_t s) {
    if (n <= g_dst_scr_cap) return true;
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(s, &cap) != hipSuccess || cap != hipStreamCaptureStatusNone) return false;
    if (g_dst_scr) { cudaFree(g_dst_scr); g_dst_scr = nullptr; g_dst_scr_cap = 0; }
    if (hipMalloc((void**) &g_dst_scr, n * sizeof(float)) != hipSuccess) { g_dst_scr = nullptr; return false; }
    g_dst_scr_cap = n;
    return true;
}
void Context::run(const Product& p, void* stream) {
    if (run_iq4_exact(p, stream)) return;    // k100ai: pre-expanded int8 IQ4_NL down (the other half of the MoE)
    if (run_q2_0_exact(p, stream)) return;   // k100ai: pre-expanded int8 Q2_0 down
    {   // k100ai diagnostic (one shot per distinct shape): a down-shaped product that took neither exact route
        if (p.w_rows == 2560 && p.w_cols == 640) {
            static bool said[8] = {};
            const int slot = (p.type >= 0 && p.type < 8) ? p.type : 7;
            if (!said[slot]) {
                said[slot] = true;
                std::fprintf(stderr, "[k100ai downfall] type=%d w_rows=%lld w_cols=%lld expert_bytes=%zu n=%d "
                                     "total_rows=%lld max_rows=%lld ld_dst=%lld dst=%d dst16=%d\n",
                             p.type, (long long) p.w_rows, (long long) p.w_cols, p.expert_bytes, p.n,
                             (long long) p.total_rows, (long long) p.max_rows, (long long) p.ld_dst,
                             (int) (p.dst != nullptr), (int) (p.dst16 != nullptr));
            }
        }
    }
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    // f16 Dm: a down product that could not take the exact route still has to be written somewhere FP32.
    const bool via16 = p.dst16 != nullptr && p.dst == nullptr;
    if (via16 && !dst_scr_ready((size_t) p.total_rows * (size_t) p.w_rows, (hipStream_t) stream)) {
        std::fprintf(stderr, "prefill mmq: no FP32 staging for an FP16 down product (rows=%lld)\n",
                     (long long) p.total_rows);
        std::exit(1);
    }
    const int64_t qk = type_blck((int) t), bpr = p.w_cols / qk;
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, via16 ? g_dst_scr : p.dst, p.y_scale,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, p.tab ? 0 : (int64_t) (p.expert_bytes / type_bytes((int) t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    if (p.tab) {
        if (p.n > kGatherGroupMax) {
            std::fprintf(stderr, "prefill mmq: a pointer table holds at most %d experts (got %d)\n", kGatherGroupMax, p.n);
            std::exit(1);
        }
#if defined(STRATA_MMQ_TAB) && !defined(__HIPCC__)
        mmq_x_tab tab{};
        for (int z = 0; z < p.n; ++z) tab.p[z] = (const char*) p.tab->p[z];
        if (!run_tab(&ctx, &a, &tab, s)) {
            std::fprintf(stderr, "prefill mmq: no pointer-table MMQ for %s with %lld rows\n", type_label((int) t),
                         (long long) p.w_rows);
            std::exit(1);
        }
#else
        run_tab(&ctx, &a, nullptr, s);
#endif
        ck(cudaGetLastError(), "mul_mat_q (pointer table)");
        return;
    }
    switch (t) {
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, a, s); break;
#endif
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s); break;
#ifdef STRATA_MMQ_TQ
        case GGML_TYPE_TQ2_T: mul_mat_q_case<GGML_TYPE_TQ2_T>(ctx, a, s); break;
        case GGML_TYPE_TQK6: mul_mat_q_case<GGML_TYPE_TQK6>(ctx, a, s); break;
        case GGML_TYPE_TQK7: mul_mat_q_case<GGML_TYPE_TQK7>(ctx, a, s); break;
#endif
#if defined(STRATA_MMQ_KQUANTS) && defined(STRATA_KQ_DSO)
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q5_1: {
            const int e = kq::run((int) t, &a, s);
            if (e != 0) {
                std::fprintf(stderr, "prefill mmq: the K-quant library failed on type %d (error %d)\n", (int) t, e);
                std::exit(1);
            }
            break;
        }
#elif defined(STRATA_MMQ_KQUANTS)
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s); break;
#endif
#if !defined(GGML_USE_HIP)   // NVFP4: CUDA only (no HIP instance, no W4A8 unit)
        case GGML_TYPE_NVFP4:
            if (fp4_activations(t)) run_nvfp4_w4a4(ctx, a, s);
            else run_nvfp4_w4a8(ctx, a, s);
            break;
#endif
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
    if (via16) strata::prefill::sums_to_f16(g_dst_scr, p.dst16, (int64_t) p.total_rows * p.w_rows, (void*) s);
}

// ---- STRATA_MMQ_DIRECT ----
bool tab_built() {
#if defined(STRATA_MMQ_TAB) && !defined(__HIPCC__)
    return true;
#else
    return false;
#endif
}

bool tab_fits(int t, int64_t w_rows) {
    return tab_built() && w_rows % 128 == 0 && tab_has_type(t) && fits(t, w_rows);
}

size_t overread_bytes(int t, int64_t cols) {
    if (!supported(t)) return 0;
    // the K tile's over-read for each (fallback=false) config on any visible GPU, the largest of them: the over-read
    // is not monotonic in the tile length (20 blocks: 12 more with a 16-block tile, 4 with a 24-block one), so each
    // config's own; MMQ_ITER_K's is the floor (it is the tile when no config is known)
    const int64_t qk = type_blck((int) t), bpn = cols / qk;
    auto over = [&](int64_t k_vram) -> int64_t {
        const int64_t bpi = k_vram / qk;
        return bpi > 0 ? (bpi - bpn % bpi) % bpi : 0;
    };
    int64_t blocks = over(MMQ_ITER_K);
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id)
        for (int J = 8; J <= 128; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, false, info.devices[id].cc);
            if (c.type != GGML_TYPE_COUNT) blocks = std::max<int64_t>(blocks, over(c.K_vram));
        }
    return (size_t) blocks * type_bytes((int) t);
}

bool whole_tiles(int t, int64_t w_rows, int n_experts) {
    if (!supported(t) || w_rows % 128 != 0 || n_experts <= 0) return false;
    // launch_mul_mat_q's grid: without stream-k one block per tile; with it one block per tile when the tiles fill
    // their waves to >= 90 % (NVIDIA), else nsm blocks that split tiles along K at boundaries set by the tile count.
    // The tile count is nty * n_experts * ntx, ntx = any positive count of J-wide column tiles (max_rows decides),
    // and ntiles >= 9 * nsm always fills its waves to >= 90 %: the ntx below that are checked one by one.
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc, nsm = info.devices[id].nsm;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false, stream_k = false;
        ggml_cuda_mmq_config c0 = ggml_cuda_mmq_get_config((ggml_type) t, 8, false, cc);
        for (int J = 8; J <= 128; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, false, cc);
            if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
            if (!any) c0 = c;
            any = true;
            // every tile width accumulates each output over the same K tiles in the same order
            if (c.I != c0.I || c.K_vram != c0.K_vram || c.sram_layout != c0.sram_layout) return false;
            stream_k = stream_k || c.stream_k;
        }
        if (!any) return false;
        if (!stream_k) continue;
        if (!GGML_CUDA_CC_IS_NVIDIA(cc) || nsm <= 0) return false;
        const int64_t per_ntx = (w_rows / c0.I) * n_experts;
        for (int64_t ntx = 1; per_ntx * ntx < 9 * (int64_t) nsm + per_ntx; ++ntx) {
            const int64_t tiles = per_ntx * ntx, waves = (tiles + nsm - 1) / nsm;
            if (100 * tiles / (nsm * waves) < 90) return false;
        }
    }
    return true;
}

int direct_mode() {
    static const int mode = [] {
        const char* v = std::getenv("STRATA_MMQ_DIRECT");
        const int m = v ? std::atoi(v) : 0;
        if (m != 1 && m != 2) return 0;
        if (!tab_built()) {
            std::fprintf(stderr, "strata: STRATA_MMQ_DIRECT=%d is ignored: this build has no pointer-table MMQ "
                                 "(CUDA with STRATA_MMQ_TAB only)\n", m);
            return 0;
        }
        return m;
    }();
    return mode;
}

size_t direct_slot_guard() { return direct_mode() != 0 ? kSlotGuard : 0; }

void prep_group(const GuardTab& g, size_t guard_bytes, void* tail_gu, void* tail_d, size_t tail_bytes, void* stream) {
#if !defined(STRATA_MMQ_TAB) || defined(__HIPCC__)
    // never reached: only a direct group calls it, and direct_mode() is 0 without the pointer-table MMQ
    (void) g; (void) guard_bytes; (void) tail_gu; (void) tail_d; (void) tail_bytes; (void) stream;
    std::fprintf(stderr, "prefill mmq: this build has no pointer-table MMQ (STRATA_MMQ_TAB)\n");
    std::exit(1);
#else
    const int n = g.n > 0 ? std::min(g.n, kGatherGroupMax) : 0;
    const size_t guard = n > 0 ? guard_bytes : 0;
    const size_t tail = tail_gu || tail_d ? tail_bytes : 0;
    if (guard == 0 && tail == 0) return;
    GuardTab gt = g;
    gt.n = guard > 0 ? n : 0;
    const bool tail16 = ((uintptr_t) tail_gu | (uintptr_t) tail_d | tail_bytes) % 16 == 0;
    const size_t span = std::max(guard, tail);
    prep_group_kernel<<<dim3(blocks((int64_t) span), (unsigned) (gt.n + 2)), 256, 0, (cudaStream_t) stream>>>(
        gt, (int64_t) guard, (uint8_t*) tail_gu, (uint8_t*) tail_d, (int64_t) tail, tail16);
    ck(cudaGetLastError(), "prep_group");
#endif
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream, const void* tail, void* tail_dst, size_t zero_bytes) {
    const cudaStream_t s = (cudaStream_t) stream;
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      (uintptr_t) tail | (uintptr_t) tail_dst | gu_half_bytes | d_bytes | zero_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16, nz = (int64_t) zero_bytes / 16;
        copy16_kernel<<<blocks(2 * na + nc + 2 * nz + 1), 256, 0, s>>>(
            (const uint4*) gate, na, (const uint4*) up, na, (uint4*) gu_dst, (const uint4*) down, nc, (uint4*) d_dst,
            (const uint4*) tail, (uint4*) tail_dst, nz);
    } else {
        if (tail) cudaMemcpyAsync(tail_dst, tail, 16, cudaMemcpyDeviceToDevice, s);
        if (zero_bytes) {
            cudaMemsetAsync((uint8_t*) gu_dst + 2 * gu_half_bytes, 0, zero_bytes, s);
            cudaMemsetAsync((uint8_t*) d_dst + d_bytes, 0, zero_bytes, s);
        }
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        copy1_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint8_t*) gate, na, (const uint8_t*) up, na,
                                                         (uint8_t*) gu_dst, (const uint8_t*) down, nc, (uint8_t*) d_dst);
    }
    ck(cudaGetLastError(), "gather_native");
}

bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream, size_t tail_off,
                         void* tail_dst, size_t zero_bytes) {
    if (g.first < 0 || g.n <= g.first || g.n > kGatherGroupMax) return false;
    if (tail_off == 0) tail_dst = nullptr;
    uintptr_t a = (uintptr_t) gu_dst | (uintptr_t) d_dst | up_off | gu_half_bytes | down_off | d_bytes | gu_stride | d_stride |
                  tail_off | (uintptr_t) tail_dst | zero_bytes;
    for (int q = g.first; q < g.n; ++q) a |= (uintptr_t) g.blob[q];
    if (a % 16 != 0) return false;
    GroupArgs ga{};
    for (int q = g.first; q < g.n; ++q) ga.blob[q] = g.blob[q];
    ga.up_off = (int64_t) up_off / 16;
    ga.down_off = (int64_t) down_off / 16;
    ga.gu_stride = (int64_t) gu_stride / 16;
    ga.d_stride = (int64_t) d_stride / 16;
    ga.tail_off = (int64_t) tail_off / 16;
    const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16, nz = (int64_t) zero_bytes / 16;
    const int64_t work = 2 * na + nc + 2 * nz + (tail_dst != nullptr ? 1 : 0);
    copy16_group_kernel<<<dim3(blocks(work), (unsigned) (g.n - g.first)), 256, 0, (cudaStream_t) stream>>>(
        ga, g.first, g.n, na, nc, nz, (uint4*) gu_dst, (uint4*) d_dst, (uint4*) tail_dst);
    ck(cudaGetLastError(), "gather_native_group");
    return true;
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    strata_q2_kernel<<<blocks(1280LL * 40 + 2560LL * 10), 256, 0, (cudaStream_t) stream>>>(blob, (uint16_t*) gu_dst,
                                                                                         (uint16_t*) d_dst);
    ck(cudaGetLastError(), "gather_strata_q2");
}

void swiglu_scaled(const float* gu, float* h, int64_t rows, int64_t n_ff, const int32_t* bounds, int n,
                   const float* tails, int64_t row0, void* stream) {
    if (rows <= 0) return;
    swiglu_scaled_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, bounds, n, tails, row0);
    ck(cudaGetLastError(), "swiglu_scaled");
}
void down_row_scales(float* sd, const int32_t* bounds, int n, const float* tails, int64_t nrows, void* stream) {
    if (nrows <= 0 || n <= 0) return;
    down_row_scales_kernel<<<blocks(nrows), 256, 0, (cudaStream_t) stream>>>(sd, bounds, n, tails, nrows);
    ck(cudaGetLastError(), "down_row_scales");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

void swiglu_quantize(const float* gu, void* hq, int t, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    const int64_t ne00 = n_ff, ne0 = pad512(n_ff), ne1 = rows;   // quantize(): cols, pad512(cols), rows
    if (ne00 % 4 != 0) {
        std::fprintf(stderr, "prefill mmq: swiglu_quantize: cols %lld is not a multiple of 4\n", (long long) ne00);
        std::exit(1);
    }
    const int64_t block_num_y = (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks((unsigned) ne1, (unsigned) block_num_y, 1u);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    const cudaStream_t s = (cudaStream_t) stream;
    switch (mmq_get_q8_1_ds_layout((ggml_type) t)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            swiglu_quantize_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, s>>>(gu, hq, ne00, ne0, ne1, n_ff, interleaved);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            swiglu_quantize_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, s>>>(gu, hq, ne00, ne0, ne1, n_ff, interleaved);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            swiglu_quantize_kernel<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, s>>>(gu, hq, ne00, ne0, ne1, n_ff, interleaved);
            break;
        default:
            std::fprintf(stderr, "prefill mmq: swiglu_quantize: type %d has no q8_1 data layout\n", t);
            std::exit(1);
    }
    ck(cudaGetLastError(), "swiglu_quantize");
}


// k100ai/STRATA_GATHER_QONCE: gather the layer's T unique q8_1 rows into expert order (rc.7's kernel).
__global__ void gather_q8_kernel(const uint8_t* __restrict__ src, const int32_t* __restrict__ ids,
                                 uint8_t* __restrict__ dst, int64_t src_rows, int64_t rows) {
    constexpr int DW = (int) (sizeof(block_q8_1_mmq) / 4);
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * DW) return;
    const int64_t r = i / DW, j = i - r * DW;
    const uint8_t* s = src + ((size_t) blockIdx.y * (size_t) src_rows + (size_t) ids[r]) * sizeof(block_q8_1_mmq);
    uint8_t* d = dst + ((size_t) blockIdx.y * (size_t) rows + (size_t) r) * sizeof(block_q8_1_mmq);
    reinterpret_cast<uint32_t*>(d)[j] = reinterpret_cast<const uint32_t*>(s)[j];
}

void gather_q8(const void* src, const int32_t* ids, void* dst, int64_t cols, int64_t src_rows, int64_t rows,
               void* stream) {
    if (rows <= 0 || src == nullptr || dst == nullptr || ids == nullptr) return;
    const int64_t nb = pad512(cols) / (4 * QK8_1);
    const dim3 g((unsigned) ((rows * (int64_t) (sizeof(block_q8_1_mmq) / 4) + 255) / 256), (unsigned) nb, 1u);
    gather_q8_kernel<<<g, 256, 0, (cudaStream_t) stream>>>((const uint8_t*) src, ids, (uint8_t*) dst, src_rows, rows);
    ck(cudaGetLastError(), "gather_q8");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

#if defined(__HIPCC__)
void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    f16_to_f32_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    ck(cudaGetLastError(), "f16_to_f32");
}

void set_bounds(int32_t* dst, int32_t rows, void* stream) {
    set_bounds_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(dst, rows);
    ck(cudaGetLastError(), "set_bounds");
}
#endif


// k100ai: the expanded int8 gate/up weights (16 experts x 1280 x 2560 codes + scales).
namespace { struct GuScratch { int8_t* codes=nullptr; float* scales=nullptr; }; GuScratch g_gu_scr; }

static bool gateup_exact_iq2_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_GATEUP_EXACT_IQ2"); return v != nullptr && v[0] == (char) 49; }();
    return on;
}

bool run_gateup_exact(const Product& p, void* stream) {
    // best-20261016: ON by default (=0 opts out).
    static const bool on = [] { const char* v = std::getenv("STRATA_MMQ_GATEUP_EXACT"); return v == nullptr || std::atoi(v) != 0; }();
    if (!on) return false;
    const ggml_type t = (ggml_type) p.type;
    int wsg = 0;
    switch (t) {
        case GGML_TYPE_IQ3_XXS: wsg = 32; break;
        case GGML_TYPE_IQ2_S:   wsg = 16; break;
        case GGML_TYPE_IQ3_S:   wsg = 32; break;
        case GGML_TYPE_IQ4_XS:  wsg = 32; break;
        // wv2x-20261016 (DRAFT): the 2.x bpw pair (19 of this model's 48 gate/up layers).  wsg is not a tuning
        // choice here: IQ2_XXS carries one scale per 32 k-values and IQ2_XS two, one per 16, which is the same
        // split the log already shows as `type=18/21 wsg=32` versus `type=22 wsg=16`.  Guarded so that the
        // expansion list and prefill.cpp's routing predicate always arm together.
        case GGML_TYPE_IQ2_XXS: if (!gateup_exact_iq2_on()) return false; wsg = 32; break;
        case GGML_TYPE_IQ2_XS:  if (!gateup_exact_iq2_on()) return false; wsg = 16; break;
        default: return false;
    }
    if (wsg == 16) {
        // IQ2_S needs one scale per 16 k-values; the 32-deep int8 MMA cannot carry two scales inside
        // one k-window, so the exact path there costs a doubled MMA (gemm_moe_wsg16_lean). It is
        // opt-in: enable with STRATA_MMQ_GATEUP_EXACT_IQ2S=1 and compare against the production
        // dp4a path in the engine before making it the default.
        // best-20261016: ON by default (=0 opts out).
        static const bool iq2s = [] { const char* v = std::getenv("STRATA_MMQ_GATEUP_EXACT_IQ2S"); return v == nullptr || std::atoi(v) != 0; }();
        if (!iq2s) return false;
    }
    if (p.w_rows != 1280 || p.w_cols != 2560 || p.ld_dst != 1280) return false;
    if (p.expert_bytes != matrix_bytes(t, 1280, 2560)) return false;
    // wv2g-20261015: the row threshold is a heuristic guard, not a correctness requirement (the
    // kernels predicate on p.bounds, and STRATA_MMQ_GATEUP_EXACT=0 was measured bit-identical in
    // output, sec.147.4). Make it tunable and count WHY calls are rejected, so the 640-of-1280
    // groups that fall back to the production dp4a MMQ can be priced instead of assumed.
    static const long minrows = [] { const char* v = std::getenv("STRATA_MMQ_GATEUP_MINROWS"); return v ? std::atol(v) : 128L; }();
    static long rej_rows = 0, rej_n = 0;
    if (p.n <= 0 || p.n > 16) { ++rej_n; return false; }
    if (p.bounds == nullptr || p.ids == nullptr) return false;
    if (p.total_rows < minrows || p.max_rows < minrows) {
        if (++rej_rows % 256 == 0)
            std::fprintf(stderr, "[GATEUP_REJECT] minrows=%ld rej_rows=%ld rej_n=%ld tr=%lld mr=%lld\n",
                         minrows, rej_rows, rej_n, (long long) p.total_rows, (long long) p.max_rows);
        return false;
    }
    // Every pointer the kernels below dereference must be validated here: the wcopy incident was
    // exactly an unvalidated pointer reaching an expander.
    if (p.w == nullptr || p.xq == nullptr || p.dst == nullptr) return false;
    // Opt-in shape statistics (STRATA_MMQ_GATEUP_STATS=1): integer accumulation only, so the
    // distribution of (n, total_rows, max_rows) per call is visible without perturbing timing.
    static const bool stats = [] { const char* v = std::getenv("STRATA_MMQ_GATEUP_STATS"); return v != nullptr && std::atoi(v) == 1; }();
    static long st_calls = 0, st_rows = 0, st_mr = 0, st_nsum = 0;
    static long st_per_type[64] = {};
    static int st_nhist[17] = {};
    if (stats) {
        ++st_calls; st_rows += (long) p.total_rows; st_mr += (long) p.max_rows; st_nsum += p.n;
        ++st_per_type[(int) t];
        if (p.n >= 0 && p.n <= 16) ++st_nhist[p.n];
        if (st_calls % 128 == 0) {
            std::fprintf(stderr, "[GATEUP_STATS] calls=%ld nsum=%ld tr=%ld mr=%ld hist=",
                         st_calls, st_nsum, st_rows, st_mr);
            for (int i = 0; i <= 16; ++i) std::fprintf(stderr, "%d%s", st_nhist[i], i == 16 ? "" : ",");
            std::fprintf(stderr, " pertype=%ld/%ld/%ld/%ld\n",
                         st_per_type[18], st_per_type[21], st_per_type[22], st_per_type[23]);
        }
    }
    // Stream ownership probe (diagnostic, does not change behaviour). The IQ4 exact path already
    // refuses to share its scratch across streams; we first measure whether that ever happens.
    {
        static void* seen_stream = nullptr;
        static long switches = 0;
        if (seen_stream == nullptr) seen_stream = stream;
        else if (seen_stream != stream && ++switches <= 8)
            std::fprintf(stderr, "[GATEUP_EXACT_V2] STREAM_SWITCH #%ld first=%p now=%p\n",
                         switches, seen_stream, stream);
    }
    if (g_gu_scr.codes == nullptr || g_gu_scr.scales == nullptr) {
        hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing((hipStream_t) stream, &cap) != hipSuccess || cap != hipStreamCaptureStatusNone) return false;
        // Both buffers are armed together or not at all. If the second allocation failed while the
        // first succeeded, the old guard (codes != nullptr) skipped re-allocation on every later
        // call and handed the expander a null scales pointer -- the same delayed-crash shape as
        // the wcopy incident. Roll the first one back so the next call retries from scratch.
        if (g_gu_scr.codes == nullptr && hipMalloc(&g_gu_scr.codes, 16ULL * 1280 * 2560) != hipSuccess) return false;
        if (g_gu_scr.scales == nullptr && hipMalloc(&g_gu_scr.scales, 16ULL * 1280 * (2560 / 16) * sizeof(float)) != hipSuccess) {
            if (g_gu_scr.codes != nullptr) { hipFree(g_gu_scr.codes); g_gu_scr.codes = nullptr; }
            return false;
        }
        std::fprintf(stderr, "[GATEUP_EXACT_V2] scratch_bytes=%zu\n",
                     (size_t) (16ULL*1280*2560 + 16ULL*1280*(2560/16)*sizeof(float)));
    }
    // one line per expert format, so coverage (how many of the 48 gate/up layers take this path) is
    // visible in the log instead of inferred
    static bool said[64] = {};
    if (!said[(int) t]) { said[(int) t] = true;
        std::fprintf(stderr, "[GATEUP_EXACT_V2] ENABLED type=%d wsg=%d\n", (int) t, wsg);
        // One-shot layout probe (only outside graph capture): dump the first bytes of the expert
        // buffer this call was handed, plus the call parameters, so the engine's real layout can be
        // compared against the pack byte for byte.
        hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing((hipStream_t) stream, &cap) == hipSuccess && cap == hipStreamCaptureStatusNone) {
            unsigned char hb[96];
            if (hipMemcpy(hb, p.w, sizeof(hb), hipMemcpyDeviceToHost) == hipSuccess) {
                std::fprintf(stderr, "[GATEUP_LAYOUT] type=%d rows=%lld cols=%lld eb=%zu ld=%lld n=%d tr=%lld mr=%lld first96=",
                             (int) t, (long long) p.w_rows, (long long) p.w_cols, (size_t) p.expert_bytes,
                             (long long) p.ld_dst, (int) p.n, (long long) p.total_rows, (long long) p.max_rows);
                for (unsigned i = 0; i < sizeof(hb); ++i) std::fprintf(stderr, "%02x", hb[i]);
                std::fprintf(stderr, "\n");
            }
        }
    }
    static const bool verify = [] { const char* v = std::getenv("STRATA_MMQ_GATEUP_VERIFY"); return v != nullptr && std::atoi(v) == 1; }();
    // The engine fills p.w from a different stream, so any host synchronisation between two runs
    // gives the pipeline time to overwrite the buffer underneath us. Freeze a private copy now,
    // before anything else touches it, and compare against that copy instead.
    static void* wcopy = nullptr;
    if (verify) {
        const size_t wb = (size_t) p.expert_bytes * (size_t) p.n;
        const size_t cap = (size_t) 16 * 1280 * 2560;
        if (wcopy == nullptr && hipMalloc(&wcopy, cap) != hipSuccess) wcopy = nullptr;
        if (wcopy != nullptr && wb > 0 && wb <= cap)
            hipMemcpyAsync(wcopy, p.w, wb, hipMemcpyDeviceToDevice, (hipStream_t) stream);
    }
    // The expander must read the weights this call was handed. wcopy is only the frozen
    // diagnostic copy (allocated when STRATA_MMQ_GATEUP_VERIFY=1); routing the expander through
    // it unconditionally made every non-verify run read a null pointer (VMFault at 0xa000).
    const void* wsrc = (wcopy != nullptr) ? (const void*) wcopy : (const void*) p.w;
    // Opt-in stage timing (STRATA_MMQ_GATEUP_TIMING=1). RING of T_N slots, read one full ring later:
    // the earlier 2-slot ping-pong read the previous call only, so hipEventElapsedTime returned
    // hipErrorNotReady whenever the host was a call ahead and the sample was silently dropped --
    // the printed totals were an undercount. Every slot is read from ONE generation, so the expand
    // and gemm sums come from the same set of calls. ok/miss say how many slots were readable.
    static const bool timing = [] { const char* v = std::getenv("STRATA_MMQ_GATEUP_TIMING"); return v != nullptr && std::atoi(v) == 1; }();
    static const int T_N = 256;
    static hipEvent_t tev[T_N][3] = {};
    static int tslot = 0;
    static long tcalls = 0, tok = 0, tmiss = 0;
    static double t_expand = 0, t_gemm = 0;
    if (timing) {
        if (tcalls == 0)
            for (int i = 0; i < T_N; ++i) for (int j = 0; j < 3; ++j) (void) hipEventCreate(&tev[i][j]);
        tslot = (int) (tcalls % T_N);
        if (tcalls >= T_N) {
            float e = 0.f, g = 0.f;
            const bool a = hipEventElapsedTime(&e, tev[tslot][0], tev[tslot][1]) == hipSuccess;
            const bool b = hipEventElapsedTime(&g, tev[tslot][1], tev[tslot][2]) == hipSuccess;
            if (a && b) { t_expand += e; t_gemm += g; ++tok; } else ++tmiss;
        }
        (void) hipEventRecord(tev[tslot][0], (hipStream_t) stream);
    }
    const int nb = 1280 * (2560 / 256);
    // the expanders are warp-cooperative: one warp covers two 256-value blocks, 8 blocks per
    // 256-thread CTA
    const unsigned gs = (unsigned) ((p.n * nb + 7) / 8);
    const hipStream_t s = (hipStream_t) stream;
    switch (t) {
        case GGML_TYPE_IQ3_XXS:
            gateup_exact::expand_iq3xxs_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
        case GGML_TYPE_IQ2_S:
            gateup_exact::expand_iq2s_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
        case GGML_TYPE_IQ3_S:
            gateup_exact::expand_iq3s_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
        case GGML_TYPE_IQ2_XXS:
            gateup_exact::expand_iq2xxs_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
        case GGML_TYPE_IQ2_XS:
            gateup_exact::expand_iq2xs_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
        default:
            gateup_exact::expand_iq4xs_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
    }
    if (timing) (void) hipEventRecord(tev[tslot][1], s);
    const dim3 g((unsigned) ((p.max_rows + 63) / 64), (unsigned) ((1280 + 127) / 128), (unsigned) p.n);
    const dim3 g16((unsigned) ((p.max_rows + 63) / 64), (unsigned) ((1280 + 63) / 64), (unsigned) p.n);
    // wsg16fix-20261014: for IQ2_S (wsg=16) the exact 16-deep split costs two MMAC per 32-k window,
    // so the kernel is supply-bound; NT=8 (BN=128) halves the per-column activation re-staging and
    // wins once the group's longest expert is large, while NT=4 (BN=64, occ 4) wins for small groups.
    // Crossover measured on HCU4 at max_rows ~5000-6000 (wsg16fix-20261014); both tiles are exact.
    // kgu-20261014: route the IQ2_S (WSG=16) gate/up product to the kgu_wsg16 kernel family
    // (kgu_wsg16.hpp).  The NT rule (BN=128 for max_rows >= 5500, else BN=64), the LINEAR store and
    // everything else are unchanged; STRATA_GATEUP_KERNEL=0 restores the shipped kernels exactly.
    static const int kgu_on = [](){const char* v=std::getenv("STRATA_GATEUP_KERNEL");return v?std::atoi(v):1;}();
    // wv2-20261020 (default OFF): STRATA_GATEUP_WSG16_NT8=1 forces the BN=128 (NT=8) tile even below the shipped
    // max_rows>=5500 crossover, so the small-group optimum can be re-measured on HCU4. Unset/0 = the parent path.
    static const int kgu_nt8 = [](){const char* v=std::getenv("STRATA_GATEUP_WSG16_NT8");return v?std::atoi(v):0;}();
    // wv2b-20261020 (default OFF): STRATA_GATEUP_MT2=1 doubles BM (64 -> 128) so the tile becomes BM=BN=128.
    // The recorded expert-GEMM traffic rule (1/BM + 1/BN) goes 1/64+1/128 = 0.0234 -> 1/128+1/128 = 0.0156
    // (1.5x less redundant weight traffic), at the recorded cost of 32 -> 64 accumulator registers a thread.
    // Both template families are generic in MT (acc[MT][NT][4], rows = wave*MT+mt), so the arithmetic per row,
    // and therefore the output bits, are unchanged. The `lean`/`staged` grids must follow BM: they are built here.
    // Unset/0: every line below falls through to the parent dispatch, byte for byte.
    static const int kgu_mt2 = [](){const char* v=std::getenv("STRATA_GATEUP_MT2");return v?std::atoi(v):0;}();
    // wv2c-20261020 (default OFF): STRATA_GATEUP_KS128=1 doubles KS (64 -> 128) on the staged tail, i.e. one whole
    // Q8D4 activation block per k0 step. kgu_wsg16.hpp:19 records this as one of the three supply-side changes that
    // took the IQ2_S family from 17.14 to 20.48 TMAC/s ("KS=128 ... halving the barrier count vs the shipped KS=64")
    // -- the wsg32 family (IQ3_XXS/IQ3_S/IQ4_XS, 58% of the gate/up calls) never got it. The per-(mt,nt,j)
    // accumulation order in k is unchanged, so the outputs stay bit-identical. Unset/0 = the parent path.
    static const int kgu_ks128 = [](){const char* v=std::getenv("STRATA_GATEUP_KS128");return v?std::atoi(v):0;}();
    // wv2h-20261015 (default OFF): STRATA_GATEUP_WSG32_NT4=1 uses the lower-VGPR BN=64 tile for the wsg32
    // staged family when the group's longest expert is below the wsg16 crossover. Unset/0 = the parent path.
    static const int wsg32_nt4 = [](){const char* v=std::getenv("STRATA_GATEUP_WSG32_NT4");return v?std::atoi(v):0;}();
    // wv2k-20261015 (default OFF): STRATA_GATEUP_LB3=1 uses the __launch_bounds__(256,3) compile of the shipped tile.
    static const int staged_lb3 = [](){const char* v=std::getenv("STRATA_GATEUP_LB3");return v?std::atoi(v):0;}();
    static const int kgu_sv4 = [](){const char* v=std::getenv("STRATA_GATEUP_SV4");return v?std::atoi(v):0;}();
    static const int kgu_w8 = [](){const char* v=std::getenv("STRATA_GATEUP_W8");return v?std::atoi(v):1;}();   // rc: on by default (bit-identical, gate/up -1.9%)
    if (kgu_on && wsg == 16) {
        const bool nt8 = (kgu_nt8 != 0) || p.max_rows >= 5500;
        if (kgu_mt2) {
            if (nt8)
                kgu_wsg16::launch<2,8,128>(g_gu_scr.codes, g_gu_scr.scales, (const kgu_wsg16::Q8D4*) p.xq, p.bounds, p.dst, 1280, 2560, (int) p.total_rows, (int) p.max_rows, p.n, s);
            else
                kgu_wsg16::launch<2,4,128>(g_gu_scr.codes, g_gu_scr.scales, (const kgu_wsg16::Q8D4*) p.xq, p.bounds, p.dst, 1280, 2560, (int) p.total_rows, (int) p.max_rows, p.n, s);
        } else if (nt8)
            kgu_wsg16::launch<1,8,128>(g_gu_scr.codes, g_gu_scr.scales, (const kgu_wsg16::Q8D4*) p.xq, p.bounds, p.dst, 1280, 2560, (int) p.total_rows, (int) p.max_rows, p.n, s);
        else
            kgu_wsg16::launch<1,4,128>(g_gu_scr.codes, g_gu_scr.scales, (const kgu_wsg16::Q8D4*) p.xq, p.bounds, p.dst, 1280, 2560, (int) p.total_rows, (int) p.max_rows, p.n, s);
    } else if (wsg == 16 && kgu_mt2) {
        const dim3 g16m2n8((unsigned) ((p.max_rows + 127) / 128), (unsigned) ((1280 + 127) / 128), (unsigned) p.n);
        const dim3 g16m2n4((unsigned) ((p.max_rows + 127) / 128), (unsigned) ((1280 + 63) / 64), (unsigned) p.n);
        if (p.max_rows >= 5500)
            gateup_exact::gemm_moe_wsg16_lean<2,8,64,true><<<g16m2n8,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
        else
            gateup_exact::gemm_moe_wsg16_lean<2,4,64,true><<<g16m2n4,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
    } else if (wsg == 16 && p.max_rows >= 5500) {
        const dim3 g16w((unsigned) ((p.max_rows + 63) / 64), (unsigned) ((1280 + 127) / 128), (unsigned) p.n);
        gateup_exact::gemm_moe_wsg16_lean<1,8,64,true><<<g16w,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
    } else if (wsg == 16)
        gateup_exact::gemm_moe_wsg16_lean<1,4,64,true><<<g16,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
    else if (kgu_mt2 || kgu_ks128) {
        // The staged tail (wsg32: IQ3_XXS / IQ3_S / IQ4_XS), with the two supply-side levers. The grid follows BM
        // (64 or 128); KS does not change it. LDS at BM=128/KS=128: sA[128][144] + sB[128][136] + sS[128][4]*4
        // = 9,216 + 17,408 + 2,048 = 28,672 B, well under the 64 KiB limit.
        const dim3 gs((unsigned) ((p.max_rows + (kgu_mt2 ? 127 : 63)) / (kgu_mt2 ? 128 : 64)),
                      (unsigned) ((1280 + 127) / 128), (unsigned) p.n);
        if (kgu_mt2 && kgu_ks128)
            gateup_exact::gemm_moe_staged<2,8,128,32,true><<<gs,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
        else if (kgu_mt2)
            gateup_exact::gemm_moe_staged<2,8,64,32,true><<<gs,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
        else
            gateup_exact::gemm_moe_staged<1,8,128,32,true><<<gs,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
    } else if (wsg32_nt4 && p.max_rows < 5500) {
        // wv2h-20261015 (default OFF; STRATA_GATEUP_WSG32_NT4=1): the wsg32 staged tail ALWAYS runs
        // <1,8,64,32> (BN=128). The ISA dump of this very TU says that kernel is 128 VGPR / 15,360 B LDS
        // => 2 waves/CU = 25% occupancy, while its sibling <1,4,64,32> (BN=64) is 77 VGPR / 10,240 B
        // => 3 waves/CU = 37.5%. The wsg16 family already switches on max_rows at 5,500 for exactly this
        // reason. The per-(mt,nt,j) accumulation order in k is unchanged by NT, so the output bits are
        // identical; only the column-tile grid changes (1280/64 = 20, not 10).
        const dim3 g4((unsigned) ((p.max_rows + 63) / 64), (unsigned) ((1280 + 63) / 64), (unsigned) p.n);
        gateup_exact::gemm_moe_staged<1,4,64,32,true><<<g4,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
    } else if (staged_lb3) {
        // wv2k-20261015 (default OFF; STRATA_GATEUP_LB3=1): the __launch_bounds__(256,3) compile of the same
        // tile. Same arithmetic, same bits; only the register allocation changes.
        gateup_exact::gemm_moe_staged_lb3<1,8,64,32,true><<<g,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
    } else if (kgu_sv4)
        // sv4-20261021: one float4 LDS read for the four WSG=32 weight scales of a (nt,lg) group.
        gateup_exact::gemm_moe_staged<1,8,64,32,true,float,true><<<g,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
    else if (kgu_w8) {   // the sv4 branch above is a single statement, so no brace closes here
        // w8-20261021: 8 waves per block => BM=128 (staging -33%), same per-thread accumulators.
        const dim3 g8((unsigned) ((p.max_rows + 127) / 128), (unsigned) ((1280 + 127) / 128), (unsigned) p.n);
        gateup_exact::gemm_moe_staged<1,8,64,32,true,float,false,8><<<g8,512,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
    }
    else
        gateup_exact::gemm_moe_staged<1,8,64,32,true><<<g,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
    if (timing) {
        (void) hipEventRecord(tev[tslot][2], s);
        ++tcalls;
        if (tcalls % 256 == 0)
            std::fprintf(stderr, "[GATEUP_TIMING] calls=%ld ok=%ld miss=%ld expand=%.1f ms gemm=%.1f ms\n",
                         tcalls, tok, tmiss, t_expand, t_gemm);
    }
    if (hipPeekAtLastError() != hipSuccess) {
        std::fprintf(stderr, "gateup exact: launch failed, falling back to production MMQ\n");
        (void) cudaGetLastError();   // clear so the next call is not blamed for this one
        return false;
    }

    // ---- in-engine parity (opt-in, STRATA_MMQ_GATEUP_VERIFY=1) --------------------------------
    // Run the production MMQ and this exact path over the *same* live call (same weights buffer,
    // same activations, same bounds/ids/output) and compare a few elements directly. Component
    // harnesses agreed to 1e-7, so this is what tells us whether the whole-model difference comes
    // from the gate/up product at all.
    if (verify && wcopy != nullptr) {
        static int nver[64] = {};
        ++nver[(int) t];
        // Measure only well after the pipeline has filled: the first calls of a prefill run while
        // several streams are still being primed, and reading the output there is not meaningful.
        if (nver[(int) t] == 50 || nver[(int) t] == 51) {
            const int vcall = nver[(int) t];
            static Context* vctx = nullptr;
            if (vctx == nullptr) vctx = new Context();
            const int nrow = (p.max_rows < 4) ? (int) p.max_rows : 4;
            const int ncol = 32;
            int hstart = 0;
            if (hipMemcpy(&hstart, p.bounds, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) hstart = 0;
            const int* ids = (const int*) p.ids;
            const float* y = (const float*) p.dst;
            std::vector<int> hid(nrow);
            std::vector<float> ours((size_t) nrow * ncol), prod((size_t) nrow * ncol);
            const bool ok_ids = hipMemcpy(hid.data(), ids + hstart, nrow * sizeof(int), hipMemcpyDeviceToHost) == hipSuccess;
            if (ok_ids) {
                for (int i = 0; i < nrow; ++i)
                    hipMemcpy(&ours[(size_t) i * ncol], y + (size_t) hid[i] * 1280, ncol * sizeof(float), hipMemcpyDeviceToHost);
                Product p2 = p; p2.w = wcopy;
                vctx->run(p2, (void*) stream);
                hipStreamSynchronize((hipStream_t) stream);
                std::vector<float> prod2((size_t) nrow * ncol);
                for (int i = 0; i < nrow; ++i)
                    hipMemcpy(&prod2[(size_t) i * ncol], y + (size_t) hid[i] * 1280, ncol * sizeof(float), hipMemcpyDeviceToHost);
                vctx->run(p2, (void*) stream);   // second production run: is it even deterministic?
                if (hipStreamSynchronize((hipStream_t) stream) == hipSuccess) {
                    for (int i = 0; i < nrow; ++i)
                        hipMemcpy(&prod[(size_t) i * ncol], y + (size_t) hid[i] * 1280, ncol * sizeof(float), hipMemcpyDeviceToHost);
                    double se = 0, sr = 0, ma = 0;
                    for (size_t i = 0; i < prod.size(); ++i) {
                        const double d = (double) ours[i] - (double) prod[i];
                        se += d * d; sr += (double) prod[i] * (double) prod[i];
                        ma = std::fabs(d) > ma ? std::fabs(d) : ma;
                    }
                    std::fprintf(stderr, "[GATEUP_VERIFY] call=%d type=%d wsg=%d rows=%d max_rows=%lld rel_l2=%.3e max_abs=%.3e prod0=%.6f ours0=%.6f\n",
                                 vcall, (int) t, wsg, nrow, (long long) p.max_rows,
                                 sr > 0 ? std::sqrt(se / sr) : std::sqrt(se), ma, prod[0], ours[0]);
                    // raw evidence: is the frozen copy actually the weights the expander read, and
                    // are we reading the output rows the doc says (dst row = ids[row])?
                    unsigned char pa[16], pb[16];
                    hipMemcpy(pa, p.w, 16, hipMemcpyDeviceToHost);
                    hipMemcpy(pb, wcopy, 16, hipMemcpyDeviceToHost);
                    std::fprintf(stderr, "[GATEUP_VERIFY]   pw16=");
                    for (int i = 0; i < 16; ++i) std::fprintf(stderr, "%02x", pa[i]);
                    std::fprintf(stderr, " wcopy16=");
                    for (int i = 0; i < 16; ++i) std::fprintf(stderr, "%02x", pb[i]);
                    std::fprintf(stderr, " same=%d hstart=%d hid=%d,%d,%d,%d\n",
                                 (int) (std::memcmp(pa, pb, 16) == 0), hstart, hid[0], hid[1], hid[2], hid[3]);
                    double se2 = 0, sr2 = 0;
                    for (size_t i = 0; i < prod.size(); ++i) {
                        const double d = (double) prod[i] - (double) prod2[i];
                        se2 += d * d; sr2 += (double) prod2[i] * (double) prod2[i];
                    }
                    std::fprintf(stderr, "[GATEUP_VERIFY]   prod_run1_vs_run2 rel_l2=%.3e\n",
                                 sr2 > 0 ? std::sqrt(se2 / sr2) : std::sqrt(se2));
                    std::fprintf(stderr, "[GATEUP_VERIFY]   ours:");
                    for (int i = 0; i < 6; ++i) std::fprintf(stderr, " %.6f", ours[i]);
                    std::fprintf(stderr, "  prod:");
                    for (int i = 0; i < 6; ++i) std::fprintf(stderr, " %.6f", prod[i]);
                    std::fprintf(stderr, "\n");
                    std::fflush(stderr);
                }
                // restore our result so the model keeps running the exact path
                expand_kernels_only:;
                switch (t) {
                    case GGML_TYPE_IQ3_XXS:
                        gateup_exact::expand_iq3xxs_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
                    case GGML_TYPE_IQ2_S:
                        gateup_exact::expand_iq2s_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
                    case GGML_TYPE_IQ3_S:
                        gateup_exact::expand_iq3s_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
                    case GGML_TYPE_IQ2_XXS:
                        gateup_exact::expand_iq2xxs_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
                    case GGML_TYPE_IQ2_XS:
                        gateup_exact::expand_iq2xs_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
                    default:
                        gateup_exact::expand_iq4xs_kernel<<<gs,256,0,s>>>((const uint8_t*) wsrc, g_gu_scr.codes, g_gu_scr.scales, nb, p.n, 1280, 2560); break;
                }
                if (wsg == 16 && p.max_rows >= 5500) {
                    const dim3 g16w((unsigned) ((p.max_rows + 63) / 64), (unsigned) ((1280 + 127) / 128), (unsigned) p.n);
                    gateup_exact::gemm_moe_wsg16_lean<1,8,64,true><<<g16w,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
                } else if (wsg == 16)
                    gateup_exact::gemm_moe_wsg16_lean<1,4,64,true><<<g16,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
                else
                    gateup_exact::gemm_moe_staged<1,8,64,32,true><<<g,256,0,s>>>(g_gu_scr.codes, g_gu_scr.scales, (const gateup_exact::Q8D4*) p.xq, p.bounds, p.ids, p.dst, 1280, 2560, (int) p.total_rows);
                if (cudaGetLastError() != cudaSuccess) { std::fprintf(stderr, "gateup verify restore failed\n"); std::exit(1); }
            }
        }
    }
    return true;
}



// ---------------------------------------------------------------------------------------------------------------
// k100ai/gfx928: the Q2_0 down product through the pre-expanded int8 MMAC route
// (src/prefill/iq4_preexpand_exact.hpp).  Upstream 0.2.2 runs llama.cpp's dp4a mul_mat_q_case for Q2_0, which on
// K100_AI measured 4.3x slower than this route (ctx32: 4503 vs 1051 ms for the whole down phase).  The kernel
// writes FP16 (kdn_down::launch takes uint16_t*), so the result goes to a scratch and is widened into p.dst -
// one extra pass over the output, ~0.3 ms for a 30K chunk.
namespace {
__global__ void k100ai_f16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(*reinterpret_cast<const __half*>(x + i));
}
}  // namespace

namespace {
struct K100aiIq4Stat {
    long hits = 0, fb = 0;
    ~K100aiIq4Stat() { std::fprintf(stderr, "[k100ai IQ4_0] hits=%ld fallbacks=%ld\n", hits, fb); }
};
K100aiIq4Stat g_k100ai_iq4;
// k100ai: the IQ4_NL down product on the pre-expanded int8 route (rc.7 downstage-20261013).  Without it every
// non-Q2_0 down layer falls to the dp4a mul_mat_q_case, which measured 1,271 ms of the 30,602-token pass.
struct K100aiQ2Stat {
    long hits = 0, fb = 0;
    ~K100aiQ2Stat() { std::fprintf(stderr, "[k100ai Q2_0] hits=%ld fallbacks=%ld\n", hits, fb); }
};
K100aiQ2Stat g_k100ai_q2;
}  // namespace

bool run_iq4_exact(const Product& p, void* stream) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MMQ_IQ4_PREEXPAND");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (!on) return false;
    if ((ggml_type) p.type != GGML_TYPE_IQ4_NL) return false;
    struct Fb { bool taken = false; ~Fb() { if (!taken) ++g_k100ai_iq4.fb; } } fb_guard;
    if (p.n <= 0 || p.n > 16 || p.max_rows <= 0 || p.total_rows <= 0) return false;
    // the same down shape as the Q2_0 route; an IQ4_NL block is 18 bytes per 32 values, so 2560x640 = 921600
    if (!(p.w_rows == 2560 && p.w_cols == 640 && p.ld_dst == 2560)) return false;
    if (p.expert_bytes != 921600) return false;
    if (!p.bounds || !p.ids || !p.w || !p.xq || (!p.dst && !p.dst16)) return false;
    static const bool exact_all = [] {
        const char* v = std::getenv("STRATA_MMQ_EXACT_ALL");
        return v != nullptr && v[0] == '1';
    }();
    if (!exact_all && !(p.total_rows >= 128 * p.n && p.max_rows >= 128)) return false;
    static int8_t* codes = nullptr;
    static float* scales = nullptr;
    static void* owner = nullptr;
    const hipStream_t s = (hipStream_t) stream;
    if (codes == nullptr || scales == nullptr) {
        hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(s, &cap) != hipSuccess || cap != hipStreamCaptureStatusNone) return false;
        if (codes == nullptr && hipMalloc((void**) &codes, 16ULL * 2560 * 640) != hipSuccess) {
            codes = nullptr;
            return false;
        }
        if (scales == nullptr && hipMalloc((void**) &scales, 16ULL * 2560 * (640 / 32) * sizeof(float)) != hipSuccess)
            return false;
    }
    if (owner != nullptr && owner != stream) return false;
    owner = stream;
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "[k100ai IQ4_0] ACTIVE n=%d total_rows=%lld max_rows=%lld stride=%zu\n",
                     p.n, (long long) p.total_rows, (long long) p.max_rows, p.expert_bytes);
    }
    ++g_k100ai_iq4.hits;
    fb_guard.taken = true;
    run_expanded_iq4(p, codes, scales, s, true);   // f16 Dm: the FP16 twin writes p.dst16
    ck(cudaGetLastError(), "k100ai IQ4 down");
    return true;
}
bool run_q2_0_exact(const Product& p, void* stream) {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MMQ_Q2_0_PREEXPAND");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (!on) return false;
    if ((ggml_type) p.type != GGML_TYPE_Q2_0) return false;
    struct Fb { bool taken = false; ~Fb() { if (!taken) ++g_k100ai_q2.fb; } } fb_guard;
    if (p.n <= 0 || p.n > 16 || p.max_rows <= 0 || p.total_rows <= 0) return false;
    if (!(p.w_rows == 2560 && p.w_cols == 640 && p.ld_dst == 2560)) return false;
    if (p.expert_bytes != matrix_bytes((int) GGML_TYPE_Q2_0, 2560, 640)) return false;
    if (!p.bounds || !p.ids || !p.w || !p.xq || (!p.dst && !p.dst16)) return false;
    // STRATA_MMQ_EXACT_ALL=1 takes every group; otherwise only groups averaging >= 128 rows per expert.
    static const bool exact_all = [] {
        const char* v = std::getenv("STRATA_MMQ_EXACT_ALL");
        return v != nullptr && v[0] == '1';
    }();
    if (!exact_all && !(p.total_rows >= 128 * p.n && p.max_rows >= 128)) return false;

    static int8_t* codes = nullptr;
    static float* scales = nullptr;
    static void* owner = nullptr;
    const hipStream_t s = (hipStream_t) stream;
    // k100ai: the FP16 scratch of the first patch-7 port is gone.  run_expanded_q2_0_f32 writes p.dst directly
    // (patch 11), so that buffer was dead - but its hipFree-on-growth kept draining the queue inside this call,
    // and the drain was charged to the `down expand` phase window.
    if (codes == nullptr || scales == nullptr) {
        hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(s, &cap) != hipSuccess || cap != hipStreamCaptureStatusNone) return false;
        if (codes == nullptr && hipMalloc((void**) &codes, 16ULL * 2560 * 640) != hipSuccess) {
            codes = nullptr;
            return false;
        }
        if (scales == nullptr && hipMalloc((void**) &scales, 16ULL * 2560 * (640 / 32) * sizeof(float)) != hipSuccess)
            return false;
    }
    if (owner != nullptr && owner != stream) return false;
    owner = stream;
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "[k100ai Q2_0] ACTIVE n=%d total_rows=%lld max_rows=%lld stride=%zu\n",
                     p.n, (long long) p.total_rows, (long long) p.max_rows, p.expert_bytes);
    }
    ++g_k100ai_q2.hits;
    fb_guard.taken = true;
    run_expanded_q2_0(p, codes, scales, s, true);   // f16 Dm: the FP16 twin writes p.dst16
    ck(cudaGetLastError(), "k100ai Q2_0 down");
    return true;
}

}  // namespace strata::prefill::mmq
