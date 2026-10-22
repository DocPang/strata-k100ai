// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#include "strata/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "strata/prefill/kernels.hpp"   // f16-20261014 ("f16 Dm"): kernels::sums_to_f16 (the fallback's conversion)

#include <cmath>   // STRATA_PF_FUSED_SQ: fabsf/fmaxf/roundf for the fused swiglu+quantize kernel below
#include <cstdio>
#include <cstdlib>
#include "gateup_exact.hpp"
#include "kgu_wsg16.hpp"   // kgu-20261014: the IQ2_S (WSG=16) gate/up kernel

namespace strata::prefill::mmq {
// downstage-20261013: definition of the optional IQ4 down boundary marker declared in moe_mmq.hpp.
void (*iq4_down_split_mark)(void* stream) = nullptr;
namespace {

#include "iq4_preexpand_exact.hpp"
struct ExactState {
    ggml_backend_cuda_context* backend=nullptr;
    void* codes=nullptr;void* scales=nullptr;
    bool enabled=false;uint64_t hits=0,fallbacks=0;
    bool q2_enabled=false;uint64_t q2_hits=0,q2_fallbacks=0;   // downstage-20261013: the Q2_0 down path
    void* owner_stream=nullptr;bool stream_set=false;
    // f16-20261014 ("f16 Dm"): the FP32 scratch the mul_mat_q_case fallback writes before it is converted into the
    // FP16 Dm (see Context::run).  total_rows * w_rows floats, grown as needed, never allocated inside a capture.
    float* dst_scr=nullptr;size_t dst_scr_cap=0;
};

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

__global__ void copy16_kernel(const uint4* __restrict__ a, int64_t na, const uint4* __restrict__ b, int64_t nb,
                              uint4* __restrict__ ab_dst, const uint4* __restrict__ c, int64_t nc, uint4* __restrict__ c_dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}
// copy16_kernel for an MMQ group: blockIdx.y is the expert (first + y)
struct GroupArgs {
    const uint8_t* blob[kGatherGroupMax];
    int64_t up_off, down_off, gu_stride, d_stride;   // in uint4
};
__global__ void copy16_group_kernel(GroupArgs ga, int first, int64_t na, int64_t nc, uint4* __restrict__ gu_dst,
                                    uint4* __restrict__ d_dst) {
    const int q = first + (int) blockIdx.y;
    const uint4* src = (const uint4*) ga.blob[q];
    uint4* ab = gu_dst + (int64_t) q * ga.gu_stride;
    uint4* cd = d_dst + (int64_t) q * ga.d_stride;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < na) ab[i] = src[i];
    else if (i < 2 * na) ab[i] = src[ga.up_off + (i - na)];
    else if (i < 2 * na + nc) cd[i - 2 * na] = src[ga.down_off + (i - 2 * na)];
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

// STRATA_PF_FUSED_SQ (the fused swiglu + q8_1 quantizer): Hq = quantize(swiglu(GU)) in ONE launch.  It is the
// composition of swiglu_kernel above and quantize_mmq_q8_1<ds_layout, /*scatter=*/false> (llama.cpp's
// quantize.cu, the kernel quantize() runs): the same 4-floats-a-thread column mapping (the same grid), the same
// f32 expression for h, the same 32-value block abs-max in the same XOR-shuffle order, the same
// `d_inv = 127.0f / amax`, the same `roundf(x * d_inv)` and the same `d = 1.0f / d_inv` (the round trip through
// d_inv, never amax / 127).  The only difference is where the four f32 inputs come from -- registers here, an
// f32 h buffer there, and a f32 store followed by a load is lossless -- so the bytes it writes to `vy` are
// exactly the bytes quantize() writes.  Both kernels live in this one translation unit, so the same expression
// is compiled by the same compiler with the same options and yields the same float.
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

}  // namespace

bool built() { return true; }

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0:
#ifdef STRATA_ORCA_Q4KS_MMQ
        case GGML_TYPE_Q5_0:   // #296 (Q4_K and Q5_1: STRATA_MMQ_KQUANTS)
#endif
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q8_0:   // the draft layer's dense matrices (E-9)
#ifdef STRATA_MMQ_KQUANTS
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q5_1:   // Unsloth's UD-Q4_K_XL experts (CUDA)
#if defined(__HIPCC__) || defined(STRATA_Q6K_EXPERTS)
        case GGML_TYPE_Q6_K:   // HIP: the dense GGUF projections (STRATA_DENSE_MMQ); CUDA: only the opt-in -DSTRATA_Q6K_EXPERTS=ON build
#endif
#endif
            return true;
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
                                     "(#420)\n", ggml_type_name((ggml_type) t), (long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

// gatherq2-20261014: one thread a (k_block, destination row) pair.  quantize_mmq_q8_1 stores the block of (k_block,
// row) at k_block*ne1 + row (ne1 = the number of rows), so the two sides have different row counts and a destination
// "row" is strided; the 36-byte quant block is copied whole.
// coalesce-20261021: one thread per DWORD of this k_block's flattened (row, dword) space, so the
// stores are perfectly coalesced.  The previous form gave each thread a whole 36-byte block, which
// made consecutive threads 36 bytes apart: nine 4-byte stores at a 36-byte stride, ~10 cache lines
// per 128 useful bytes, and 470x off DRAM (STATUS §277).  Pure copy => bit-identical.
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

Context::Context() {
    int dev=0;cudaGetDevice(&dev);auto* st=new ExactState;
    st->backend=new ggml_backend_cuda_context(dev);
    // best-20261016: ON by default (=0 opts out).
    const char* env=std::getenv("STRATA_MMQ_IQ4_PREEXPAND");st->enabled=(env==nullptr)||std::atoi(env)!=0;
    if(st->enabled)std::fprintf(stderr,"[IQ4_PREEXPAND_EXACT_V2] ENABLED workspace=lazy max_experts=16 M_avg_min=128\n");
    // downstage-20261013: the Q2_0 down product through the same expand+int8-MMAC route (default on;
    // STRATA_MMQ_Q2_0_PREEXPAND=0 keeps the production dp4a mul_mat_q_case<GGML_TYPE_Q2_0>).
    const char* envq=std::getenv("STRATA_MMQ_Q2_0_PREEXPAND");st->q2_enabled=(envq==nullptr||std::atoi(envq)!=0);
    if(st->q2_enabled)std::fprintf(stderr,"[Q2_0_PREEXPAND_EXACT] ENABLED max_experts=16\n");
    ctx_=st;
}
Context::~Context() {
    auto* st=(ExactState*)ctx_;
    if(st->enabled)std::fprintf(stderr,"[IQ4_PREEXPAND_EXACT_V2] SUMMARY hits=%llu fallbacks=%llu\n",(unsigned long long)st->hits,(unsigned long long)st->fallbacks);
    if(st->q2_enabled)std::fprintf(stderr,"[Q2_0_PREEXPAND_EXACT] SUMMARY hits=%llu fallbacks=%llu\n",(unsigned long long)st->q2_hits,(unsigned long long)st->q2_fallbacks);
    if(st->codes)cudaFree(st->codes);if(st->scales)cudaFree(st->scales);
    if(st->dst_scr)cudaFree(st->dst_scr);   // f16-20261014: the f16 Dm fallback's FP32 scratch
    delete st->backend;delete st;
}

// STRATA_MMQ_EXACT_ALL=1 (default 0): the exact int8 pre-expanded down product accepts every group, not only
// groups whose experts average >= 128 rows each (M_avg_min=128, the same heuristic the ENABLED banner prints).
// That threshold is a PERFORMANCE judgement - the reason 5,432 of 8,272 down calls in the 30,602-token run took
// llama.cpp's slow fallback - not a correctness bound: the exact kernels are bounds/ids driven and the
// expansion is bit-exact against ggml's to_float for the same bytes.  Opt-in, default off.
inline bool mmq_exact_all() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MMQ_EXACT_ALL"); return v != nullptr && v[0] == '1'; }();
    return on;
}
void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk;
    auto& st = *(ExactState*)ctx_;
    auto& ctx = *st.backend;
    const cudaStream_t s = (cudaStream_t) stream;
    const bool shape_ok=t==GGML_TYPE_IQ4_NL&&p.w_rows==2560&&p.w_cols==640&&p.ld_dst==2560&&p.expert_bytes==921600&&p.n<=16&&(mmq_exact_all()?p.total_rows>0:(p.total_rows>=128*p.n&&p.max_rows>=128))&&p.bounds&&p.ids;
    // downstage-20261013: the same down shape in the Q2_0 pack (d_type 42), whose GGUF block is 18 bytes
    // per 64 values, so matrix_bytes(42,2560,640) = 460800.
    // f16-20261014 ("f16 Dm"): a down product now carries dst16 instead of dst, so accept either.
    const bool shape_ok_q2=t==GGML_TYPE_Q2_0&&p.w_rows==2560&&p.w_cols==640&&p.ld_dst==2560&&p.expert_bytes==matrix_bytes(t,2560,640)&&p.n<=16&&(mmq_exact_all()?p.total_rows>0:(p.total_rows>=128*p.n&&p.max_rows>=128))&&p.bounds&&p.ids&&p.w!=nullptr&&p.xq!=nullptr&&(p.dst!=nullptr||p.dst16!=nullptr);
    // No scratch for small-M fallback. Never allocate inside graph capture; a capture before first warm call uses the original path.
    if(((st.enabled&&shape_ok)||(st.q2_enabled&&shape_ok_q2))&&!st.codes){
        hipStreamCaptureStatus capture=hipStreamCaptureStatusNone;
        ck(hipStreamIsCapturing((hipStream_t)s,&capture),"IQ4 exact capture status");
        if(capture==hipStreamCaptureStatusNone){
            ck(cudaMalloc(&st.codes,16ULL*2560*640),"IQ4 exact code scratch");
            ck(cudaMalloc(&st.scales,16ULL*2560*(640/32)*sizeof(float)),"IQ4 exact scale scratch");
            std::fprintf(stderr,"[IQ4_PREEXPAND_EXACT_V2] ALLOCATED scratch_bytes=29491200 outside_capture=1\n");
        }
    }
    if(st.enabled&&shape_ok&&st.codes&&(!st.stream_set||st.owner_stream==stream)){
        if(st.hits==0)std::fprintf(stderr,"[IQ4_PREEXPAND_EXACT_V2] ACTIVE type=%d n=%d total_rows=%lld max_rows=%lld N=%lld K=%lld stride=%zu\n",p.type,p.n,(long long)p.total_rows,(long long)p.max_rows,(long long)p.w_rows,(long long)p.w_cols,p.expert_bytes);
        st.stream_set=true;st.owner_stream=stream;++st.hits;
        run_expanded_iq4(p,(int8_t*)st.codes,(float*)st.scales,s,true);
        ck(cudaGetLastError(),"IQ4 preexpand exact");return;
    }
    if(st.q2_enabled&&shape_ok_q2&&st.codes&&(!st.stream_set||st.owner_stream==stream)){
        if(st.q2_hits==0)std::fprintf(stderr,"[Q2_0_PREEXPAND_EXACT] ACTIVE type=%d n=%d total_rows=%lld max_rows=%lld N=%lld K=%lld stride=%zu\n",p.type,p.n,(long long)p.total_rows,(long long)p.max_rows,(long long)p.w_rows,(long long)p.w_cols,p.expert_bytes);
        st.stream_set=true;st.owner_stream=stream;++st.q2_hits;
        run_expanded_q2_0(p,(int8_t*)st.codes,(float*)st.scales,s,true);
        ck(cudaGetLastError(),"Q2_0 preexpand exact");return;
    }
    // f16-20261014 ("f16 Dm"): a down product (dst16 set) whose exact path cannot run - the small-M shapes, a stream
    // that is not the one the scratch was allocated on, or a capture before the first warm call - falls back to
    // llama.cpp's mul_mat_q_case, which stores FP32.  It stores into this scratch instead of Dm and the conversion
    // below writes Dm.  The row map is the identity (ids[rr] == rr, the down call sites' invariant), so the move is
    // elementwise.  The scratch is grown here, never inside a capture (an unallocated capture fails closed).
    // F1 (apply_scratch_lazy): reached only by a call that really falls through to mul_mat_q_case - the
    // exact int8 branches above return.  `a` moves with the block on purpose (see apply_scratch_lazy.py).
    if (p.dst16 != nullptr) {
        const size_t need = (size_t) p.total_rows * (size_t) p.w_rows;
        if (need > st.dst_scr_cap) {
            hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
            ck(hipStreamIsCapturing((hipStream_t) s, &capture), "f16 Dm fallback capture status");
            if (capture != hipStreamCaptureStatusNone) {
                std::fprintf(stderr, "prefill mmq: the f16 Dm fallback needs a %lld-float scratch and the stream is "
                                     "capturing (warm the prompt path first)\n", (long long) need);
                std::exit(1);
            }
            if (st.dst_scr) { cudaFree(st.dst_scr); st.dst_scr = nullptr; st.dst_scr_cap = 0; }
            if (cudaMalloc((void**) &st.dst_scr, need * sizeof(float)) != cudaSuccess) {
                cudaGetLastError();
                std::fprintf(stderr, "prefill mmq: cannot allocate the f16 Dm fallback scratch (%lld floats)\n",
                             (long long) need);
                std::exit(1);
            }
            st.dst_scr_cap = need;
        }
    }
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds,
                        p.dst16 != nullptr ? st.dst_scr : p.dst, nullptr,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    if(st.enabled&&t==GGML_TYPE_IQ4_NL)++st.fallbacks;
    if(st.q2_enabled&&t==GGML_TYPE_Q2_0)++st.q2_fallbacks;
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
#ifdef STRATA_MMQ_KQUANTS
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, a, s); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, a, s); break;
#if defined(__HIPCC__) || defined(STRATA_Q6K_EXPERTS)
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, a, s); break;
#endif
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, a, s); break;
#endif
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    // f16-20261014 ("f16 Dm"): the fallback's FP32 scratch -> the product's FP16 Dm (identity row map: the scratch's
    // row rr is Dm's row rr, exactly the mapping the fallback stored with ids_dst[rr] == rr).
    if (p.dst16 != nullptr)
        sums_to_f16(st.dst_scr, p.dst16, (int64_t) ((size_t) p.total_rows * (size_t) p.w_rows), s);
    ck(cudaGetLastError(), "mul_mat_q");
}

#include <vector>
#include <cmath>
namespace { struct GuScratch { int8_t* codes=nullptr; float* scales=nullptr; }; GuScratch g_gu_scr; }

// Exact gate/up product for every expert format this model uses. The expansion is bit-exact against
// ggml's to_float for the same bytes; the GEMM reads the production Q8 D4 activations. Only the
// weight representation changes, so numerical semantics are preserved.
// wv2x-20261016 (DRAFT): the single switch for the IQ2 (2.x bpw) extension of the whitelist below.  Default OFF.
// Unset/0 leaves the shipped four-format whitelist in place, so anything that reaches this function for type 16/17
// still returns false at the `default` and the behaviour is byte for byte today's.  prefill.cpp reads the same
// variable for the routing half of the change (fused_nat), which is what actually gets the 19 layers here.
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

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream) {
    const cudaStream_t s = (cudaStream_t) stream;
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      gu_half_bytes | d_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
        copy16_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint4*) gate, na, (const uint4*) up, na, (uint4*) gu_dst,
                                                          (const uint4*) down, nc, (uint4*) d_dst);
    } else {
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        copy1_kernel<<<blocks(2 * na + nc), 256, 0, s>>>((const uint8_t*) gate, na, (const uint8_t*) up, na,
                                                         (uint8_t*) gu_dst, (const uint8_t*) down, nc, (uint8_t*) d_dst);
    }
    ck(cudaGetLastError(), "gather_native");
}

bool gather_native_group(const GatherGroup& g, size_t up_off, size_t gu_half_bytes, size_t down_off, size_t d_bytes,
                         void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, void* stream) {
    if (g.first < 0 || g.n <= g.first || g.n > kGatherGroupMax) return false;
    uintptr_t a = (uintptr_t) gu_dst | (uintptr_t) d_dst | up_off | gu_half_bytes | down_off | d_bytes | gu_stride | d_stride;
    for (int q = g.first; q < g.n; ++q) a |= (uintptr_t) g.blob[q];
    if (a % 16 != 0) return false;
    GroupArgs ga{};
    for (int q = g.first; q < g.n; ++q) ga.blob[q] = g.blob[q];
    ga.up_off = (int64_t) up_off / 16;
    ga.down_off = (int64_t) down_off / 16;
    ga.gu_stride = (int64_t) gu_stride / 16;
    ga.d_stride = (int64_t) d_stride / 16;
    const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
    copy16_group_kernel<<<dim3(blocks(2 * na + nc), (unsigned) (g.n - g.first)), 256, 0, (cudaStream_t) stream>>>(
        ga, g.first, na, nc, (uint4*) gu_dst, (uint4*) d_dst);
    ck(cudaGetLastError(), "gather_native_group");
    return true;
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    strata_q2_kernel<<<blocks(1280LL * 40 + 2560LL * 10), 256, 0, (cudaStream_t) stream>>>(blob, (uint16_t*) gu_dst,
                                                                                         (uint16_t*) d_dst);
    ck(cudaGetLastError(), "gather_strata_q2");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

// STRATA_PF_FUSED_SQ: swiglu + its q8_1 quantization in one launch.  The launch frame is quantize()'s (:195)
// frame for the same arguments, so the grid, the block size and therefore the thread -> (k_block, iqs, row)
// mapping are identical to the two-call path's quantizer.
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

}  // namespace strata::prefill::mmq
