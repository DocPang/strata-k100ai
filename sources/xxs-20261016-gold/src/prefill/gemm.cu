// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
#include "strata/prefill/moe_mmq.hpp"
#endif

#ifdef STRATA_USE_HIP
#include "wmma_gemm.h"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif

namespace strata::prefill {
namespace {
#if defined(__HIPCC__)
// Y's rows, written by an FP16-out GEMM as FP16 at the start of each FP32 row (ldc = 2 ldy halves), widened in place.
// Float c overwrites halves 2c and 2c+1, so a row is walked from its end in blocks: a block's halves are read into
// registers, the block syncs, then writes its floats - which only cover halves of blocks already read.
// STRATA_DBG_NAN: the FP16 outputs that are not finite (an FP32-accumulated sum past 65504 becomes inf in FP16).
__device__ unsigned long long g_f16_nonfinite = 0;
__global__ void widen_rows_f16(float* __restrict__ Y, int64_t n, int64_t ldy, int count_nonfinite) {
    float* y = Y + (int64_t) blockIdx.x * ldy;
    const __half* h = reinterpret_cast<const __half*>(y);
    const int64_t nb = (n + blockDim.x - 1) / blockDim.x;
    for (int64_t b = nb - 1; b >= 0; --b) {
        const int64_t c = b * blockDim.x + threadIdx.x;
        const float v = c < n ? __half2float(h[c]) : 0.0f;
        if (count_nonfinite && c < n && !isfinite(v)) atomicAdd(&g_f16_nonfinite, 1ull);
        __syncthreads();
        if (c < n) y[c] = v;
        __syncthreads();
    }
}
// BF16 weight rows -> FP16, saturated (bench/results/2026-10-04-rdna2-fp16-prompt: none leaves FP16's range)
__global__ void bf16_to_f16_rows(const uint16_t* __restrict__ s, __half* __restrict__ d, int64_t n) {
    for (int64_t i = blockIdx.x * (int64_t) blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float f = __uint_as_float((uint32_t) s[i] << 16);
        d[i] = __float2half(isnan(f) ? f : fminf(fmaxf(f, -65504.0f), 65504.0f));   // as hf_sat: a NaN stays NaN
    }
}
#endif

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// #247/#325: on Windows (seen on gfx1201), hipBLAS can return success with the correct BF16/FP16 product for some
// shapes (hc up once T >= 96, the router) and still leave hipErrorInvalidValue set, which the next kernel's error
// check turns into an exit. The multiply has finished, so that one stale error is cleared after a GEMM that succeeded;
// any other error still stops the engine. Windows only: on Linux a stale hipErrorInvalidValue is a real error from
// an earlier call and keeps being reported. A no-op everywhere else (CUDA compiles none of it).
#if defined(__HIPCC__) && defined(_WIN32)
void absorb_hipblas_sticky(const char* what) {
    const hipError_t sticky = hipGetLastError();
    if (sticky == hipSuccess || sticky == hipErrorInvalidValue) return;
    std::fprintf(stderr, "prefill gemm: %s left %s\n", what, hipGetErrorString(sticky));
    std::exit(1);
}
#define STRATA_ABSORB_HIPBLAS_STICKY(what) absorb_hipblas_sticky(what)
#else
#define STRATA_ABSORB_HIPBLAS_STICKY(what) ((void) 0)
#endif

// A setup call whose failure the engine survives (the handle keeps its defaults), as before #240 - but said.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d (continuing)\n", what, (int) s);
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy, int ldb = 0) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, ldb > k ? ldb : k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (!path || !*path) return nullptr;

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

// the hipBLASLt solution index the tuning table gives this call (-1: none / unsupported)
int hipblaslt_solution_for(void* opaque_state, strata::prefill::hipblaslt::InputType type, int64_t t, int64_t n, int64_t k,
                           int64_t ldy, float beta) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX || ldy < n) return -1;
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported || resolved.workspace_bytes > state->workspace_bytes) return -1;
    auto algo = resolved.algo;
    return hipblaslt_ext::getIndexFromAlgo(algo);
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream, int64_t ldx = 0) {
    static const int64_t min_t = [] { const char* e = std::getenv("STRATA_HIPBLASLT_MIN_T"); return e ? std::max<int64_t>(0, std::atoll(e)) : 0; }();
    if (t < min_t) return false;
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy, (int) ldx)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

}  // namespace

bool prompt_f16() {
#if defined(__HIPCC__)
    constexpr int kMaxDev = 64;
    static std::atomic<int8_t> cached[kMaxDev] = {};   // 0 unknown, 1 off, 2 on
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= kMaxDev) { (void) hipGetLastError(); return false; }
    if (const int8_t c = cached[dev].load(std::memory_order_relaxed)) return c == 2;
    // Opt-in (#835): only STRATA_HIP_PROMPT_F16=1 turns it on.  It changes the prompt path's numbers (an FP16 rounding of
    // each 16-bit GEMM's output), so no card gets it unasked.  On gfx103x it is ~2x faster, so we say so once.
    const char* e = std::getenv("STRATA_HIP_PROMPT_F16");
    const bool on = e != nullptr && e[0] == '1';
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) { (void) hipGetLastError(); p.gcnArchName[0] = 0; }
    if (on) std::fprintf(stderr, "strata prefill: HIP device %d%s%s - STRATA_HIP_PROMPT_F16=1: the prompt's 16-bit GEMMs run FP16 in and out "
                                 "(rocBLAS is tuned only for that on gfx103x; not bit-identical to the default)%s",
                         dev, p.gcnArchName[0] ? " " : "", p.gcnArchName, "\n");
    else if (e == nullptr && std::strncmp(p.gcnArchName, "gfx103", 6) == 0)
        std::fprintf(stderr, "strata prefill: HIP device %d %s - tip: STRATA_HIP_PROMPT_F16=1 reads long prompts about 2x faster here "
                             "(FP16 prompt GEMMs, rocBLAS is tuned only for FP16 on gfx103x; the numbers differ slightly, see docs/AMD_HIP.md)%s",
                     dev, p.gcnArchName, "\n");
    cached[dev].store(on ? 2 : 1, std::memory_order_relaxed);
    return on;
#else
    return false;
#endif
}

Gemm::~Gemm() {
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    delete static_cast<strata::prefill::mmq::Context*>(mmq_ctx_);
    if (mmq_buf_) cudaFree(mmq_buf_);
#endif
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (tc_w_) cudaFree(tc_w_);
    if (tc_x_) cudaFree(tc_x_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // #240: every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

#if !defined(__HIPCC__)
// ---- below sm_80: BF16 GEMMs without BF16 tensor cores (PR #655, #540, #395) ----------------------------------------
// Turing and Volta have no BF16 tensor cores: cublasGemmEx on CUDA_R_16BF inputs falls back to a SIMT fp32 kernel
// (magma_sgemmEx), a fifth of a 4K prompt on an RTX 2080 Ti (#655) and ~10% of a V100's prompt (#540).  BF16 -> FP16
// is exact for every value inside FP16's normal range (the 7-bit mantissa fits in 10 bits); weights and normalized
// activations sit there.  With the FP16 path, the weight is converted once per call and the activations in row
// slices, into the instance's own buffers, and the product runs as the FP16 GEMM (fp32 accumulate) on the tensor
// cores.  Finite values beyond FP16's range are clamped to +-65504 (#540) instead of becoming Inf.  A beta = 1
// product (STRATA_PREFILL_BF16X2's remainder, ~2^-9 of the original, deep in FP16's subnormal band) keeps cuBLAS.
//   Default: on for compute capability 7.0 - 7.4 (Volta: only the experimental STRATA_EXPERIMENTAL_SM60 build runs
//   there), OFF for 7.5 (RTX 20 in the ready-made engine: the sums round differently, so it is opt-in until it has
//   been gated).  STRATA_BF16_TC=1 / =0 turns it on / off on any 7.x card (=2, a test mode: on any card).
// Pascal (6.x) has no tensor cores and cuBLAS has no BF16 GEMM for it (#395, measured NOT_SUPPORTED on a P40): both
// operands are widened to fp32 by an exact shift and the product is cublasSgemm with the same fp32 accumulator.
namespace {
__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ in, __half* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = __uint_as_float((uint32_t) in[i] << 16);
        if (f > 65504.0f && !isinf(f)) f = 65504.0f;
        else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
        out[i] = __float2half_rn(f);
    }
}
void bf16_to_f16(const uint16_t* in, uint16_t* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, reinterpret_cast<__half*>(out), n);
}
#if defined(STRATA_EXPERIMENTAL_SM60)   // Pascal runs only the experimental build
__global__ void bf16_to_f32_kernel(const uint16_t* __restrict__ in, float* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float((uint32_t) in[i] << 16);
}
void bf16_to_f32(const uint16_t* in, float* out, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, st>>>(in, out, n);
}
#endif
// The current device's compute capability as 10 * major + minor, per device (a layer split can mix cards); 0: not
// known, read as "not an old card" so a failed query keeps the cuBLAS BF16 call.
int current_cc() {
    static std::atomic<int> cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    int v = cc[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        v = 10 * major + minor;
        cc[dev].store(v, std::memory_order_relaxed);
    }
    return v;
}
// 0: cuBLAS's BF16 GEMM, 1: through FP16 (tensor cores), 2: through FP32 (Pascal)
int bf16_path() {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_BF16_TC");
        return v != nullptr && v[0] != '\0' ? std::atoi(v) : -1;
    }();
    const int cc = current_cc();
    if (forced == 2 && cc > 0) return 1;   // a test mode: the FP16 path on any card (gemm_bf16_parity on sm_80+)
    if (cc <= 0 || cc >= 80) return 0;
    if (cc < 70) return 2;
    if (forced >= 0) return forced != 0 ? 1 : 0;
    return cc < 75 ? 1 : 0;
}
bool grow(uint16_t*& p, int64_t& have, int64_t want) {   // `have`, `want`: 2-byte elements
    if (have >= want) return true;
    if (p) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc((void**) &p, (size_t) want * 2) != cudaSuccess) { cudaGetLastError(); p = nullptr; return false; }
    have = want;
    return true;
}
constexpr int64_t kXSliceElems = 16ll << 20;   // 32 MiB of FP16 activations per slice (64 MiB as fp32)
}  // namespace
#endif
bool Gemm::bf16_hcd_exact(const uint16_t* X, int64_t ldx, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K) {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE) && defined(STRATA_USE_HIP)
    static std::atomic<bool> told{false};
    if (!hipblaslt_state_ || N != 320 || K != 10240 || T < 1) {
        if (N == 320 && K == 10240 && !told.exchange(true))
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: no hipBLASLt tuning table is loaded, so the exact kernel has nothing to match: "
                                 "the hyper-connection down projection stays on hipBLAS\n");
        return false;
    }
    // The kernel copies hipBLASLt solution 1176 / 1177's summation order, which is only true of the library build the
    // table was calibrated with: take it only when the table makes hipBLASLt pick one of those two for this T.
    const int id = hipblaslt_solution_for(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, T, N, K, N, 0.0f);
    if (id != 1176 && id != 1177) {
        if (T >= 4096 && !told.exchange(true))   // (a short chunk may legitimately get another solution: not worth a line)
            std::fprintf(stderr, "strata: STRATA_HCD_EXACT: the tuning table gives the hyper-connection down projection (N 320, K 10240) "
                                 "solution %d at T %lld, not 1176 / 1177: hipBLASLt runs it (a changed table or library; "
                                 "tools/hip/gfx1151-hipblaslt-100401.txt is the one measured)\n",
                         id, (long long) T);
        return false;
    }
    return strata_pf_hcdown_exact_bf16(X, ldx, W, Y, T, N, K, stream_);
#else
    (void) X; (void) ldx; (void) W; (void) Y; (void) T; (void) N; (void) K;
    return false;
#endif
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// STRATA_HC_LT (opt-in).  The hyper-connection read's two 16-bit GEMMs - down (D 10240 -> LR 320) and up
// (LR 320 -> D 10240), prefill.cpp's N 2560 / HC 4 / D 10240 / LR 320 - run on the FP16 prompt activation
// image (set_act_f16), which sends them to rocBLAS's f16_inplace.  On gfx928 hipBLASLt's BF16 matrix-core
// solutions are ~2x faster for those two shapes (measured on the card: up 5.43 ms vs 10.7 ms, down 4.51 ms
// vs 10.8 ms).  The resident weights are already BF16, so only the activation image has to change; this
// converts it to BF16 with the same rounding `bf()` (kernels.cu) applies, so the image is what the non-FP16
// path would have held.  Shapes the tuning table does not cover return false and the FP16 path runs as before.
__global__ void strata_hc_f16_to_bf16_kernel(const uint16_t* __restrict__ in, uint16_t* __restrict__ out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u = __float_as_uint(__half2float(__ushort_as_half(in[i])));
    u += 0x7fffu + ((u >> 16) & 1u);
    out[i] = (uint16_t) (u >> 16);
}
#endif
void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta, int64_t ldx) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (ldx <= K) ldx = 0;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1; STRATA_WMMA_BF16=0 excludes bf16): RDNA3 / RDNA3.5 WMMA dense GEMM, before
    // hipBLASLt; it returns false for shapes it does not take (and off gfx11) and this falls through
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool bf16_on = [] { const char* v = std::getenv("STRATA_WMMA_BF16"); return !v || (v[0] != 0 && v[0] != '0'); }();
    if (wmma_on && bf16_on && ldx == 0 && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_bf16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
    if (f16_io_) {   // X is the FP16 image (set_act_f16); W goes through the dequantization scratch as FP16
        // the only accumulating BF16 product is the BF16X2 low part, which bf16x2_mode() turns off with this path
        if (beta != 0.0f) { std::fprintf(stderr, "prefill gemm: an accumulating BF16 GEMM on the FP16 prompt path\n"); std::exit(1); }
#if defined(STRATA_HIPBLASLT_AVAILABLE)
        // STRATA_HC_LT=up | down | <truthy> (both).  Off (unset/0) by default, and only these two shapes are
        // routed - every other 16-bit GEMM keeps the FP16 path even if the tuning table covers it.  The BF16
        // activation image is a function-local buffer: the prompt path has one Gemm instance, so nothing else
        // moves and no header changes.
        static const int hc_lt = [] {
            const char* v = std::getenv("STRATA_HC_LT");
            // hcup-20261013: the up projection is the DEFAULT (unset).  down stays on the FP16 path.
            if (v == nullptr || v[0] == '\0') return 1;
            if (v[0] == '0') return 0;
            if (std::strcmp(v, "up") == 0) return 1;
            if (std::strcmp(v, "down") == 0) return 2;
            return 3;
        }();
        const bool hc_lt_shape = ((hc_lt & 1) != 0 && N == 10240 && K == 320) ||
                                 ((hc_lt & 2) != 0 && N == 320 && K == 10240);
        static uint16_t* hc_xbf = nullptr;
        static int64_t hc_xbf_elems = 0;
        if (hc_lt_shape) {
            const int64_t xn = T * K;
            if (hc_xbf_elems < xn) {
                if (hc_xbf) cudaFree(hc_xbf);
                hc_xbf = nullptr; hc_xbf_elems = 0;
                if (cudaMalloc((void**) &hc_xbf, (size_t) xn * 2) == cudaSuccess) hc_xbf_elems = xn;
                else cudaGetLastError();
            }
            if (hc_xbf_elems >= xn) {
                strata_hc_f16_to_bf16_kernel<<<(unsigned) ((xn + 255) / 256), 256, 0, (cudaStream_t) stream_>>>(
                    X, hc_xbf, xn);
                if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, hc_xbf, W, Y, T, N, K,
                                  ldy, beta, stream_, 0)) {
                    STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt bf16 (hc)");
                    return;
                }
            }
        }
#endif
        const int64_t rows = std::min<int64_t>(N, scratch_elems_ / K);
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min(rows, N - r0);
            bf16_to_f16_rows<<<1024, 256, 0, (cudaStream_t) stream_>>>(W + r0 * K, (__half*) scratch_, n * K);
            f16_inplace(X, scratch_, Y + r0, T, n, K, ldy);
        }
        return;
    }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_, ldx)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt bf16");
        return;
    }
#endif
#if !defined(__HIPCC__)
    if (const int path = K > 0 ? bf16_path() : 0; path == 1 && N > 1 && beta == 0.0f) {
        // a single output row stays cuBLAS's GEMV, faster than the conversions; beta = 1: see above
        const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        if (grow(tc_w_, tc_w_elems_, N * K) && grow(tc_x_, tc_x_elems_, x_rows * K)) {
            bf16_to_f16(W, tc_w_, N * K, (cudaStream_t) stream_);
            for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                const int64_t n = std::min<int64_t>(x_rows, T - t0);
                bf16_to_f16(X + t0 * K, tc_x_, n * K, (cudaStream_t) stream_);
                f16(tc_x_, tc_w_, Y + t0 * ldy, n, N, K, ldy, beta);
            }
            return;
        }
    }
#if defined(STRATA_EXPERIMENTAL_SM60)
    else if (path == 2) {
        // Pascal: fp32 copies (2 elements of the 2-byte buffers each).  Every tile is a disjoint block of Y, so each
        // gets the caller's beta.
        const int64_t x_rows = std::max<int64_t>(1, std::min<int64_t>(T, kXSliceElems / K));
        if (grow(tc_w_, tc_w_elems_, 2 * N * K) && grow(tc_x_, tc_x_elems_, 2 * x_rows * K)) {
            float* const wf = reinterpret_cast<float*>(tc_w_);
            float* const xf = reinterpret_cast<float*>(tc_x_);
            bf16_to_f32(W, wf, N * K, (cudaStream_t) stream_);
            for (int64_t t0 = 0; t0 < T; t0 += x_rows) {
                const int64_t n = std::min<int64_t>(x_rows, T - t0);
                bf16_to_f32(X + t0 * K, xf, n * K, (cudaStream_t) stream_);
                ck(cublasSgemm((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) n, (int) K, &alpha,
                               wf, (int) K, xf, (int) K, &beta, Y + t0 * ldy, (int) ldy),
                   "cublasSgemm (bf16 on Pascal)");
            }
            return;
        }
    }
#endif
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) (ldx ? ldx : K), &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta, bool y_f16) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
#ifdef STRATA_USE_HIP
    // #313 (opt-in STRATA_WMMA_GEMM=1): RDNA3 / RDNA3.5 WMMA dense GEMM, before hipBLASLt (falls through when false)
    static const bool wmma_on = [] { const char* v = std::getenv("STRATA_WMMA_GEMM"); return v && v[0] != 0 && v[0] != '0'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pf_on && T >= pf_min_t && strata_pf_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) return;   // S23: opt-in
    if (wmma_on && T >= 16 && (beta == 0.0f || beta == 1.0f) &&
        strata_wmma_gemm_f16(X, W, Y, T, N, K, ldy, beta, stream_)) {
        return;
    }
#endif
#if defined(__HIPCC__)
        if (f16_io_ && beta == 0.0f) { f16_inplace(X, W, Y, T, N, K, ldy, y_f16); return; }
#endif
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        STRATA_ABSORB_HIPBLAS_STICKY("hipBLASLt f16");
        return;
    }
#endif
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
    STRATA_ABSORB_HIPBLAS_STICKY("cublasGemmEx f16");
}

#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
bool Gemm::native_mmq(const uint16_t* X, int type, const void* W, float* Y, int64_t T, int64_t N, int64_t K,
                      int64_t ldy) {
    namespace mmq = strata::prefill::mmq;
    constexpr int64_t kRows = 1024, kMaxK = 8192;
    static const bool enabled = [] {
        const char* e = std::getenv("STRATA_DENSE_MMQ");
        return e && e[0] == '1';
    }();
    if (!enabled || mmq_failed_ || !mmq::built() || !mmq::fits(type, N)) return false;
    // K must be a multiple of 256: llama.cpp's MMQ loads the weights in 256-value K chunks, and the
    // chunk past a partial row reads past the row (the next row's bytes - or, for the last weight
    // row, bytes past the tensor, which no caller guarantees to be zeros). The MoE path is safe
    // because its gather buffers carry a zeroed tail; a dense tensor is passed as-is, so a K that
    // is not a full chunk multiple (e.g. the 640-value shared-expert down: 20 blocks of 32) stays on
    // the dequantize + cuBLAS path.
    if (K > kMaxK || K % 256 != 0) return false;
    int64_t R = scratch_elems_ / (2 * K);   // FP32 activations for a chunk fit in the FP16 dequant scratch
    if (R > kRows) R = kRows;
    if (R > T) R = T;
    if (R < 1) return false;
    if (ldy <= 0) ldy = N;

    const auto up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    const size_t ident_off = up(mmq::q8_bytes(kRows, kMaxK));
    const size_t bounds_off = ident_off + up((size_t) kRows * 4);
    if (!mmq_buf_) {
        if (cudaMalloc(&mmq_buf_, bounds_off + 256) != cudaSuccess) {
            cudaGetLastError();
            mmq_failed_ = true;
            return false;
        }
        mmq_ctx_ = new mmq::Context();
        mmq::iota((int32_t*) ((uint8_t*) mmq_buf_ + ident_off), kRows, stream_);
        std::fprintf(stderr, "prefill gemm: dense MMQ on (STRATA_DENSE_MMQ=1)\n");
    }
    void* xq = mmq_buf_;
    const int32_t* ident = (const int32_t*) ((uint8_t*) mmq_buf_ + ident_off);
    int32_t* bounds = (int32_t*) ((uint8_t*) mmq_buf_ + bounds_off);
    float* xf = (float*) scratch_;
    auto* ctx = static_cast<mmq::Context*>(mmq_ctx_);

    for (int64_t r0 = 0; r0 < T; r0 += R) {
        const int64_t rows = (T - r0 < R) ? T - r0 : R;
        mmq::f16_to_f32(X + r0 * K, xf, rows * K, stream_);
        mmq::quantize(xf, nullptr, xq, type, K, K, rows, stream_);
        mmq::set_bounds(bounds, (int32_t) rows, stream_);
        mmq::Product p;
        p.w = W;
        p.type = type;
        p.w_rows = N;
        p.w_cols = K;
        p.expert_bytes = mmq::matrix_bytes(type, N, K);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ident;
        p.total_rows = rows;
        p.max_rows = rows;
        p.dst = Y + r0 * ldy;
        p.ld_dst = ldy;
        ctx->run(p, stream_);
    }
    return true;
}
#endif
// wv2m-20261015 (opt-in STRATA_HC_MMAC=1): the fused hyper-connection down+inject GEMM for gfx928 -- the same
// job as wmma_gemm.cu's gfx11 kernel_hcdown (which is NOT compiled into this engine: sort-native configured
// STRATA_ENABLE_HIP=OFF, so that TU is absent and strata_pf_hcdown_bf16 is the header stub), on this card's
// bf16 MMAC. Fragment mapping identical to gr_upmix_mmac's:
//   A lane l (of 64) holds A[l%16][(l/16)*4 + k];  B lane l holds B[(l/16)*4 + k][l%16];
//   C lane l holds C[l%16][(l/16)*4 + j] for j 0..3 (four outputs per lane).
// One wave = a 16-token x 16-output tile; a block of 8 waves stages a 128-token x 64-k X chunk and a
// 128-output x 64-k weight chunk into LDS, then each wave walks the whole K loop. The weight tile spans
// [w_down; w_inject] (row n < nd from Wd, else Wi); the epilogue writes lo or inj accordingly.
// Rounding-level: the K order differs from hipBLASLt's -- the gates are the whole-model token sha AND the
// retrieval-quality check.
#if defined(__HIPCC__)
typedef __bf16 hcd_b4 __attribute__((ext_vector_type(4)));
typedef float  hcd_f4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ hcd_f4 hcd_mmac(hcd_b4 a, hcd_b4 b, hcd_f4 c) {
#if defined(__gfx928__)
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
#else
    (void) a; (void) b; return c;   // not a gfx928 device pass: never launched (runtime gate)
#endif
}
constexpr int HCD_WAVES = 4;            // 4 waves of 64 = 256 threads per block (double-buffered LDS fits)
constexpr int HCD_BM = 16 * HCD_WAVES;  // 64 tokens per block
constexpr int HCD_BN = 64;              // outputs per block (nd + ni = 324 => 6 blocks)
constexpr int HCD_BK = 64;
constexpr int HCD_LDK = HCD_BK + 8;
__global__ void __launch_bounds__(HCD_WAVES * 64) kernel_hcdown_mmac(
    const uint16_t* __restrict__ X, int ldx, const uint16_t* __restrict__ Wd, const uint16_t* __restrict__ Wi,
    int nd, int ni, float* __restrict__ lo, float* __restrict__ inj, int M, int K) {
#if defined(__gfx928__)
    // wv2n-20261015 (STRATA_HC_MMAC_DB=1): double-buffered staging -- load chunk k+1 into the other buffer
    // while chunk k is consumed, one barrier pair per chunk instead of two (160 -> 80 pairs over K=10240).
    // Same fragments, same K order, same bits as the single-buffered HC_MMAC kernel.
    __shared__ __align__(16) uint16_t sA[2][HCD_BM][HCD_LDK];
    __shared__ __align__(16) uint16_t sB[2][HCD_BN][HCD_LDK];
    const int N = nd + ni;
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
    const int r = lane & 15, g = lane >> 4;
    const int num_m = (M + HCD_BM - 1) / HCD_BM;
    const int m0 = (blockIdx.x % num_m) * HCD_BM, n0 = (blockIdx.x / num_m) * HCD_BN;
    const int t0 = m0 + 16 * wave;               // this wave's 16 tokens
    const int n_base = n0 + 16 * wave;           // this wave's 16 outputs
    const int sr = tid >> 3, sq = tid & 7;       // staged tile: 128 rows x 64 k, 8 halves per lane
    auto ldA = [&](int k0, int row) -> uint4 {
        return *reinterpret_cast<const uint4*>(X + (size_t) min(m0 + row, M - 1) * (int64_t) ldx + k0 + 8 * sq);
    };
    auto ldB = [&](int k0, int row) -> uint4 {
        const int n = min(n0 + row, N - 1);
        const uint16_t* src = n < nd ? Wd + (size_t) n * (size_t) K : Wi + (size_t) (n - nd) * (size_t) K;
        return *reinterpret_cast<const uint4*>(src + k0 + 8 * sq);
    };
    uint4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3;
    auto load = [&](int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
    };
    auto store = [&](int b) {
        *reinterpret_cast<uint4*>(&sA[b][sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA[b][sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA[b][sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA[b][sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB[b][sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB[b][sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB[b][sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB[b][sr + 96][8 * sq]) = rb3;
    };
    hcd_f4 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] = hcd_f4{0.f, 0.f, 0.f, 0.f};
    const bool hcd_db = true;   // wv2n: double-buffering unconditional (same bits; the single-buffer
                                // variant is the sec.172 measurement, kept in the .pre_hcddb backup)
    if (!hcd_db) {
        // the single-buffered path (unchanged bits, the sec.172 baseline)
        __shared__ __align__(16) uint16_t sA1[HCD_BM][HCD_LDK];
        __shared__ __align__(16) uint16_t sB1[HCD_BN][HCD_LDK];
        load(0);
        *reinterpret_cast<uint4*>(&sA1[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA1[sr + 32][8 * sq]) = ra1;
        *reinterpret_cast<uint4*>(&sA1[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA1[sr + 96][8 * sq]) = ra3;
        *reinterpret_cast<uint4*>(&sB1[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB1[sr + 32][8 * sq]) = rb1;
        *reinterpret_cast<uint4*>(&sB1[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB1[sr + 96][8 * sq]) = rb3;
        __syncthreads();
        for (int k0 = 0; k0 < K; k0 += HCD_BK) {
            const bool more = k0 + HCD_BK < K;
            if (more) load(k0 + HCD_BK);
#pragma unroll
            for (int ks = 0; ks < HCD_BK; ks += 16) {
                const hcd_b4 b = *reinterpret_cast<const hcd_b4*>(&sB1[16 * wave + r][ks + g * 4]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const hcd_b4 a = *reinterpret_cast<const hcd_b4*>(&sA1[16 * j + r][ks + g * 4]);
                    acc[j] = hcd_mmac(a, b, acc[j]);
                }
            }
            if (more) {
                __syncthreads();
                *reinterpret_cast<uint4*>(&sA1[sr][8 * sq]) = ra0; *reinterpret_cast<uint4*>(&sA1[sr + 32][8 * sq]) = ra1;
                *reinterpret_cast<uint4*>(&sA1[sr + 64][8 * sq]) = ra2; *reinterpret_cast<uint4*>(&sA1[sr + 96][8 * sq]) = ra3;
                *reinterpret_cast<uint4*>(&sB1[sr][8 * sq]) = rb0; *reinterpret_cast<uint4*>(&sB1[sr + 32][8 * sq]) = rb1;
                *reinterpret_cast<uint4*>(&sB1[sr + 64][8 * sq]) = rb2; *reinterpret_cast<uint4*>(&sB1[sr + 96][8 * sq]) = rb3;
                __syncthreads();
            }
        }
    } else {
        store(0);
        __syncthreads();
        int b = 0;
        for (int k0 = 0; k0 < K; k0 += HCD_BK) {
            const bool more = k0 + HCD_BK < K;
            if (more) load(k0 + HCD_BK);
#pragma unroll
            for (int ks = 0; ks < HCD_BK; ks += 16) {
                const hcd_b4 bb = *reinterpret_cast<const hcd_b4*>(&sB[b][16 * wave + r][ks + g * 4]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const hcd_b4 a = *reinterpret_cast<const hcd_b4*>(&sA[b][16 * j + r][ks + g * 4]);
                    acc[j] = hcd_mmac(a, bb, acc[j]);
                }
            }
            if (more) {
                __syncthreads();          // all waves done reading buffer b
                store(b ^ 1);             // stage k+1 into the other buffer
                __syncthreads();
                b ^= 1;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n = n_base + g * 4 + j;
        if (n >= N) continue;
        const int m = t0 + r;
        if (m >= M) continue;
        if (n < nd) lo[(size_t) m * nd + n] = acc[j][0];
        else inj[(size_t) m * ni + (n - nd)] = acc[j][0];
    }
#endif
}
bool Gemm::hcdown_mmac(const uint16_t* X, int64_t ldx, const uint16_t* Wd, const uint16_t* Wi, int64_t nd,
                       int64_t ni, float* lo, float* inj, int64_t T, int64_t K) {
    if (!X || !Wd || !Wi || !lo || !inj || T < 64 || nd < 1 || ni < 0 || K % 64 != 0 || ldx < K || ldx % 8 != 0) return false;
    if ((nd + ni) % 16 != 0) return false;       // the epilogue needs whole 16-wide n slices per wave
    static int ok = -1;
    if (ok < 0) {
        int dev = 0; cudaDeviceProp prop{};
        ok = (cudaGetDevice(&dev) == hipSuccess && cudaGetDeviceProperties(&prop, dev) == hipSuccess &&
              strstr(prop.gcnArchName, "gfx928") != nullptr) ? 1 : 0;
    }
    if (ok != 1) return false;
    const int64_t mt = (T + HCD_BM - 1) / HCD_BM, nt = (nd + ni + HCD_BN - 1) / HCD_BN;
    kernel_hcdown_mmac<<<(unsigned) (mt * nt), HCD_WAVES * 64, 0, (cudaStream_t) stream_>>>(
        X, (int) ldx, Wd, Wi, (int) nd, (int) ni, lo, inj, (int) T, (int) K);
    const cudaError_t e = cudaGetLastError();
    if (e != hipSuccess) { (void) cudaGetLastError(); return false; }
    return true;
}
#endif

// wv2q-20261015 (opt-in STRATA_PADR_LT=1): the padr GEMM through hipBLASLt with a per-shape solution ID
// from STRATA_PADR_LT_SOL (e.g. "10240x2560:3885,6144x2560:3885,2560x6144:3884"), f16-out + widen exactly
// like the rocBLAS pair it replaces. Rounding-level (Lt's K order differs from rocBLAS's): the whole-model
// token sha is the gate. Fail-closed: any setup miss falls back to the rocBLAS pair.
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
namespace {
struct PadrLt {
    hipblasLtHandle_t handle = nullptr;
    // cache: (N,K,ldw,T) -> {desc, la, lb, lc, algo}
    struct Entry {
        hipblasLtMatmulDesc_t op = nullptr;
        hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
        hipblasLtMatmulAlgo_t algo{};
        bool valid = false;
    };
    std::vector<std::pair<uint64_t, Entry>> cache;
    hipblasLtMatmulHeuristicResult_t make(hipblasLtHandle_t h, long N, long T, long K, long ldw, long ldb, long ldy,
                                          int sol, Entry& e) {
        hipblasLtMatmulDesc_t op = nullptr;
        hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS) return {};
        hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta);
        hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb);
        if (hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, K, N, ldw) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, T, ldb) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, N, T, 2 * ldy) != HIPBLAS_STATUS_SUCCESS) {
            if (la) hipblasLtMatrixLayoutDestroy(la);
            if (lb) hipblasLtMatrixLayoutDestroy(lb);
            hipblasLtMatmulDescDestroy(op);
            return {};
        }
        std::vector<int> ids{sol};
        std::vector<hipblasLtMatmulHeuristicResult_t> cand;
        if (hipblaslt_ext::getAlgosFromIndex(h, ids, cand) != HIPBLAS_STATUS_SUCCESS || cand.empty()) {
            hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc);
            hipblasLtMatmulDescDestroy(op);
            return {};
        }
        e.op = op; e.la = la; e.lb = lb; e.lc = lc; e.algo = cand.front().algo; e.valid = true;
        return cand.front();
    }
};
PadrLt g_padr_lt;
bool padr_lt_ready = false;
std::vector<std::pair<uint64_t, int>> g_sol_map;   // (N<<32|K) -> solution id
int padr_lt_sol_for(long N, long K) {
    for (auto& kv : g_sol_map) if (kv.first == ((uint64_t) N << 32 | (uint32_t) K)) return kv.second;
    return -1;
}
}  // namespace
#endif

void Gemm::f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                        bool y_f16) {
#if defined(__HIPCC__)
    // wv2j-20261015 (opt-in STRATA_GEMM_F32OUT=1; unset = the FP16-out + widen pair, unchanged): write the
    // FP32 accumulator straight to Y instead of rounding it to FP16 and widening it back. One fewer FP16
    // rounding and one fewer read+write of N*T halves; the products, the accumulation order and the input
    // bit widths are untouched, so the correctness gate is the whole-model token sha.
    static const bool f32out = [] { const char* v = std::getenv("STRATA_GEMM_F32OUT"); return v && v[0] == '1'; }();
        if (f32out && !y_f16) {   // wv2r-20261016: FP32 in Y is not a compact FP16 image
        const float one = 1.0f, zero = 0.0f;
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, CUDA_R_16F,
                        (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT),
           "cublasGemmEx f16 f32out");
        return;
    }
    const float one = 1.0f, zero = 0.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, CUDA_R_16F,
                                        (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16 out");
        static const bool dbg_nan = std::getenv("STRATA_DBG_NAN") != nullptr;
    if (y_f16) {
        static bool yf16_said = false;
        if (!yf16_said) {
            yf16_said = true;
            std::fprintf(stderr, "prefill gemm: f16_inplace compact FP16 image (STRATA_GEMM_Y_F16), widen skipped\n");
        }
    } else {
        widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, dbg_nan ? 1 : 0);
    }
    if (dbg_nan) {   // debug only: a sync per GEMM
        unsigned long long bad = 0, zero_count = 0;
        (void) hipStreamSynchronize((hipStream_t) stream_);
        if (hipMemcpyFromSymbol(&bad, HIP_SYMBOL(g_f16_nonfinite), sizeof bad) == hipSuccess && bad != 0) {
            std::fprintf(stderr, "strata prefill: STRATA_DBG_NAN: %llu non-finite FP16 GEMM outputs (T=%lld N=%lld K=%lld; FP16 ends at 65504)%s",
                         bad, (long long) T, (long long) N, (long long) K, "\n");
            (void) hipMemcpyToSymbol(HIP_SYMBOL(g_f16_nonfinite), &zero_count, sizeof zero_count);
        }
    }
#else
    (void) X; (void) W; (void) Y; (void) T; (void) N; (void) K; (void) ldy;
#endif
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx, bool y_f16) {
#ifdef STRATA_USE_HIP
    // S23 (opt-in STRATA_PF_PAD=1, with STRATA_PF_GEMM=1 on chunks of STRATA_PF_SWITCH_MIN_T+ tokens): the weight
    // dequantized into the scratch with row stride K + 64 halves, so no 4 KB-multiple stride camps on the memory
    // channels (gemm_probe8: K 6144 19 -> 29 TFLOPS from W alone, 35 with X padded too).  Bitwise the same products.
    static const bool pad_on = [] { const char* v = std::getenv("STRATA_PF_PAD"); return v && v[0] == '1'; }();
    static const bool pf_on = [] { const char* v = std::getenv("STRATA_PF_GEMM"); return v && v[0] == '1'; }();
    static const int64_t pf_min_t = [] { const char* v = std::getenv("STRATA_PF_SWITCH_MIN_T"); return v ? (int64_t) std::atoll(v) : (int64_t) 0; }();
    if (pad_on && pf_on && T >= pf_min_t && T >= 64 && N >= 512 && K % 32 == 0 && N * (K + 64) <= scratch_elems_) {
        const int64_t ldw = K + 64;
        if (strata::kernels::dequant_f16_ld(ggml_type, W_blocks, 0, N, K, ldw, scratch_, stream_) &&
            strata_pf_gemm_f16_ld(X, ldx > 0 ? ldx : K, scratch_, ldw, Y, T, N, K, ldy, beta, stream_)) {
            return;
        }
    }
#endif
#if defined(__HIPCC__)
    // pad-20261014 (opt-in STRATA_PF_PAD_ROCBLAS=1, default off): dequantize the FP16 weight with row stride
    // K + pad and hand rocBLAS lda = K + pad; the output stays FP16 at ldc = 2*ldy and widen_rows_f16 follows,
    // exactly as f16_inplace does.  pad is chosen PER SHAPE, and the two stages are COUPLED:
    //
    //   K 2560 -> 32, X stride irrelevant (Stage 1).  Parent's whole-model A/B (HCU5, ltlong, 30,602 tokens):
    //     the gdn phase 2,225/2,209 -> 2,012/2,014 ms = -204 ms, qsa proj -48.5 ms -- it matches the isolated
    //     GEMM prediction (-80.3 + -125.8 = -206 ms) to 1%, so this transfer is ~100%; logits 5/5 bit-identical.
    //     Isolated (dense_replay3 mode 1, T = 30602): attn_qkv 27.903 -> 25.182, attn_gate 13.595 -> 10.613,
    //     attn_q 30.343 -> 22.661 ms/layer.
    //   K 6144 -> 32 ONLY while X arrives padded too (ldx != 0 && ldx != K), else 0.  W-only is a REGRESSION
    //     here: with X still at 6144 the whole model's gdn-out phase went 1,016/1,021 -> 1,129/1,129 (+110 ms)
    //     even though the isolated GEMM reads 20.677 -> 18.414 ms - a component win that does not transfer.
    //     W 32 + X 32 is the only winning cell (ssm_out 14.536, attn_output 14.233 ms); W 32 + X 64 is the
    //     worst (21.014 / 21.140), which is why prefill.cpp's ZV_PAD is 32 and not the 64 it shipped with.
    //   else -> 0.  The K test already excludes hc_down (K=10240), hc_up (320), hc_inject (10240) and
    //     ffn_down_shexp (640).  N >= 1024 excludes attn_k/attn_v (512) and ffn_gate_shexp (640): the latter is
    //     in the parent's N >= 512 sketch but REGRESSES (1.474 -> 2.052 ms/layer), so this is a correction.
    //     hc_down regresses W-only too (5.318 -> 7.163 ms/layer).
    //
    // In scope in the shipped model: attn_qkv/attn_q/attn_gate (K 2560) and, only with the X padding on,
    // attn_output/ssm_out (K 6144).  m.mixed_h (qkv/gate/attn_q) and m.xn16 (hc_down) are OUT of scope: their
    // producers would need new padded allocations.  Whole-model A/B/A and the acceptance logits gate are the
    // parent's HCU5 job and are not run here.
    // Precondition: this needs f16_io_ (the stage's 16-bit GEMMs on the FP16 in/out image,
    // STRATA_HIP_PROMPT_F16=1).  Without it the pad path is inert and nothing changes.  Fail-closed: a strided
    // dequant that cannot run, or a scratch too small, falls through; the ldx != K branch below never aborts.
    // wv2-20261020 (default OFF, byte-for-byte the parent path): STRATA_PF_PAD_ROCBLAS is now a VALUE.
    //   unset / 0 -> off (parent default);  1 -> pad 32 (the parent shipped behaviour, unchanged);
    //   2 -> pad 64 on the K == 2560 shapes ONLY.  The K == 6144 shape keeps pad 32 because that is the cell
    // the parent whole-model A/B measured as its best (W 32 + X 32).  The padding is a rocBLAS lda/ldw
    // argument, so it changes the GEMM reduction order and is NOT expected to be bitwise identical - the
    // bit-exactness of this one is the parent agent whole-model token-level gate, not this tree.
    static const int padr_mode = [] { const char* v = std::getenv("STRATA_PF_PAD_ROCBLAS"); return v ? std::atoi(v) : 0; }();
    const bool padr_on = (padr_mode == 1 || padr_mode == 2);
    const bool x_padded = (ldx != 0 && ldx != K);
    const int64_t padr = (N >= 1024) ? ((K == 2560) ? (padr_mode == 2 ? 64 : 32)
                                                  : ((K == 6144 && x_padded) ? 32 : 0)) : 0;
    if (padr_on && padr > 0 && f16_io_ && beta == 0.0f && T >= 64 && N * (K + padr) <= scratch_elems_) {
        if (ldy <= 0) ldy = N;
        const int64_t ldw = K + padr;
        const int64_t ldb = ldx > 0 ? ldx : K;   // Stage 2a: X may arrive padded at its own stride
        // wv2i-20261015 (opt-in STRATA_GEMM_SPLIT=1; unset = no events, no path change): the 256-slot ring
        // (M41) splits the padr path into [dequant_f16_ld] + [cublasGemmEx] + [widen_rows_f16], summed PER
        // SHAPE so the dense-projection ledger can be attributed. All stages are on stream_ in order, so
        // elapsed times between the events are exact stage times.
        static const bool gsplit = std::getenv("STRATA_GEMM_SPLIT") != nullptr;
        static const int GS_N = 32;   // build#18: 256 exceeded the call count (~90-100 padr calls per pass), so the read gate never opened (M43: ring depth must fit the call count)
        static hipEvent_t gev[GS_N][4] = {};
        static int gs_slot = 0;
        static long gs_calls = 0;
        struct GsAgg { long calls = 0; double deq = 0, g = 0, wid = 0; };
        static GsAgg gs_agg[8];
        static int gs_shape = -1;
        if (gsplit) {
            int si = 0;
            for (; si < 8; ++si) {   // per-shape slots keyed by (N, K)
                if (gs_agg[si].calls == 0) { gs_agg[si] = GsAgg{}; gs_shape = si; break; }
                GsAgg probe; (void) probe;
                // reuse slot si only if it is the same shape: keep a tiny shape table
                static long long sN[8] = {}, sK[8] = {};
                if (sN[si] == 0) { sN[si] = N; sK[si] = K; gs_shape = si; break; }
                if (sN[si] == N && sK[si] == K) { gs_shape = si; break; }
            }
            if (si == 8) gs_shape = 0;   // more than 8 shapes: fold into slot 0
            if (gs_calls == 0)
                for (int ie = 0; ie < GS_N; ++ie) for (int je = 0; je < 4; ++je) (void) hipEventCreate(&gev[ie][je]);
            gs_slot = (int) (gs_calls % GS_N);
            if (gs_calls >= GS_N) {
                float da = 0.f, db = 0.f, dc = 0.f;
                const bool a = hipEventElapsedTime(&da, gev[gs_slot][0], gev[gs_slot][1]) == hipSuccess;
                const bool b = hipEventElapsedTime(&db, gev[gs_slot][1], gev[gs_slot][2]) == hipSuccess;
                const bool c = hipEventElapsedTime(&dc, gev[gs_slot][2], gev[gs_slot][3]) == hipSuccess;
                if (a && b && c) { gs_agg[gs_shape].deq += da; gs_agg[gs_shape].g += db; gs_agg[gs_shape].wid += dc; ++gs_agg[gs_shape].calls; }
            }
            if (gs_calls == 0)
                std::fprintf(stderr, "[GEMM_SPLIT] instrument active N=%lld K=%lld ldx=%lld ldy=%lld beta=%f T=%lld\\n",
                             (long long) N, (long long) K, (long long) ldx, (long long) ldy, beta, (long long) T);
            (void) hipEventRecord(gev[gs_slot][0], (hipStream_t) stream_);
        }
        if (strata::kernels::dequant_f16_ld(ggml_type, W_blocks, 0, N, K, ldw, scratch_, stream_)) {
            if (gsplit) (void) hipEventRecord(gev[gs_slot][1], (hipStream_t) stream_);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
            // wv2q-20261015 (opt-in STRATA_PADR_LT=1): the Lt f16-out GEMM, per-shape solution from
            // STRATA_PADR_LT_SOL. Fail-closed to the rocBLAS pair below on any miss.
            static const bool padr_lt_on = [] { const char* v = std::getenv("STRATA_PADR_LT"); return v && v[0] == '1'; }();
            if (padr_lt_on && !y_f16) {   // wv2r-20261016: the Lt path's lc is 2*ldy and its widen is unconditional
                if (!padr_lt_ready) {
                    padr_lt_ready = true;
                    if (hipblasLtCreate(&g_padr_lt.handle) == HIPBLAS_STATUS_SUCCESS) {
                        const char* spec = std::getenv("STRATA_PADR_LT_SOL");
                        if (spec) {
                            const char* p = spec;
                            while (*p) {
                                char* endp = nullptr;
                                long n = strtol(p, &endp, 10);
                                if (endp && *endp == 'x') {
                                    long k = strtol(endp + 1, &endp, 10);
                                    if (endp && *endp == ':') {
                                        long sol = strtol(endp + 1, &endp, 10);
                                        g_sol_map.push_back({(uint64_t) n << 32 | (uint32_t) k, (int) sol});
                                    }
                                }
                                const char* q = strchr(p, ';');
                                if (!q) break;
                                p = q + 1;
                            }
                        }
                    }
                }
                if (g_padr_lt.handle) {
                    const int sol = padr_lt_sol_for(N, K);
                    if (sol > 0) {
                        uint64_t key = (uint64_t) N << 32 | (uint32_t) K | (uint64_t) T << 20;
                        PadrLt::Entry* e = nullptr;
                        for (auto& kv : g_padr_lt.cache) if (kv.first == key) { e = &kv.second; break; }
                        if (!e) {
                            PadrLt::Entry ne;
                            if (g_padr_lt.make(g_padr_lt.handle, N, T, K, ldw, ldb, ldy, sol, ne).state == HIPBLAS_STATUS_SUCCESS)
                                g_padr_lt.cache.push_back({key, ne});
                            for (auto& kv : g_padr_lt.cache) if (kv.first == key) { e = &kv.second; break; }
                        }
                        if (e && e->valid) {
                            static void* lt_ws = nullptr;
                            static size_t lt_ws_b = 0;
                            if (!lt_ws) { lt_ws_b = 64ull << 20; if (hipMalloc(&lt_ws, lt_ws_b) != hipSuccess) lt_ws = nullptr; }
                            if (lt_ws) {
                                const float lt_one = 1.0f, lt_zero = 0.0f;
                                const hipblasStatus_t lt_st = hipblasLtMatmul(g_padr_lt.handle, e->op, &lt_one, scratch_, e->la, X, e->lb,
                                                    &lt_zero, Y, e->lc, Y, e->lc, &e->algo, lt_ws, lt_ws_b, 0);
                                if (lt_st == HIPBLAS_STATUS_SUCCESS) {
                                    if (gsplit) (void) hipEventRecord(gev[gs_slot][2], (hipStream_t) stream_);
                                    widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, 0);
                                    if (gsplit) (void) hipEventRecord(gev[gs_slot][3], (hipStream_t) stream_);
                                    static bool lt_said = false;
                                    if (!lt_said) {
                                        lt_said = true;
                                        std::fprintf(stderr, "prefill gemm: STRATA_PADR_LT active (sol map %s)\n",
                                                     std::getenv("STRATA_PADR_LT_SOL") ? std::getenv("STRATA_PADR_LT_SOL") : "");
                                    }
                                    if (gsplit) {
                                        if (++gs_calls % 64 == 0) {
                                            static long long sNp[8] = {}, sKp[8] = {};
                                            sNp[gs_shape] = N; sKp[gs_shape] = K;
                                            for (int si2 = 0; si2 < 8; ++si2)
                                                std::fprintf(stderr, "[GEMM_SPLIT] shape=%d N=%lld K=%lld calls=%ld deq=%.1f gemm=%.1f widen=%.1f ms\n",
                                                             si2, (long long) sNp[si2], (long long) sKp[si2], gs_agg[si2].calls,
                                                             gs_agg[si2].deq, gs_agg[si2].g, gs_agg[si2].wid);
                                        }
                                    }
                                    return;
                                } else {
                                    static bool lt_err_said = false;
                                    if (!lt_err_said) {
                                        lt_err_said = true;
                                        std::fprintf(stderr, "prefill gemm: STRATA_PADR_LT matmul failed status=%d (N=%lld K=%lld T=%lld); falling back\n",
                                                     (int) lt_st, (long long) N, (long long) K, (long long) T);
                                    }
                                }
                            }
                        } else {
                            static bool lt_noalgo_said = false;
                            if (!lt_noalgo_said) {
                                lt_noalgo_said = true;
                                std::fprintf(stderr, "prefill gemm: STRATA_PADR_LT no usable entry for N=%lld K=%lld T=%lld\n",
                                             (long long) N, (long long) K, (long long) T);
                            }
                        }
                    }
                }
            }
#endif
            const float padr_one = 1.0f, padr_zero = 0.0f;
            static const bool f32out = [] { const char* v = std::getenv("STRATA_GEMM_F32OUT"); return v && v[0] == '1'; }();
            if (f32out) {   // wv2j-20261015: FP32 direct out, no widen (same gate as f16_inplace above)
                ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &padr_one,
                                scratch_, CUDA_R_16F, (int) ldw, X, CUDA_R_16F, (int) ldb, &padr_zero, Y, CUDA_R_32F,
                                (int) ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                   "cublasGemmEx f16 pad f32out");
                if (gsplit) (void) hipEventRecord(gev[gs_slot][2], (hipStream_t) stream_);
            } else {
                ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &padr_one,
                                scratch_, CUDA_R_16F, (int) ldw, X, CUDA_R_16F, (int) ldb, &padr_zero, Y, CUDA_R_16F,
                                (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                   "cublasGemmEx f16 pad");
                if (gsplit) (void) hipEventRecord(gev[gs_slot][2], (hipStream_t) stream_);
                if (y_f16) {
                    static bool yf16_said = false;
                    if (!yf16_said) {
                        yf16_said = true;
                        std::fprintf(stderr, "prefill gemm: STRATA_GEMM_Y_F16: compact FP16 image, widen_rows_f16 skipped\n");
                    }
                } else {
                    widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, 0);
                }
            }
            if (gsplit) {
                (void) hipEventRecord(gev[gs_slot][3], (hipStream_t) stream_);
                if (++gs_calls % 64 == 0) {   // build#16: 256 was too deep for ~144-192 padr calls per 2-pass run
                    static long long sNp[8] = {}, sKp[8] = {};
                    sNp[gs_shape] = N; sKp[gs_shape] = K;   // best effort label for slot 0's fold-over
                    for (int si2 = 0; si2 < 8; ++si2)
                        std::fprintf(stderr, "[GEMM_SPLIT] shape=%d N=%lld K=%lld calls=%ld deq=%.1f gemm=%.1f widen=%.1f ms\n",
                                     si2, (long long) sNp[si2], (long long) sKp[si2], gs_agg[si2].calls,
                                     gs_agg[si2].deq, gs_agg[si2].g, gs_agg[si2].wid);
                }
            }
            static bool padr_said = false;
            if (!padr_said) {
                padr_said = true;
                std::fprintf(stderr, "prefill gemm: STRATA_PF_PAD_ROCBLAS=%d (lda = K + %lld, ldb = %lld)\n",
                             padr_mode, (long long) padr, (long long) ldb);
            }
            return;
        }
    }
#endif
    if (ldx > 0 && ldx != K) {
#ifdef STRATA_USE_HIP
        // X arrived padded but the weight cannot be (an i-quant without a strided dequant, or the scratch is too
        // small for N x (K + 64)): the weight unpadded, X read at its stride - the same products, the same bits
        const int64_t rows = scratch_elems_ / K;
        bool ok = rows > 0;
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; ok && r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            ok = strata_pf_gemm_f16_ld(X, ldx, scratch_, K, Y + r0, T, n, K, ldy, beta, stream_);
        }
        if (ok) return;
#endif
#if defined(__HIPCC__)
        // pad-20261014 fail-closed fallback: on this build STRATA_USE_HIP is never defined, so the branch above is
        // gone and a padded X used to reach std::exit(1).  Do what that branch intended, with rocBLAS: dequantize
        // the weight UNPADDED into the scratch and run the ordinary GEMM of this function's tail, only with the
        // activation at its true row stride ldb = ldx (Gemm::f16 hardcodes ldb = K, which a padded X cannot use).
        // f16_io_ keeps the FP16-out + widen_rows_f16 pair that f16_inplace uses; without it the output is FP32 at
        // ldc = ldy, exactly like the tail below.  Same products, same bits.  beta == 0 only: that is the case
        // this path was written for.  It is slower than the weight stride, but STRATA_PF_PAD must never abort.
        if (beta == 0.0f && N * K <= scratch_elems_) {   // single shot; every in-scope shape fits
            if (ldy <= 0) ldy = N;
            strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
            const float one = 1.0f, zero = 0.0f;
            const cublasStatus_t st =
                f16_io_ ? cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K,
                                       &one, scratch_, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) ldx, &zero, Y,
                                       CUDA_R_16F, (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                        : cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K,
                                       &one, scratch_, CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) ldx, &zero, Y,
                                       CUDA_R_32F, (int) ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) {
                if (f16_io_ && !y_f16) widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, 0);
                static bool padx_said = false;
                if (!padx_said) {
                    padx_said = true;
                    std::fprintf(stderr, "prefill gemm: padded X (ldx = %lld, K = %lld): unpadded weight, strided X\n",
                                 (long long) ldx, (long long) K);
                }
                return;
            }
        }
#endif
        std::fprintf(stderr, "prefill gemm: a padded X (ldx %lld, K %lld) needs STRATA_PF_PAD's path\n",
                     (long long) ldx, (long long) K);
        std::exit(1);
    }
#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)
    if (beta == 0.0f && (ldx == 0 || ldx == K) && native_mmq(X, ggml_type, W_blocks, Y, T, N, K, ldy)) return;
#endif
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
                        f16(X, scratch_, Y + r0, T, n, K, ldy, beta, y_f16);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
        f16(X, scratch_, Y, T, N, K, ldy, beta, y_f16);
}

}  // namespace strata::prefill
