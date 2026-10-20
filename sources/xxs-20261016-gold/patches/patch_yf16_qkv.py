#!/usr/bin/env python3
r"""wv2r-20261016 (15): STRATA_GEMM_Y_F16=qkv - kill widen_rows_f16 for the attn_qkv projection.

Idea
----
Gemm::native writes the rocBLAS FP16 result at ldc = 2*ldy (one f32 row's worth of halves, only the first n used)
and then widen_rows_f16 expands those n halves IN PLACE into n floats.  Measured: 405.4 ms per 30,602-token
prefill pass over 96 calls (wv2gsp-ins-gsp1017.log:273-275); the attn_qkv shape (N=10240, 42 calls) is ~51% of it.

If the GEMM instead writes a COMPACT FP16 image (ldc = ldy) and skips the widen, then every element keeps the SAME
linear index as the f32 layout had (row r, column c -> r*ldy + c).  A consumer that read Y[i] can read
__half2float(((const __half*)Y)[i]) and gets exactly the value widen would have stored - bit-identical by
construction, with no index arithmetic change at all.

Scope of this patch: m.qkv only (attn_qkv.weight, N = C = 10240).  Its sole consumer is gdn_conv (prefill.cpp:2746),
which is only ever called from the prefill path, and all four of its kernels address the buffer as qkv[t*C + c].

Gates
-----
STRATA_GEMM_Y_F16 = "qkv" (or "1" for every name this patch knows).  Unset/empty = every call is byte-for-byte
today's.  Both sides (the GEMM and the consumer) read the same helper, so they cannot disagree.

Usage: python3 patch_yf16_qkv.py [engine_tree]
"""
import hashlib, os, pathlib, shutil, sys

W = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/data/experiments/strata_prefill_reanalysis_20261011/kgupad-wv2-20261020")
GH = W / "source/include/strata/prefill/gemm.hpp"
GC = W / "source/src/prefill/gemm.cu"
KH = W / "source/include/strata/prefill/kernels.hpp"
KC = W / "source/src/prefill/kernels.cu"
PP = W / "source/src/prefill/prefill.cpp"

# ---------------------------------------------------------------- gemm.hpp
GH_OLD = r"""    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f, int64_t ldx = 0);"""
GH_NEW = r"""    /// `y_f16` (wv2r-20261016, STRATA_GEMM_Y_F16): write a COMPACT FP16 image (ldc = ldy) and skip
    /// widen_rows_f16, so the consumer reads __half2float(Y[i]) at the same index - bit-identical, because that
    /// is exactly the value widen would have stored.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f, int64_t ldx = 0, bool y_f16 = false);"""

# ---------------------------------------------------------------- gemm.cu
GC_SIG_OLD = r"""void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx) {"""
GC_SIG_NEW = r"""void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta, int64_t ldx, bool y_f16) {"""

GC_LT_OLD = r"""            if (padr_lt_on) {"""
GC_LT_NEW = r"""            if (padr_lt_on && !y_f16) {   // wv2r-20261016: the Lt path's lc is 2*ldy and its widen is unconditional"""

GC_PADR_OLD = r"""                                (int) (2 * ldy), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                   "cublasGemmEx f16 pad");
                if (gsplit) (void) hipEventRecord(gev[gs_slot][2], (hipStream_t) stream_);
                widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, 0);
            }"""
GC_PADR_NEW = r"""                                (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
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
            }"""

GC_F16IO_LDC_OLD = r"""                                       CUDA_R_16F, (int) (2 * ldy), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                        : cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K,"""
GC_F16IO_LDC_NEW = r"""                                       CUDA_R_16F, (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)
                        : cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K,"""

GC_F16IO_W_OLD = r"""                if (f16_io_) widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, 0);"""
GC_F16IO_W_NEW = r"""                if (f16_io_ && !y_f16) widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, 0);"""

# ---------------------------------------------------------------- kernels.hpp
KH_OLD = r"""void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream);"""
KH_NEW = r"""void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream,
              bool y_f16 = false);"""

# ---------------------------------------------------------------- kernels.cu
KC_TILED_OLD = r"""__global__ void gdn_conv_tiled_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                      const float* __restrict__ w, float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {
        const float x = qkv[t * C + c];"""
KC_TILED_NEW = r"""// wv2r-20261016: YF16 = the caller's GEMM wrote a COMPACT FP16 image (STRATA_GEMM_Y_F16).  __half2float of the
// half at the SAME index is exactly the f32 widen_rows_f16 would have stored, so YF16 is bit-identical.
template <bool YF16>
__global__ void gdn_conv_tiled_kernel(const float* __restrict__ hist, const float* __restrict__ qkv,
                                      const float* __restrict__ w, float* __restrict__ h, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int64_t t0 = (int64_t) blockIdx.y * CONV_TILE;
    if (t0 >= T) return;
    const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
    auto rd = [&](int64_t i) -> float {
        return YF16 ? __half2float(reinterpret_cast<const __half*>(qkv)[i]) : qkv[i];
    };
    auto input = [&](int64_t t) -> float { return t >= 0 ? rd(t * C + c) : hist[c * 3 + (int) (t + 3)]; };
    float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int64_t t = t0; t < t1; ++t) {
        const float x = rd(t * C + c);"""

KC_HIST_OLD = r"""__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int64_t t = T - 3 + k;
        v[k] = t >= 0 ? qkv[t * C + c] : hist[c * 3 + (int) (t + 3)];
    }"""
KC_HIST_NEW = r"""template <bool YF16>
__global__ void gdn_conv_hist_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int64_t T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int k = 0; k < 3; ++k) {
        const int64_t t = T - 3 + k;
        v[k] = t >= 0 ? (YF16 ? __half2float(reinterpret_cast<const __half*>(qkv)[t * C + c]) : qkv[t * C + c])
                      : hist[c * 3 + (int) (t + 3)];
    }"""

KC_WRAP_OLD = r"""void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {"""
KC_WRAP_NEW = r"""void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream,
              bool y_f16) {
    // wv2r-20261016: y_f16 = the caller's GEMM left a COMPACT FP16 image (STRATA_GEMM_Y_F16).  Only the two tiled
    // kernels carry the FP16 reader, so y_f16 forces that pair - the same expression per element, the same bits -
    // instead of ever letting a float read touch FP16 bytes.
    if (y_f16) {
        static bool said = false;
        if (!said) {
            said = true;
            std::fprintf(stderr, "strata: gdn_conv reads the compact FP16 qkv image (STRATA_GEMM_Y_F16)\n");
        }
        gdn_conv_tiled_kernel<true><<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                      (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
        gdn_conv_hist_kernel<true><<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
        gdn_l2_kernel<<<dim3(2 * HK, (unsigned) T), S, 0, (cudaStream_t) stream>>>(h, eps);
        check("gdn_conv");
        return;
    }
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {"""

KC_CALL_TILED_OLD = r"""            gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
            check("gdn_conv");
            return;
        }
        gdn_conv_tiled_kernel<<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
        gdn_conv_hist_kernel<<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);"""
KC_CALL_TILED_NEW = r"""            gdn_conv_hist_kernel<false><<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);
            check("gdn_conv");
            return;
        }
        gdn_conv_tiled_kernel<false><<<dim3(C / 128, (unsigned) ((T + CONV_TILE - 1) / CONV_TILE)), 128, 0,
                                (cudaStream_t) stream>>>(history, qkv, conv_w, h, T);
        gdn_conv_hist_kernel<false><<<C / 128, 128, 0, (cudaStream_t) stream>>>(history, qkv, T);"""

# ---------------------------------------------------------------- prefill.cpp
PP_HELPER_OLD = r"""bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0, int64_t ldx = 0) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy, 0.0f, ldx);
    return true;
}"""
PP_HELPER_NEW = r"""// wv2r-20261016: STRATA_GEMM_Y_F16 - comma-separated projection names ("qkv", or "1" for all) whose GEMM writes a
// COMPACT FP16 image and skips widen_rows_f16.  Both the GEMM (via native_proj) and the consumer read THIS helper,
// so the two sides cannot disagree.  Unset/empty = every call is byte-for-byte today's.
bool yf16_on(const char* name) {
    static const char* v = std::getenv("STRATA_GEMM_Y_F16");
    if (!v || !*v) return false;
    static const std::string spec(v);
    if (spec == "1") return true;
    const std::string want(name);
    size_t p = 0;
    while (p <= spec.size()) {
        size_t q = spec.find(',', p);
        if (q == std::string::npos) q = spec.size();
        if (q - p == want.size() && spec.compare(p, want.size(), want) == 0) return true;
        if (q >= spec.size()) break;
        p = q + 1;
    }
    return false;
}
bool native_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y, int64_t T, const std::string& name,
                 std::string& err, int64_t ldy = 0, int64_t ldx = 0, bool y_f16 = false) {
    if (!w->native_data) { err = "prefill: " + name + " has no native GGUF blocks (run with --native)"; return false; }
    gm.native(X, w->native_type, w->native_data, Y, T, w->ne1, w->ne0, ldy, 0.0f, ldx, y_f16);
    return true;
}"""

PP_CALL_OLD = r"""                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;"""
PP_CALL_NEW = r"""                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err, 0, 0,
                                     yf16_on("qkv"))) return false;"""

PP_CONV_OLD = r"""                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);"""
PP_CONV_NEW = r"""                    gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs, yf16_on("qkv"));"""

PLAN = [
    (GH, "gemm.hpp native decl",   [(GH_OLD, GH_NEW, 1)]),
    (GC, "gemm.cu",                [(GC_SIG_OLD, GC_SIG_NEW, 1), (GC_LT_OLD, GC_LT_NEW, 1),
                                    (GC_PADR_OLD, GC_PADR_NEW, 1), (GC_F16IO_LDC_OLD, GC_F16IO_LDC_NEW, 1),
                                    (GC_F16IO_W_OLD, GC_F16IO_W_NEW, 1)]),
    (KH, "kernels.hpp gdn_conv",   [(KH_OLD, KH_NEW, 1)]),
    (KC, "kernels.cu",             [(KC_TILED_OLD, KC_TILED_NEW, 1), (KC_HIST_OLD, KC_HIST_NEW, 1),
                                    (KC_WRAP_OLD, KC_WRAP_NEW, 1), (KC_CALL_TILED_OLD, KC_CALL_TILED_NEW, 1)]),
    (PP, "prefill.cpp",            [(PP_HELPER_OLD, PP_HELPER_NEW, 1), (PP_CALL_OLD, PP_CALL_NEW, 1),
                                    (PP_CONV_OLD, PP_CONV_NEW, 1)]),
]

for path, label, edits in PLAN:
    if not path.exists():
        print("FAIL: %s not found (%s)" % (path, label)); sys.exit(2)

# ---- pass 1: verify every anchor, write nothing
texts = {}
for path, label, edits in PLAN:
    s = path.read_text()
    texts[path] = s
    if "STRATA_GEMM_Y_F16" in s:
        print("OK: %s already patched" % label); continue
    for old, _new, cnt in edits:
        got = s.count(old)
        if got != cnt:
            print("FAIL: %s anchor %r occurs %d times, expected %d - refusing to write"
                  % (label, old.splitlines()[0][:60], got, cnt)); sys.exit(2)
print("ALL_ANCHORS_OK")

# ---- pass 2: apply
for path, label, edits in PLAN:
    s = texts[path]
    if "STRATA_GEMM_Y_F16" in s:
        continue
    bak = path.with_suffix(path.suffix + ".pre_yf16")
    if not bak.exists():
        shutil.copy2(path, bak)
    before = hashlib.sha256(s.encode()).hexdigest()
    out = s
    for old, new, _cnt in edits:
        out = out.replace(old, new, 1)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(out)
    os.replace(tmp, path)
    print("PATCH %s: %d -> %d B  sha %s -> %s  (backup %s)"
          % (path.name, len(s), len(out), before[:16], hashlib.sha256(out.encode()).hexdigest()[:16], bak.name))
