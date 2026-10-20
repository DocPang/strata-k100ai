#!/usr/bin/env python3
r"""wv2r-20261016 (20): teach Gemm::f16 / Gemm::f16_inplace about y_f16.

Bug found by the h5yf16c batch (card4 yf arm: "non-finite first logits: 248320"): the padr branch's scope is
N >= 1024 && (K == 2560 || (K == 6144 && x_padded)) (gemm.cu:1101), so the shared-expert projections
(ffn_gate/up_shexp N=640, ffn_down_shexp K=640) never take it.  A call with ldx == 0 falls through the
`if (ldx > 0 && ldx != K)` block and ends at `f16(X, scratch_, Y, T, N, K, ldy, beta)` (gemm.cu:1346), and with
f16_io_ set that becomes `Gemm::f16_inplace` - whose own widen_rows_f16 was still unconditional.  The consumer
then read the FP16-at-2*ldy bytes as a compact image -> NaNs.

This patch threads y_f16 through f16() -> f16_inplace() the same way patch_yf16_qkv.py threaded it through the
padr and padded-X branches.

Usage: python3 patch_yf16_fix.py [engine_tree]
"""
import hashlib, os, pathlib, re as _re, shutil, sys

W = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/data/experiments/strata_prefill_reanalysis_20261011/kgupad-wv2-20261020")
GH = W / "source/include/strata/prefill/gemm.hpp"
GC = W / "source/src/prefill/gemm.cu"


def _anchor(text):
    lines = [l.strip() for l in text.splitlines()]
    return _re.compile(r"\s*".join(_re.escape(l) for l in lines))


GH_EDITS = [
 (r"""void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);""",
  r"""void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f, bool y_f16 = false);""", 1),
 (r"""void f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy);""",
  r"""void f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                       bool y_f16 = false);""", 1),
]

GC_EDITS = [
 # f16(): signature + the f16_io_ hand-off
 (r"""void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {""",
  r"""void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta, bool y_f16) {""", 1),
 (r"""    if (f16_io_ && beta == 0.0f) { f16_inplace(X, W, Y, T, N, K, ldy); return; }""",
  r"""    if (f16_io_ && beta == 0.0f) { f16_inplace(X, W, Y, T, N, K, ldy, y_f16); return; }""", 1),
 # f16_inplace(): signature
 (r"""void Gemm::f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy) {""",
  r"""void Gemm::f16_inplace(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                        bool y_f16) {""", 1),
 # f32out would leave FP32 in Y, which an FP16 reader cannot use: keep it off when y_f16
 (r"""    if (f32out) {
        const float one = 1.0f, zero = 0.0f;
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, CUDA_R_16F,""",
  r"""    if (f32out && !y_f16) {   // wv2r-20261016: FP32 in Y is not a compact FP16 image
        const float one = 1.0f, zero = 0.0f;
        ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, CUDA_R_16F,""", 1),
 # the compact ldc
 (r"""                    (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (2 * ldy), CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16 out");""",
  r"""                    (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16 out");""", 1),
 # the widen
 (r"""    static const bool dbg_nan = std::getenv("STRATA_DBG_NAN") != nullptr;
    widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, dbg_nan ? 1 : 0);""",
  r"""    static const bool dbg_nan = std::getenv("STRATA_DBG_NAN") != nullptr;
    if (y_f16) {
        static bool yf16_said = false;
        if (!yf16_said) {
            yf16_said = true;
            std::fprintf(stderr, "prefill gemm: f16_inplace compact FP16 image (STRATA_GEMM_Y_F16), widen skipped\n");
        }
    } else {
        widen_rows_f16<<<(unsigned) T, 256, 0, (cudaStream_t) stream_>>>(Y, N, ldy, dbg_nan ? 1 : 0);
    }""", 1),
 # native()'s two f16() tail calls
 (r"""            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);""",
  r"""            f16(X, scratch_, Y + r0, T, n, K, ldy, beta, y_f16);""", 1),
 (r"""    f16(X, scratch_, Y, T, N, K, ldy, beta);
}""",
  r"""    f16(X, scratch_, Y, T, N, K, ldy, beta, y_f16);
}""", 1),
]

PLAN = [(GH, "gemm.hpp", GH_EDITS), (GC, "gemm.cu", GC_EDITS)]
for path, label, edits in PLAN:
    if not path.exists():
        print("FAIL: %s not found (%s)" % (path, label)); sys.exit(2)

texts = {}
for path, label, edits in PLAN:
    s = path.read_text()
    texts[path] = s
    if "f16_inplace compact FP16 image" in s:
        print("OK: %s already patched" % label); continue
    for old, _new, cnt in edits:
        got = len(_anchor(old).findall(s))
        if got != cnt:
            print("FAIL: %s anchor %r occurs %d times, expected %d - refusing to write"
                  % (label, old.splitlines()[0].strip()[:70], got, cnt)); sys.exit(2)
print("ALL_ANCHORS_OK")

for path, label, edits in PLAN:
    s = texts[path]
    if "f16_inplace compact FP16 image" in s:
        continue
    bak = path.with_suffix(path.suffix + ".pre_yf16fix")
    if not bak.exists():
        shutil.copy2(path, bak)
    before = hashlib.sha256(s.encode()).hexdigest()
    out = s
    for old, new, _cnt in edits:
        out = _anchor(old).sub(lambda m: new, out, count=1)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(out)
    os.replace(tmp, path)
    print("PATCH %s: %d -> %d B  sha %s -> %s  (backup %s)"
          % (path.name, len(s), len(out), before[:16], hashlib.sha256(out.encode()).hexdigest()[:16], bak.name))
