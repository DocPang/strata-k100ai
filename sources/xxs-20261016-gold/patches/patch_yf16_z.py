#!/usr/bin/env python3
r"""wv2r-20261016 (21): STRATA_GEMM_Y_F16=z - the attn_gate GEMM (N=6144, K=2560) writes a compact FP16 image too.

m.z's GEMM does take the padr branch (N>=1024 && K==2560), so only the consumer side needs the FP16 reader.
Three kernels read z, across gdn_recurrence_variant's branches:
  gdn_out_norm_loop_kernel<WY>   the default path (variant 1; STRATA_GDN_NOY=1 -> WY=false)
  gdn_out_norm_kernel            variant 3 / the column-split tail
  gdn_rec_kernel                 serial / variant 2
All three address z with the f32 layout's index (row * S + col, S = 256), so the compact f16 image keeps the same
index and __half2float at that index is exactly the value widen_rows_f16 would have stored.

Usage: python3 patch_yf16_z.py [engine_tree]
"""
import hashlib, os, pathlib, re as _re, shutil, sys

W = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/data/experiments/strata_prefill_reanalysis_20261011/kgupad-wv2-20261020")
KH = W / "source/include/strata/prefill/kernels.hpp"
KC = W / "source/src/prefill/kernels.cu"
PP = W / "source/src/prefill/prefill.cpp"


def _anchor(text):
    lines = [l.strip() for l in text.splitlines()]
    return _re.compile(r"\s*".join(_re.escape(l) for l in lines))


KH_EDITS = [
 (r"""void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream,
                    int64_t ld16 = 0);   // ld16: y16's row stride (0 = 6144; S23 STRATA_PF_PAD pads it)""",
  r"""void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream,
                    int64_t ld16 = 0,   // ld16: y16's row stride (0 = 6144; S23 STRATA_PF_PAD pads it)
                    bool z_f16 = false);   // wv2r-20261016: z is a compact FP16 image (STRATA_GEMM_Y_F16)""", 1),
]

KC_EDITS = [
 # ---- gdn_out_norm_loop_kernel
 (r"""__global__ void __launch_bounds__(S) gdn_out_norm_loop_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                              float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                              int64_t rows, int64_t ld16) {""",
  r"""__global__ void __launch_bounds__(S) gdn_out_norm_loop_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                              float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                              int64_t rows, int64_t ld16, bool z_f16) {""", 1),
 (r"""        const float bp = oc * rsqrtf(ss / (float) S + eps) * gamma[col], sg = sigm(z[at]);""",
  r"""        const float zv = z_f16 ? __half2float(reinterpret_cast<const __half*>(z)[at]) : z[at];
        const float bp = oc * rsqrtf(ss / (float) S + eps) * gamma[col], sg = sigm(zv);""", 1),
 # ---- gdn_out_norm_kernel
 (r"""__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                         int64_t ld16) {""",
  r"""__global__ void __launch_bounds__(S) gdn_out_norm_kernel(const float* __restrict__ z, const float* __restrict__ gamma,
                                                         float eps, float* __restrict__ y, uint16_t* __restrict__ y16,
                                                         int64_t ld16, bool z_f16) {""", 1),
 (r"""    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(z[t * HV * S + head * S + col]);""",
  r"""    const float zv = z_f16 ? __half2float(reinterpret_cast<const __half*>(z)[t * HV * S + head * S + col])
                           : z[t * HV * S + head * S + col];
    const float v = oc * rsqrtf(ss / (float) S + eps) * gamma[col] * sigm(zv);""", 1),
 # ---- gdn_rec_kernel
 (r"""__global__ void __launch_bounds__(S * RG) gdn_rec_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                         const float* __restrict__ gate,
                                                         const float* __restrict__ beta, const float* __restrict__ z,
                                                         const float* __restrict__ gamma, float eps,
                                                         float* __restrict__ y, uint16_t* __restrict__ y16, int64_t T,
                                                         int64_t ld16) {""",
  r"""__global__ void __launch_bounds__(S * RG) gdn_rec_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                         const float* __restrict__ gate,
                                                         const float* __restrict__ beta, const float* __restrict__ z,
                                                         const float* __restrict__ gamma, float eps,
                                                         float* __restrict__ y, uint16_t* __restrict__ y16, int64_t T,
                                                         int64_t ld16, bool z_f16) {""", 1),
 (r"""            const float v = oc * rsqrtf(ss / (float) S + eps) * g_col * sigm(z[t * HV * S + head * S + col]);""",
  r"""            const float zv = z_f16 ? __half2float(reinterpret_cast<const __half*>(z)[t * HV * S + head * S + col])
                                   : z[t * HV * S + head * S + col];
            const float v = oc * rsqrtf(ss / (float) S + eps) * g_col * sigm(zv);""", 1),
 # ---- the variant entry point
 (r"""void gdn_recurrence_variant(int variant, float* state, const float* h, const float* gate, const float* beta,
                            const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T,
                            void* stream, int64_t ld16) {""",
  r"""void gdn_recurrence_variant(int variant, float* state, const float* h, const float* gate, const float* beta,
                            const float* z, const float* gamma, float eps, float* y, uint16_t* y16, int64_t T,
                            void* stream, int64_t ld16, bool z_f16) {""", 1),
 # ---- the 5 call sites that read z
 (r"""            gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16);""",
  r"""            gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16, z_f16);""", 2),
 (r"""            gdn_out_norm_loop_kernel<true><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                z, gamma, eps, y, y16, T * HV, ld16);""",
  r"""            gdn_out_norm_loop_kernel<true><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                z, gamma, eps, y, y16, T * HV, ld16, z_f16);""", 2),
 (r"""                gdn_out_norm_loop_kernel<false><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                    z, gamma, eps, y, y16, T * HV, ld16);""",
  r"""                gdn_out_norm_loop_kernel<false><<<(unsigned) std::min<int64_t>(T * HV, 4096), S, 0, (cudaStream_t) stream>>>(
                    z, gamma, eps, y, y16, T * HV, ld16, z_f16);""", 1),
 (r"""        gdn_rec_kernel<<<HV, dim3(S, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, z, gamma, eps, y, y16, T, ld16);""",
  r"""        gdn_rec_kernel<<<HV, dim3(S, RG), 0, (cudaStream_t) stream>>>(state, h, gate, beta, z, gamma, eps, y, y16, T, ld16, z_f16);""", 1),
 (r"""        gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16);
    }
    check("gdn_recurrence");""",
  r"""        gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, (cudaStream_t) stream>>>(z, gamma, eps, y, y16, ld16, z_f16);
    }
    check("gdn_recurrence");""", 1),
 # ---- the public wrapper
 (r"""void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream, int64_t ld16) {""",
  r"""void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream, int64_t ld16,
                    bool z_f16) {""", 1),
 (r"""    gdn_recurrence_variant(head && !serial ? 1 : 0, state, h, gate, beta, z, gamma, eps, y, y16, T, stream, ld16);""",
  r"""    gdn_recurrence_variant(head && !serial ? 1 : 0, state, h, gate, beta, z, gamma, eps, y, y16, T, stream, ld16, z_f16);""", 1),
]

PP_EDITS = [
 (r"""                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;""",
  r"""                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err, 0, 0,
                                     yf16_on("z"))) return false;""", 1),
 (r"""                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs,
                                   ld_y);""",
  r"""                    gdn_recurrence(state, m.hbuf, m.gate, m.beta, m.z, (const float*) wnm->data, EPS, m.y, m.y_h, T, m.cs,
                                   ld_y, yf16_on("z"));""", 1),
]

PLAN = [(KH, "kernels.hpp", KH_EDITS), (KC, "kernels.cu", KC_EDITS), (PP, "prefill.cpp", PP_EDITS)]
for path, label, edits in PLAN:
    if not path.exists():
        print("FAIL: %s not found (%s)" % (path, label)); sys.exit(2)

texts = {}
for path, label, edits in PLAN:
    s = path.read_text()
    texts[path] = s
    if "z_f16" in s:
        print("OK: %s already patched" % label); continue
    for old, _new, cnt in edits:
        got = len(_anchor(old).findall(s))
        if got != cnt:
            print("FAIL: %s anchor %r occurs %d times, expected %d - refusing to write"
                  % (label, old.splitlines()[0].strip()[:70], got, cnt)); sys.exit(2)
print("ALL_ANCHORS_OK")

for path, label, edits in PLAN:
    s = texts[path]
    if "z_f16" in s:
        continue
    bak = path.with_suffix(path.suffix + ".pre_yf16z")
    if not bak.exists():
        shutil.copy2(path, bak)
    before = hashlib.sha256(s.encode()).hexdigest()
    out = s
    for old, new, _cnt in edits:
        out = _anchor(old).sub(lambda m: new, out, count=_cnt)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(out)
    os.replace(tmp, path)
    print("PATCH %s: %d -> %d B  sha %s -> %s  (backup %s)"
          % (path.name, len(s), len(out), before[:16], hashlib.sha256(out.encode()).hexdigest()[:16], bak.name))
