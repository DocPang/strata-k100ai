#!/usr/bin/env python3
r"""wv2r-20261016 (17): extend STRATA_GEMM_Y_F16 to m.Qf, m.shared, m.sgate, m.sup.

Same mechanism as patch_yf16_qkv.py: the GEMM writes a COMPACT FP16 image (ldc = ldy) and skips widen_rows_f16, so
each consumer reads __half2float(((const __half*)Y)[i]) at the SAME index - bit-identical, because that is the value
widen would have stored.  Every one of these consumers addresses Y with the f32 layout's index:
  split_q_kernel      qf[t*24*512 + h*512 + d]        (row stride 12288 = N)
  gate_attn_kernel    qf[t*24*512 + h*512 + 256 + d]  (same row stride)
  swiglu_pair_kernel  g[i], u[i]                      (linear, row stride 640 = N)
  moe_combine*        shared[i]                       (linear, row stride N)
so no index arithmetic changes anywhere.

NOT in this patch: m.z (gdn_recurrence's z - three reader kernels across the variant branches) and m.bo (a half-way
scratch shared with moe_combine's f32 writer).

Gates: STRATA_GEMM_Y_F16 is a comma-separated list; this patch adds the names "qf", "sd", "sg", "su" (or "1" = all).
Unset/empty = byte-for-byte today's behaviour.

Usage: python3 patch_yf16_more.py [engine_tree]
"""
import hashlib, os, pathlib, shutil, sys

import re as _re

def _anchor(text):
    """Match `text` with every line's leading/trailing whitespace collapsed, so a multi-line anchor does not depend
    on the exact indentation the file happens to use."""
    lines = [l.strip() for l in text.splitlines()]
    return _re.compile(r"\s*".join(_re.escape(l) for l in lines))


W = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/data/experiments/strata_prefill_reanalysis_20261011/kgupad-wv2-20261020")
KH = W / "source/include/strata/prefill/kernels.hpp"
KC = W / "source/src/prefill/kernels.cu"
PP = W / "source/src/prefill/prefill.cpp"

# ---------------------------------------------------------------- kernels.hpp
KH_EDITS = [
 (r"""void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream);""",
  r"""void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream,
                 bool g_f16 = false, bool u_f16 = false);""", 1),
 (r"""void moe_combine(const uint16_t* D, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream);""",
  r"""void moe_combine(const uint16_t* D, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream, bool shared_f16 = false);""", 1),
 (r"""void moe_combine_peer(const uint16_t* D, const int32_t* slot, const float* w, const float* shared, const float* sg,
                      const float* peer, int64_t rows_local, float* bo, int64_t T, void* stream);""",
  r"""void moe_combine_peer(const uint16_t* D, const int32_t* slot, const float* w, const float* shared, const float* sg,
                      const float* peer, int64_t rows_local, float* bo, int64_t T, void* stream,
                      bool shared_f16 = false);""", 1),
 (r"""void moe_combine_peer16(const uint16_t* D, const int32_t* slot, const float* w, const float* shared, const float* sg,
                        const uint16_t* peer16, int64_t rows_local, float* bo, int64_t T, void* stream);""",
  r"""void moe_combine_peer16(const uint16_t* D, const int32_t* slot, const float* w, const float* shared, const float* sg,
                        const uint16_t* peer16, int64_t rows_local, float* bo, int64_t T, void* stream,
                        bool shared_f16 = false);""", 1),
 (r"""void split_q(const float* q_full, float* q, int64_t T, void* stream);""",
  r"""void split_q(const float* q_full, float* q, int64_t T, void* stream, bool qf_f16 = false);""", 1),
 (r"""void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream, int64_t ld16 = 0);""",
  r"""void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream, int64_t ld16 = 0,
               bool qf_f16 = false);""", 1),
]

# ---------------------------------------------------------------- kernels.cu
KC_EDITS = [
 # ---- swiglu_pair
 (r"""__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g[i];
    h16[i] = hf_sat(a / (1.0f + __expf(-a)) * u[i]);
}""",
  r"""// wv2r-20261016: g_f16 / u_f16 - the caller's GEMM left a COMPACT FP16 image (STRATA_GEMM_Y_F16); __half2float at
// the same index is exactly the f32 widen_rows_f16 would have stored, so the output bits are unchanged.
__global__ void swiglu_pair_kernel(const float* __restrict__ g, const float* __restrict__ u, uint16_t* __restrict__ h16,
                                   int64_t n, bool g_f16, bool u_f16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 640) return;
    const float a = g_f16 ? __half2float(reinterpret_cast<const __half*>(g)[i]) : g[i];
    const float b = u_f16 ? __half2float(reinterpret_cast<const __half*>(u)[i]) : u[i];
    h16[i] = hf_sat(a / (1.0f + __expf(-a)) * b);
}""", 1),
 (r"""void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n);
    check("swiglu_pair");
}""",
  r"""void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream, bool g_f16, bool u_f16) {
    swiglu_pair_kernel<<<blocks_for(n * 640), 256, 0, (cudaStream_t) stream>>>(g, u, h16, n, g_f16, u_f16);
    check("swiglu_pair");
}""", 1),
 # ---- moe_combine (scalar)
 (r"""                                   const float* __restrict__ sg, float* __restrict__ bo, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) s = fmaf(w[t * 10 + k], f16v(Dm[(int64_t) slot[t * 10 + k] * N + d]), s);
    bo[i] = s + shared[i] * sigm(sg[t]);
}""",
  r"""                                   const float* __restrict__ sg, float* __restrict__ bo, int64_t T, bool sh_f16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * N) return;
    const int64_t t = i / N, d = i % N;
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 10; ++k) s = fmaf(w[t * 10 + k], f16v(Dm[(int64_t) slot[t * 10 + k] * N + d]), s);
    const float sh = sh_f16 ? __half2float(reinterpret_cast<const __half*>(shared)[i]) : shared[i];
    bo[i] = s + sh * sigm(sg[t]);
}""", 1),
 # ---- moe_combine (peer)
 (r"""                                          const float* __restrict__ sg, const float* __restrict__ peer, int64_t rows_local,
                                          float* __restrict__ bo, int64_t T) {""",
  r"""                                          const float* __restrict__ sg, const float* __restrict__ peer, int64_t rows_local,
                                          float* __restrict__ bo, int64_t T, bool sh_f16) {""", 1),
 (r"""    bo[i] = (s + peer[i]) + shared[i] * sigm(sg[t]);
}""",
  r"""    bo[i] = (s + peer[i]) + (sh_f16 ? __half2float(reinterpret_cast<const __half*>(shared)[i]) : shared[i]) * sigm(sg[t]);
}""", 1),
 # ---- moe_combine (peer16)
 (r"""                                          const float* __restrict__ sg, const uint16_t* __restrict__ peer16,
                                          int64_t rows_local, float* __restrict__ bo, int64_t T) {""",
  r"""                                          const float* __restrict__ sg, const uint16_t* __restrict__ peer16,
                                          int64_t rows_local, float* __restrict__ bo, int64_t T, bool sh_f16) {""", 1),
 (r"""    bo[i] = (s + __half2float(__ushort_as_half(peer16[i]))) + shared[i] * sigm(sg[t]);
}""",
  r"""    bo[i] = (s + __half2float(__ushort_as_half(peer16[i]))) +
            (sh_f16 ? __half2float(reinterpret_cast<const __half*>(shared)[i]) : shared[i]) * sigm(sg[t]);
}""", 1),
 # ---- moe_combine wrapper (fail-closed away from the float4 kernel)
 (r"""void moe_combine(const uint16_t* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {""",
  r"""void moe_combine(const uint16_t* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream, bool shared_f16) {""", 1),
 (r"""    if (comb_vec &&
        ((reinterpret_cast<uintptr_t>(Dm) | reinterpret_cast<uintptr_t>(shared) | reinterpret_cast<uintptr_t>(bo)) & 15) == 0) {""",
  r"""    if (comb_vec && !shared_f16 &&   // wv2r-20261016: the float4 kernel has no FP16 reader
        ((reinterpret_cast<uintptr_t>(Dm) | reinterpret_cast<uintptr_t>(shared) | reinterpret_cast<uintptr_t>(bo)) & 15) == 0) {""", 1),
 (r"""    moe_combine_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, bo, T);
    check("moe_combine");
}""",
  r"""    moe_combine_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, bo, T, shared_f16);
    check("moe_combine");
}""", 1),
 # ---- moe_combine_peer16 wrapper
 (r"""                        const uint16_t* peer16, int64_t rows_local, float* bo, int64_t T, void* stream) {
    moe_combine_peer16_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, peer16,
                                                                                      rows_local, bo, T);""",
  r"""                        const uint16_t* peer16, int64_t rows_local, float* bo, int64_t T, void* stream, bool shared_f16) {
    moe_combine_peer16_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, peer16,
                                                                                      rows_local, bo, T, shared_f16);""", 1),
 # ---- moe_combine_peer wrapper
 (r"""                      const float* peer, int64_t rows_local, float* bo, int64_t T, void* stream) {
    moe_combine_peer_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, peer, rows_local,
                                                                                    bo, T);""",
  r"""                      const float* peer, int64_t rows_local, float* bo, int64_t T, void* stream, bool shared_f16) {
    moe_combine_peer_kernel<<<blocks_for(T * N), 256, 0, (cudaStream_t) stream>>>(Dm, slot, w, shared, sg, peer, rows_local,
                                                                                    bo, T, shared_f16);""", 1),
 # ---- split_q
 (r"""__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    q[i] = qf[t * 24 * 512 + h * 512 + d];
}""",
  r"""__global__ void split_q_kernel(const float* __restrict__ qf, float* __restrict__ q, int64_t T, bool qf_f16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    const int64_t at = t * 24 * 512 + h * 512 + d;
    q[i] = qf_f16 ? __half2float(reinterpret_cast<const __half*>(qf)[at]) : qf[at];
}""", 1),
 (r"""void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T);
    check("split_q");
}""",
  r"""void split_q(const float* q_full, float* q, int64_t T, void* stream, bool qf_f16) {
    split_q_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(q_full, q, T, qf_f16);
    check("split_q");
}""", 1),
 # ---- gate_attn
 (r"""__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T, int64_t ld16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    o16[t * ld16 + h * 256 + d] = hf(a[i] * (1.0f / (1.0f + expf(-qf[t * 24 * 512 + h * 512 + 256 + d]))));
}""",
  r"""__global__ void gate_attn_kernel(const float* __restrict__ a, const float* __restrict__ qf, uint16_t* __restrict__ o16,
                                 int64_t T, int64_t ld16, bool qf_f16) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 24 * 256) return;
    const int64_t t = i / (24 * 256), h = (i / 256) % 24, d = i % 256;
    const int64_t at = t * 24 * 512 + h * 512 + 256 + d;
    const float g = qf_f16 ? __half2float(reinterpret_cast<const __half*>(qf)[at]) : qf[at];
    o16[t * ld16 + h * 256 + d] = hf(a[i] * (1.0f / (1.0f + expf(-g))));
}""", 1),
 (r"""void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream, int64_t ld16) {
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T,
                                                                                ld16 > 0 ? ld16 : 24 * 256);""",
  r"""void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream, int64_t ld16,
                bool qf_f16) {
    gate_attn_kernel<<<blocks_for(T * 24 * 256), 256, 0, (cudaStream_t) stream>>>(attn, q_full, out16, T,
                                                                                ld16 > 0 ? ld16 : 24 * 256, qf_f16);""", 1),
]

# ---------------------------------------------------------------- prefill.cpp
PP_EDITS = [
 (r"""    if (spec == "1") return true;""",
  r"""    if (spec == "1") {
        static bool said = false;
        if (!said) { said = true; std::fprintf(stderr, "strata: STRATA_GEMM_Y_F16=1 active (every listed projection writes a compact FP16 image)\n"); }
        return true;
    }""", 1),
 (r"""        if (q - p == want.size() && spec.compare(p, want.size(), want) == 0) return true;""",
  r"""        if (q - p == want.size() && spec.compare(p, want.size(), want) == 0) {
            static bool said = false;
            if (!said) { said = true; std::fprintf(stderr, "strata: STRATA_GEMM_Y_F16=%s active (compact FP16 GEMM images)\n", spec.c_str()); }
            return true;
        }""", 1),
 (r"""                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err)) return false;""",
  r"""                    if (!native_proj(m.gemm, wq, m.mixed_h, m.Qf, T, v.name("attn_q.weight"), err, 0, 0,
                                     yf16_on("qf"))) return false;""", 1),
 (r"""                        if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err)) return false;
                        if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err)) return false;""",
  r"""                        if (!native_proj(m.gemm, wsg, m.mixed_h, m.sgate, T, v.name("ffn_gate_shexp.weight"), err, 0, 0,
                                         yf16_on("sg"))) return false;
                        if (!native_proj(m.gemm, wsu, m.mixed_h, m.sup, T, v.name("ffn_up_shexp.weight"), err, 0, 0,
                                         yf16_on("su"))) return false;""", 1),
 (r"""                        if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err)) return false;""",
  r"""                        if (!native_proj(m.gemm, wsd, m.sh_h, m.shared, T, v.name("ffn_down_shexp.weight"), err, 0, 0,
                                         yf16_on("sd"))) return false;""", 1),
 (r"""                    split_q(m.Qf, m.q, T, m.cs);""",
  r"""                    split_q(m.Qf, m.q, T, m.cs, yf16_on("qf"));""", 1),
 (r"""                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs, ld_a);""",
  r"""                    gate_attn(m.attn, m.Qf, m.attn_h, T, m.cs, ld_a, yf16_on("qf"));""", 1),
 (r"""                        swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs);""",
  r"""                        swiglu_pair(m.sgate, m.sup, m.sh_h, T, m.cs, yf16_on("sg"), yf16_on("su"));""", 1),
 (r"""                        moe_combine_peer16(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.pp->host_sum16, m.pp->back_at, m.bo, T, m.cs);""",
  r"""                        moe_combine_peer16(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.pp->host_sum16, m.pp->back_at, m.bo, T, m.cs,
                                           yf16_on("sd"));""", 1),
 (r"""                        moe_combine_peer(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.pp->host_sum, m.pp->back_at, m.bo, T, m.cs);""",
  r"""                        moe_combine_peer(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.pp->host_sum, m.pp->back_at, m.bo, T, m.cs,
                                         yf16_on("sd"));""", 1),
 (r"""                        moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs);""",
  r"""                        moe_combine(m.Dm, m.slot_dev, m.w, m.shared, m.sg, m.bo, T, m.cs, yf16_on("sd"));""", 1),
]

PLAN = [(KH, "kernels.hpp", KH_EDITS), (KC, "kernels.cu", KC_EDITS), (PP, "prefill.cpp", PP_EDITS)]

for path, label, edits in PLAN:
    if not path.exists():
        print("FAIL: %s not found (%s)" % (path, label)); sys.exit(2)

texts = {}
for path, label, edits in PLAN:
    s = path.read_text()
    texts[path] = s
    if "compact FP16 GEMM images" in s or "compact FP16 image)" in s:
        print("OK: %s already patched" % label); continue
    for old, _new, cnt in edits:
        got = len(_anchor(old).findall(s))
        if got != cnt:
            print("FAIL: %s anchor %r occurs %d times, expected %d - refusing to write"
                  % (label, old.splitlines()[0].strip()[:70], got, cnt)); sys.exit(2)
print("ALL_ANCHORS_OK")

for path, label, edits in PLAN:
    s = texts[path]
    if "compact FP16 GEMM images" in s or "compact FP16 image)" in s:
        continue
    bak = path.with_suffix(path.suffix + ".pre_yf16m")
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
