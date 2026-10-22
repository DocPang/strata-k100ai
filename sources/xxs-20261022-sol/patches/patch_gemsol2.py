#!/usr/bin/env python3
"""patch_gemsol2.py (STATUS §287): extend the pinned-solution path to the PADDED (padr) call site.

The padded dense shapes (attn_qkv N=10240, attn_gate N=6144, attn_q N=12288 at K=2560) run through
Gemm::native's padr branch, not Gemm::f16_inplace, so gemm-sol needs a second hook there.
Same env (STRATA_GEMM_SOL), same private handle, same fail-closed fallback to hipBLAS.
"""
import sys
PATH = sys.argv[1]
src = open(PATH).read()

OLD = """            } else {
                ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &padr_one,
                                scratch_, CUDA_R_16F, (int) ldw, X, CUDA_R_16F, (int) ldb, &padr_zero, Y, CUDA_R_16F,
                                (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                   "cublasGemmEx f16 pad");"""
NEW = """            } else {
                const int pf_padr_sol = pf_sol_lookup((long long) N, (long long) K);   // gemm-sol-20261022
                if (!(pf_padr_sol > 0 &&
                      pf_gemm_sol(pf_padr_sol, (int) N, T, K, (int64_t) ldw, (int64_t) ldb,
                                  (int64_t) (y_f16 ? ldy : 2 * ldy), (const void*) scratch_, (const void*) X,
                                  (void*) Y, (void*) stream_)))
                ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &padr_one,
                                scratch_, CUDA_R_16F, (int) ldw, X, CUDA_R_16F, (int) ldb, &padr_zero, Y, CUDA_R_16F,
                                (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                   "cublasGemmEx f16 pad");"""
n = src.count(OLD)
if n != 1:
    print("ANCHOR_FAILED count=%d" % n); sys.exit(2)
src = src.replace(OLD, NEW, 1)
open(PATH, "w").write(src)
print("PATCH2_OK")
