#!/usr/bin/env python3
"""patch_gemsol.py -- STATUS §286: pin a rocBLAS Tensile solution for a dense projection shape.

hipBLAS exposes no solution index (hipblasGemmAlgo_t has only HIPBLAS_GEMM_DEFAULT), so the pinned
path calls rocBLAS directly on a private handle bound to the same stream.  Fail-closed: any error
(handle create, invalid index, ...) permanently disables the path and the shipped hipBLAS call runs.

Env: STRATA_GEMM_SOL="NxK=sol[,...]"  e.g.  "2560x6144=20376".  Unset/empty = today's behaviour.
"""
import sys, io

PATH = sys.argv[1]
src = open(PATH).read()
orig = src

HELPER = r'''// gemm-sol-20261022 (STATUS §286): pin a rocBLAS Tensile solution for a dense projection shape.
// Off by default (STRATA_GEMM_SOL="NxK=sol,...").  hipBLAS's hipblasGemmAlgo_t has only
// HIPBLAS_GEMM_DEFAULT, so there is no way to select a solution through hipBLAS; this calls rocBLAS
// directly on a private handle bound to the same stream.  Fail-closed: on any error the pinned path
// disables itself for good and the shipped hipBLAS call is used instead.
static int pf_sol_lookup(long long N, long long K) {
    static const std::vector<std::pair<std::pair<long long, long long>, int>> tab = [] {
        std::vector<std::pair<std::pair<long long, long long>, int>> t;
        const char* e = std::getenv("STRATA_GEMM_SOL");
        if (e == nullptr) return t;
        const char* p = e;
        while (*p) {
            char* end = nullptr;
            const long long n = std::strtoll(p, &end, 10);
            if (end == p || *end != 'x') break;
            p = end + 1;
            const long long k = std::strtoll(p, &end, 10);
            if (end == p || *end != '=') break;
            p = end + 1;
            const long s = std::strtol(p, &end, 10);
            if (n > 0 && k > 0 && s > 0) t.push_back({{n, k}, (int) s});
            p = end;
            while (*p == ',' || *p == ' ') ++p;
        }
        return t;
    }();
    for (auto& q : tab) if (q.first.first == N && q.first.second == K) return q.second;
    return 0;
}

// Returns true only when the pinned-solution GEMM actually ran and succeeded.
static bool pf_gemm_sol(int sol, int N, int64_t T, int64_t K, int64_t lda, int64_t ldb, int64_t ldc,
                        const void* A, const void* B, void* C, void* stream) {
    static rocblas_handle rh = nullptr;
    static bool dead = false;
    if (dead) return false;
    if (rh == nullptr && rocblas_create_handle(&rh) != rocblas_status_success) { dead = true; return false; }
    if (rh == nullptr) { dead = true; return false; }
    (void) rocblas_set_stream(rh, (hipStream_t) stream);
    const float one = 1.0f, zero = 0.0f;
    const rocblas_status rs = rocblas_gemm_ex(
        rh, rocblas_operation_transpose, rocblas_operation_none, (rocblas_int) N, (rocblas_int) T,
        (rocblas_int) K, &one, A, rocblas_datatype_f16_r, (rocblas_int) lda, B, rocblas_datatype_f16_r,
        (rocblas_int) ldb, &zero, C, rocblas_datatype_f16_r, (rocblas_int) ldc, C, rocblas_datatype_f16_r,
        (rocblas_int) ldc, rocblas_datatype_f32_r, rocblas_gemm_algo_solution_index, (uint32_t) sol, 0);
    if (rs != rocblas_status_success) {
        static bool said = false;
        if (!said) { said = true; std::fprintf(stderr, "gemm-sol: solution %d rejected (status %d), falling back\n", sol, (int) rs); }
        dead = true;
        return false;
    }
    return true;
}

void Gemm::f16_inplace('''

INCA = '#include <cublas_v2.h>'
if src.count(INCA) != 1:
    print("ANCHOR0_FAILED count=%d" % src.count(INCA)); sys.exit(5)
# rocBLAS's headers use unqualified std:: internally, so they MUST be included at file scope,
# never inside namespace strata::prefill (that is what broke build #1).
src = src.replace(INCA, INCA + '\n#include <rocblas/rocblas.h>', 1)

A = "void Gemm::f16_inplace("
n = src.count(A)
if n != 1:
    print("ANCHOR1_FAILED count=%d" % n); sys.exit(2)
src = src.replace(A, HELPER, 1)

OLD = """    const float one = 1.0f, zero = 0.0f;
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, CUDA_R_16F,
                                        (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16 out");"""
NEW = """    const float one = 1.0f, zero = 0.0f;
    const int pf_sol = pf_sol_lookup((long long) N, (long long) K);   // gemm-sol-20261022
    if (!(pf_sol > 0 && pf_gemm_sol(pf_sol, (int) N, T, K, (int64_t) K, (int64_t) K,
                                    (int64_t) (y_f16 ? ldy : 2 * ldy), (const void*) W, (const void*) X,
                                    (void*) Y, (void*) stream_)))
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, W, CUDA_R_16F,
                                        (int) K, X, CUDA_R_16F, (int) K, &zero, Y, CUDA_R_16F, (int) (y_f16 ? ldy : 2 * ldy), CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16 out");"""
n = src.count(OLD)
if n != 1:
    print("ANCHOR2_FAILED count=%d" % n); sys.exit(3)
src = src.replace(OLD, NEW, 1)

if src == orig:
    print("NOP"); sys.exit(4)
open(PATH, "w").write(src)
print("PATCH_OK")
