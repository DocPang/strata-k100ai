# k100ai / gfx928 (wave64 compat layer) platform shim.
#
# 1) Upstream gates hipBLASLt and rocBLAS on STRATA_ENABLE_HIP (the wave32 RDNA backend), so the gfx906 wave64 build
#    gets neither and falls back to the native ggml GEMMs.  The sources already branch on the two macros, so enabling
#    them here needs no source change.
# 2) The hipBLASLt path in gemm.cu calls strata_pf_hcdown_exact_bf16, defined only in wmma_gemm.cu (RDNA3 builds).
#    src/prefill/k100ai_stubs.cu supplies the no-gfx11 answer (false), the same one that file own stub gives.
find_library(K100AI_HIPBLASLT NAMES hipblaslt HINTS /opt/dtk/lib REQUIRED)
find_library(K100AI_ROCBLAS  NAMES rocblas  HINTS /opt/dtk/lib REQUIRED)
target_link_libraries(strata_prefill PRIVATE ${K100AI_HIPBLASLT} ${K100AI_ROCBLAS})
target_compile_definitions(strata_prefill PRIVATE STRATA_HIPBLASLT_AVAILABLE=1 STRATA_ROCBLAS_AVAILABLE=1)
target_sources(strata_prefill PRIVATE src/prefill/k100ai_stubs.cu)
message(STATUS "k100ai: hipBLASLt + rocBLAS enabled for the gfx906 wave64 backend")
