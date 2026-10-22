#!/usr/bin/env python3
"""Add librocblas to the strata_prefill link line: gemm.cu now calls rocblas_gemm_ex directly
(hipBLAS does not expose a solution index).  Fail-closed: anchors must match exactly once."""
import sys
PATH = sys.argv[1]
src = open(PATH).read()
A = "  target_link_libraries(strata_prefill PUBLIC strata_engine ${_strata_gpu_blas_target} ${_strata_gpu_runtime_target})"
if src.count(A) != 1:
    print("ANCHOR_FAILED count=%d" % src.count(A)); sys.exit(2)
ADD = A + """
  # gemm-sol-20261022 (STATUS §286): gemm.cu calls rocblas_gemm_ex directly to pin a Tensile
  # solution per (N,K); hipBLAS only exposes HIPBLAS_GEMM_DEFAULT, so librocblas must be explicit.
  find_library(STRATA_ROCBLAS_LIB NAMES rocblas HINTS /opt/dtk/lib)
  if(STRATA_ROCBLAS_LIB)
    target_link_libraries(strata_prefill PUBLIC ${STRATA_ROCBLAS_LIB})
  else()
    message(WARNING "librocblas not found: STRATA_GEMM_SOL would fail to link")
  endif()"""
src = src.replace(A, ADD, 1)
open(PATH, "w").write(src)
print("CMAKE_PATCH_OK")
