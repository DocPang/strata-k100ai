// k100ai / gfx928 (wave64 compat layer): the one symbol the hipBLASLt path needs that this backend does not build.
//
// wmma_gemm.h declares strata_pf_hcdown_exact_bf16 unconditionally, but only wmma_gemm.cu defines it, and that file is
// built for the RDNA3 backend alone.  Upstream never noticed because the wave64 backend never enabled hipBLASLt; with
// STRATA_HIPBLASLT_AVAILABLE=1 the call in gemm.cu becomes reachable.  Without gfx11 matrix cores there is no exact
// kernel to launch, so this returns false exactly like the file own stub (wmma_gemm.cu, non-gfx11 branch).
#include <cstdint>

bool strata_pf_hcdown_exact_bf16(const uint16_t*, int64_t, const uint16_t*, float*, int64_t, int64_t, int64_t,
                                 void*) {
    return false;
}
