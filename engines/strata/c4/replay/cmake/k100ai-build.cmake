# Reconstruct the build-side settings used outside the historical CMake graph.
if(NOT STRATA_HIP_GFX906 OR NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx928")
  message(FATAL_ERROR "This replay contract is specific to gfx928 wave64 HIP compatibility builds")
endif()
find_library(K100AI_HIPBLASLT NAMES hipblaslt HINTS /opt/dtk/lib REQUIRED)
target_link_libraries(strata PRIVATE ${K100AI_HIPBLASLT})
set_property(SOURCE src/prefill/gemm.cu APPEND PROPERTY COMPILE_DEFINITIONS STRATA_HIPBLASLT_AVAILABLE=1)
