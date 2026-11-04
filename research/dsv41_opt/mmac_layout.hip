// Validate the gfx928 BF16 MMAC register-fragment mapping on one wave.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(expr)                                                                 \
    do {                                                                         \
        const hipError_t status_ = (expr);                                       \
        if (status_ != hipSuccess) {                                             \
            std::fprintf(stderr, "%s at %d\n", hipGetErrorString(status_),       \
                         __LINE__);                                               \
            std::exit(2);                                                        \
        }                                                                        \
    } while (0)

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float floatx4 __attribute__((ext_vector_type(4)));

__device__ static uint16_t bf16_bits(float v) {
    uint32_t u = __float_as_uint(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ static bf16x4 pack(float a, float b, float c, float d) {
    const uint16_t h[4] = {bf16_bits(a), bf16_bits(b), bf16_bits(c), bf16_bits(d)};
    return *reinterpret_cast<const bf16x4*>(h);
}

__global__ void probe(const float* a, const float* b, float* c, int variant) {
    const int lane = threadIdx.x;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    float av[4], bv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        av[j] = a[rr * 16 + 4 * gg + j];
        bv[j] = b[rr * 16 + 4 * gg + j];
    }
    floatx4 acc{0.f, 0.f, 0.f, 0.f};
    acc = __builtin_amdgcn_mmac_f32_16x16x16bf16(pack(av[0], av[1], av[2], av[3]),
                                                 pack(bv[0], bv[1], bv[2], bv[3]), acc);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        c[rr * 16 + gg + 4 * j] = acc[j];
    }
}

int main() {
    std::vector<float> a(256), b(256), got(256), want(256, 0.0f);
    for (int i = 0; i < 16; ++i) {
        for (int k = 0; k < 16; ++k) {
            a[i * 16 + k] = (float) ((i * 17 + k * 3) % 31 - 15);
            b[i * 16 + k] = (float) ((i * 5 + k * 7) % 23 - 11);
        }
    }
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            for (int k = 0; k < 16; ++k)
                want[i * 16 + j] += a[i * 16 + k] * b[j * 16 + k];

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CK(hipMalloc(&da, a.size() * sizeof(float)));
    CK(hipMalloc(&db, b.size() * sizeof(float)));
    CK(hipMalloc(&dc, got.size() * sizeof(float)));
    CK(hipMemcpy(da, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(db, b.data(), b.size() * sizeof(float), hipMemcpyHostToDevice));
    int failures = 0;
    CK(hipMemset(dc, 0, got.size() * sizeof(float)));
    probe<<<1, 64>>>(da, db, dc, 0);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(got.data(), dc, got.size() * sizeof(float), hipMemcpyDeviceToHost));

    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::fabs(got[i] - want[i]));
    std::printf("worst absolute error: %.6f\n", worst);
    return worst <= 0.5f ? 0 : 1;
}
