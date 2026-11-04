// Minimal K100AI/HIP read and copy bandwidth probe for the DSV41 research run.
// It deliberately uses a persistent grid and 16-byte accesses so the result is
// a device-local memory ceiling rather than a PCIe or copy-engine result.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(expr)                                                                  \
    do {                                                                          \
        const hipError_t status_ = (expr);                                        \
        if (status_ != hipSuccess) {                                              \
            std::fprintf(stderr, "%s at %d\n", hipGetErrorString(status_),        \
                         __LINE__);                                                \
            std::exit(2);                                                         \
        }                                                                         \
    } while (0)

__global__ void read_kernel(const uint4* data, size_t count, uint64_t* sink) {
    uint64_t sum = 0;
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        const uint4 value = data[i];
        sum += value.x ^ value.y ^ value.z ^ value.w;
    }
    if (sum == 0xffffffffffffffffull) {
        atomicXor(sink, sum);
    }
}

__global__ void copy_kernel(const uint4* src, uint4* dst, size_t count) {
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void write_kernel(uint4* dst, size_t count) {
    const uint4 value = make_uint4(0x13579bdfu, 0x2468ace0u, 0x55aa55aau, 0xa55aa55au);
    const size_t stride = (size_t) blockDim.x * gridDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        dst[i] = value;
    }
}

static double elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms = 0.0f;
    CK(hipEventElapsedTime(&ms, start, stop));
    return ms;
}

template <typename Fn>
static double best_ms(Fn fn, int repeats) {
    hipEvent_t start, stop;
    CK(hipEventCreate(&start));
    CK(hipEventCreate(&stop));
    double best = 1e100;
    for (int rep = 0; rep < repeats; ++rep) {
        CK(hipEventRecord(start));
        fn();
        CK(hipEventRecord(stop));
        CK(hipEventSynchronize(stop));
        best = std::min(best, elapsed_ms(start, stop));
    }
    CK(hipEventDestroy(start));
    CK(hipEventDestroy(stop));
    return best;
}

int main(int argc, char** argv) {
    const size_t mib = argc > 1 ? (size_t) std::strtoull(argv[1], nullptr, 10) : 2048;
    const int blocks = argc > 2 ? std::max(1, std::atoi(argv[2])) : 2048;
    const size_t bytes = mib << 20;
    const size_t count = bytes / sizeof(uint4);

    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s\n", prop.name);
    std::printf("allocation: %zu MiB, grid: %d blocks x 256 threads\n", mib, blocks);

    uint4 *a = nullptr, *b = nullptr;
    uint64_t* sink = nullptr;
    CK(hipMalloc(&a, bytes));
    CK(hipMalloc(&b, bytes));
    CK(hipMalloc(&sink, sizeof(*sink)));
    CK(hipMemset(a, 0x5a, bytes));
    CK(hipMemset(b, 0, bytes));
    CK(hipMemset(sink, 0, sizeof(*sink)));

    const double read_ms = best_ms(
        [&] { read_kernel<<<blocks, 256>>>(a, count, sink); }, 9);
    const double write_ms = best_ms(
        [&] { write_kernel<<<blocks, 256>>>(b, count); }, 9);
    const double copy_ms = best_ms(
        [&] { copy_kernel<<<blocks, 256>>>(a, b, count); }, 9);
    CK(hipDeviceSynchronize());

    const double gib = (double) bytes / 1e9;
    std::printf("read : %.3f ms, %.1f GB/s\n", read_ms, gib / (read_ms * 1e-3));
    std::printf("write: %.3f ms, %.1f GB/s\n", write_ms, gib / (write_ms * 1e-3));
    std::printf("copy : %.3f ms, %.1f GB/s moved (read+write %.1f GB/s)\n",
                copy_ms, 2.0 * gib / (copy_ms * 1e-3),
                2.0 * gib / (copy_ms * 1e-3));
    return 0;
}
