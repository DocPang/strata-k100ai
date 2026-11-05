// Exercises the portable mapped-host buffer DSV41 uses for cross-device handoffs.
#include "strata/dsv41/hip_or_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                          \
    do {                                                                               \
        hipError_t e_ = (x);                                                           \
        if (e_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),     \
                         __FILE__, __LINE__);                                          \
            std::exit(2);                                                              \
        }                                                                              \
    } while (0)

__global__ void write_device_rows(float* out, int device, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float) (device * 100000 + i);
    return;
}

int main() {
    constexpr int devices = 4, n = 4096;
    int count = 0;
    CK(hipGetDeviceCount(&count));
    if (count < devices) {
        std::fprintf(stderr, "need %d devices, found %d\n", devices, count);
        return 1;
    }

    float* host = nullptr;
    CK(hipHostMalloc(&host, (size_t) devices * n * sizeof(float),
                     hipHostMallocMapped | hipHostMallocPortable));
    for (int i = 0; i < devices * n; ++i) host[i] = -1.0f;

    std::vector<float*> aliases(devices);
    for (int device = 0; device < devices; ++device) {
        CK(hipSetDevice(device));
        CK(hipHostGetDevicePointer((void**) &aliases[device], host, 0));
        if (aliases[device] == nullptr) {
            std::fprintf(stderr, "device %d has no mapped alias\n", device);
            return 1;
        }
    }

    for (int device = 0; device < devices; ++device) {
        CK(hipSetDevice(device));
        write_device_rows<<<(n + 255) / 256, 256>>>(aliases[device] + (size_t) device * n, device, n);
        CK(hipGetLastError());
    }
    for (int device = 0; device < devices; ++device) {
        CK(hipSetDevice(device));
        CK(hipDeviceSynchronize());
    }

    for (int device = 0; device < devices; ++device)
        for (int i = 0; i < n; ++i) {
            const float expected = (float) (device * 100000 + i);
            if (host[(size_t) device * n + i] != expected) {
                std::fprintf(stderr, "device %d value %d: %.9g, expected %.9g\n", device, i,
                             host[(size_t) device * n + i], expected);
                return 1;
            }
        }

    CK(hipHostFree(host));
    std::printf("four-card portable mapped host handoff: ALL OK\n");
    return 0;
}
