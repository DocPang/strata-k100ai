// Compare the shared native MMVQ Q5_K path with the DSV41 Q5_K results recorded
// by ds4_gguf_test --bench-q5-attn, using the model's exact GEMV shapes.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace K = strata::kernels;

#define CU(x)                                                                                                                        \
    do {                                                                                                                             \
        const cudaError_t e_ = (x);                                                                                                  \
        if (e_ != cudaSuccess) {                                                                                                     \
            std::fprintf(stderr, "CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);                                  \
            std::exit(1);                                                                                                            \
        }                                                                                                                            \
    } while (false)

namespace {

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size() * sizeof(T))));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

std::vector<uint8_t> q5_weights(int n, int k) {
    constexpr size_t block_bytes = 176;
    const size_t row_bytes = size_t(k / 256) * block_bytes;
    std::vector<uint8_t> w(row_bytes * size_t(n));
    uint32_t state = 0x9285a5a5u;
    for (auto& v : w) {
        state = state * 1664525u + 1013904223u;
        v = uint8_t(state >> 24);
    }
    const __half d = __float2half_rn(0.001f), dm = __float2half_rn(0.0005f);
    uint16_t db = 0, dmb = 0;
    std::memcpy(&db, &d, sizeof(db));
    std::memcpy(&dmb, &dm, sizeof(dmb));
    for (size_t at = 0; at < w.size(); at += block_bytes) {
        std::memcpy(&w[at], &db, sizeof(db));
        std::memcpy(&w[at + 2], &dmb, sizeof(dmb));
    }
    return w;
}

void bench(cudaStream_t stream, int n_out, int n_in, bool eight_jobs) {
    const int jobs = eight_jobs ? 8 : 1;
    std::vector<uint8_t> hw = q5_weights(n_out * jobs, n_in);
    std::vector<float> hx(size_t(n_in) * jobs, 0.01f);
    uint8_t* w = dev(hw);
    float* x = dev(hx);
    float* y = nullptr;
    CU(cudaMalloc(&y, size_t(n_out) * jobs * sizeof(float)));
    void* q8 = nullptr;
    const size_t q8_bytes = K::native_q8_1_bytes(n_in, 1);
    CU(cudaMalloc(&q8, size_t(jobs) * q8_bytes));

    const size_t row_bytes = size_t(n_in / 256) * 176;
    const double bytes = double(size_t(n_out) * jobs) * row_bytes;
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));

    auto timed = [&](const char* what, auto&& fn) {
        fn();
        CU(cudaStreamSynchronize(stream));
        CU(cudaEventRecord(e0, stream));
        for (int i = 0; i < 50; ++i) fn();
        CU(cudaEventRecord(e1, stream));
        CU(cudaEventSynchronize(e1));
        float ms = 0;
        CU(cudaEventElapsedTime(&ms, e0, e1));
        const double us = ms * 1000.0 / 50.0;
        std::printf("native Q5_K %-34s %8.1f us  %6.0f GB/s\n", what, us, bytes / (us * 1e-6) / 1e9);
    };

    auto native_many = [&](bool quantize) {
        for (int j = 0; j < jobs; ++j) {
            const uint8_t* wj = w + size_t(j) * n_out * row_bytes;
            const float* xj = x + size_t(j) * n_in;
            float* yj = y + size_t(j) * n_out;
            void* qj = static_cast<uint8_t*>(q8) + size_t(j) * q8_bytes;
            if (quantize) K::native_q5_k_f32(wj, xj, qj, yj, n_in, n_out, 1, stream);
            else K::native_q5_k_mmvq(wj, qj, yj, n_in, n_out, 1, stream);
        }
    };

    char what[96];
    std::snprintf(what, sizeof(what), "%dx%d, %d job%s, f32", n_out, n_in, jobs, jobs == 1 ? "" : "s");
    timed(what, [&] { native_many(true); });
    native_many(true);
    CU(cudaStreamSynchronize(stream));
    std::snprintf(what, sizeof(what), "%dx%d, %d job%s, prequantized", n_out, n_in, jobs, jobs == 1 ? "" : "s");
    timed(what, [&] { native_many(false); });

    CU(cudaEventDestroy(e0));
    CU(cudaEventDestroy(e1));
    CU(cudaFree(w));
    CU(cudaFree(x));
    CU(cudaFree(y));
    CU(cudaFree(q8));
}

}  // namespace

int main() {
    cudaStream_t stream;
    CU(cudaStreamCreate(&stream));
    bench(stream, 32768, 1280, false);
    bench(stream, 1024, 4096, true);
    bench(stream, 5120, 8192, false);
    CU(cudaStreamDestroy(stream));
    return 0;
}
