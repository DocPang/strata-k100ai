// Exercises the DSV41 top-k and MoE-sort kernels on a logical 32-lane warp.
// The inputs include ties around half-wave boundaries, which catches a wave64
// ballot or sync that accidentally spans two logical CUDA warps.
#include "../../src/dsv41/ds4_kernels.cuh"

#include "strata/dsv41/hip_or_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace ds4 = strata::ds4;

#define CK(x)                                                                          \
    do {                                                                               \
        hipError_t e_ = (x);                                                           \
        if (e_ != hipSuccess) {                                                        \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),     \
                         __FILE__, __LINE__);                                          \
            std::exit(2);                                                              \
        }                                                                              \
    } while (0)

template <typename T>
static T* upload(const std::vector<T>& v) {
    T* p = nullptr;
    CK(hipMalloc(&p, std::max<size_t>(1, v.size() * sizeof(T))));
    if (!v.empty()) CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}

template <typename T>
static std::vector<T> download(const T* p, size_t n) {
    std::vector<T> v(n);
    if (n) CK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

static bool topk_case(std::mt19937& rng, int T, int n, int k) {
    const long long stride = n + 7;
    std::vector<float> values((size_t) T * stride, -INFINITY);
    std::vector<int> expected((size_t) T * k);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < n; ++i) {
            // Coarse scores create ties at every lane and half-wave boundary.
            const int score = (int) (rng() % 97);
            const float frac = i == n - 1 ? 0.0f : (float) (rng() % 8) * 1e-4f;
            values[(size_t) t * stride + i] = (float) score + frac;
        }
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const float av = values[(size_t) t * stride + a];
            const float bv = values[(size_t) t * stride + b];
            return av > bv || (av == bv && a < b);
        });
        order.resize(k);
        std::sort(order.begin(), order.end());
        std::copy(order.begin(), order.end(), expected.begin() + (size_t) t * k);
    }

    float* d_values = upload(values);
    int* d_out = nullptr;
    CK(hipMalloc(&d_out, (size_t) T * k * sizeof(int)));
    ds4::Rows rows{};
    for (int t = 0; t < T; ++t) rows.n[t] = n, rows.k[t] = k;
    ds4::topk(d_values, stride, T, rows, nullptr, 0, 1, d_out, k, 0);
    CK(hipDeviceSynchronize());
    std::vector<int> got = download(d_out, (size_t) T * k);
    CK(hipFree(d_out));
    CK(hipFree(d_values));
    // topk() documents an unordered selection.  Compare the selected sets per
    // row; moe_sort() below checks the ordered placement.
    for (int t = 0; t < T; ++t) {
        std::sort(got.begin() + (size_t) t * k, got.begin() + (size_t) (t + 1) * k);
        std::sort(expected.begin() + (size_t) t * k, expected.begin() + (size_t) (t + 1) * k);
    }
    if (got != expected) {
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < k; ++i) {
                const int a = got[(size_t) t * k + i], b = expected[(size_t) t * k + i];
                if (a != b) {
                    std::fprintf(stderr, "topk FAIL T=%d n=%d k=%d row=%d pos=%d got=%d expected=%d\n",
                                 T, n, k, t, i, a, b);
                    std::fprintf(stderr, "got     :");
                    for (int j = 0; j < k; ++j) std::fprintf(stderr, " %d", got[(size_t) t * k + j]);
                    std::fprintf(stderr, "\nexpected:");
                    for (int j = 0; j < k; ++j) std::fprintf(stderr, " %d", expected[(size_t) t * k + j]);
                    std::fprintf(stderr, "\nvalues  :");
                    for (int j = 0; j < k; ++j) {
                        const int id = expected[(size_t) t * k + j];
                        std::fprintf(stderr, " %d=%.6g", id, values[(size_t) t * stride + id]);
                    }
                    std::fprintf(stderr, "\n");
                    return false;
                }
            }
        }
    }
    return true;
}

static bool sort_case() {
    constexpr int T = 5, K = ds4::kPicks, NVS = 7;
    std::vector<int> idx((size_t) T * K), vs(NVS);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i) idx[(size_t) t * K + i] = (t * 3 + i * 5) % NVS;
    for (int e = 0; e < NVS; ++e) vs[e] = (e * 5 + 2) % NVS;

    std::vector<int> expected_bounds(NVS + 1, 0), expected_tok, expected_pos((size_t) T * K, -1);
    for (int s = 0; s < NVS; ++s) {
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < K; ++i) {
                if (vs[idx[(size_t) t * K + i]] == s) {
                    expected_pos[(size_t) t * K + i] = (int) expected_tok.size();
                    expected_tok.push_back(t);
                }
            }
        }
        expected_bounds[s + 1] = (int) expected_tok.size();
    }

    int* d_idx = upload(idx);
    int* d_vs = upload(vs);
    int *d_bounds = nullptr, *d_tok = nullptr, *d_pos = nullptr;
    CK(hipMalloc(&d_bounds, (NVS + 1) * sizeof(int)));
    CK(hipMalloc(&d_tok, (size_t) T * K * sizeof(int)));
    CK(hipMalloc(&d_pos, (size_t) T * K * sizeof(int)));
    ds4::moe_sort(d_idx, T, K, d_vs, NVS, d_bounds, d_tok, d_pos, 0);
    CK(hipDeviceSynchronize());
    const auto bounds = download(d_bounds, NVS + 1);
    const auto tok = download(d_tok, expected_tok.size());
    const auto pos = download(d_pos, expected_pos.size());
    CK(hipFree(d_pos));
    CK(hipFree(d_tok));
    CK(hipFree(d_bounds));
    CK(hipFree(d_vs));
    CK(hipFree(d_idx));
    if (bounds != expected_bounds || tok != expected_tok || pos != expected_pos) {
        std::fprintf(stderr, "moe_sort FAIL\n");
        return false;
    }
    return true;
}

static bool route_case() {
    constexpr int T = 4, N = 384, K = 6, LogitStride = 1024;
    constexpr float scale = 1.5f;
    std::vector<float> logits((size_t) T * LogitStride), bias(N);
    for (int t = 0; t < T; ++t)
        for (int e = 0; e < N; ++e)
            logits[(size_t) t * LogitStride + e] =
                -5.0f + 10.0f * (float) ((t * 97 + e * 13) % 997) / 996.0f;
    for (int e = 0; e < N; ++e) bias[e] = -0.5f + (float) ((e * 29) % 101) / 100.0f;

    std::vector<int> expected_idx((size_t) T * K);
    std::vector<float> expected_w((size_t) T * K);
    for (int t = 0; t < T; ++t) {
        std::vector<float> score(N);
        for (int e = 0; e < N; ++e) {
            const float x = logits[(size_t) t * LogitStride + e];
            score[e] = std::sqrt(std::log1p(std::exp(x)));
        }
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const float av = score[a] + bias[a], bv = score[b] + bias[b];
            return av > bv || (av == bv && a < b);
        });
        order.resize(K);
        float sum = 0.0f;
        for (int e : order) sum += score[e];
        for (int i = 0; i < K; ++i) {
            expected_idx[(size_t) t * K + i] = order[i];
            expected_w[(size_t) t * K + i] = score[order[i]] / (sum + 1e-20f) * scale;
        }
    }

    float* d_logits = upload(logits);
    float* d_bias = upload(bias);
    int* d_idx = nullptr;
    float* d_w = nullptr;
    CK(hipMalloc(&d_idx, (size_t) T * ds4::kPicks * sizeof(int)));
    CK(hipMalloc(&d_w, (size_t) T * ds4::kPicks * sizeof(float)));
    ds4::route(d_logits, d_bias, N, K, T, scale, d_idx, d_w, 0);
    CK(hipDeviceSynchronize());
    const auto idx = download(d_idx, (size_t) T * ds4::kPicks);
    const auto w = download(d_w, (size_t) T * ds4::kPicks);
    CK(hipFree(d_w));
    CK(hipFree(d_idx));
    CK(hipFree(d_bias));
    CK(hipFree(d_logits));
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < K; ++i) {
            if (idx[(size_t) t * ds4::kPicks + i] != expected_idx[(size_t) t * K + i] ||
                std::fabs(w[(size_t) t * ds4::kPicks + i] - expected_w[(size_t) t * K + i]) > 2e-6f) {
                std::fprintf(stderr, "route FAIL row=%d pick=%d idx=%d/%d w=%.9g/%.9g\n", t, i,
                             idx[(size_t) t * ds4::kPicks + i], expected_idx[(size_t) t * K + i],
                             w[(size_t) t * ds4::kPicks + i], expected_w[(size_t) t * K + i]);
                return false;
            }
        }
    }
    return true;
}

int main() {
    std::mt19937 rng(0x928u);
    const int shapes[][2] = {{31, 6}, {32, 8}, {33, 8}, {63, 16}, {64, 16},
                             {65, 17}, {1024, 64}, {2048, 512}};
    for (const auto& shape : shapes) {
        if (!topk_case(rng, 4, shape[0], shape[1])) return 1;
    }
    if (!sort_case()) return 1;
    if (!route_case()) return 1;
    std::printf("top-k, MoE sort, and routing: ALL OK\n");
    return 0;
}
