// Opt-in K100AI/gfx928 indexer scorer.
//
// The generic scorer gives one thread one compressed position and evaluates
// all 32 heads with scalar FMAs.  This path computes a 16-query x 16-key tile
// per wave on gfx928's BF16 MMAC and keeps the same per-head ReLU and weighted
// head reduction.  Query values use a two-term BF16 split, while the keys are
// already BF16-grid values.  Candidate-restricted scoring and non-gfx928
// devices keep the original kernel.
#if defined(STRATA_USE_HIP)
namespace qsa_mmac928 {

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float floatx4 __attribute__((ext_vector_type(4)));

constexpr int QT = 16;
constexpr int KB = 64;

__device__ __forceinline__ uint16_t bf16_bits(float v) {
    uint32_t u = __float_as_uint(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ __forceinline__ floatx4 mmac(bf16x4 a, bf16x4 b, floatx4 c) {
#if defined(__gfx928__)
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
#else
    (void) a;
    (void) b;
    return c;
#endif
}

template <int KF>
__device__ __forceinline__ void load_key8(const uint8_t* __restrict__ k, int row, int col,
                                          uint16_t* __restrict__ dst) {
    const uint8_t* kr = k + (size_t) row * kv_row_bytes(KF, 128);
    if constexpr (KF == kKvFp4p) {
        const uint32_t codes = *reinterpret_cast<const uint32_t*>(kr + col / 2);
        const float scale = ue8m0_val(kr[64 + col / 32]);
#pragma unroll
        for (int x = 0; x < 8; ++x)
            dst[x] = bf16_bits(e2m1_val14(codes >> (4 * x)) * 16384.0f * scale);
    } else {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(kr) + col;
#pragma unroll
        for (int x = 0; x < 8; ++x) dst[x] = src[x];
    }
}

// One CTA is four waves: 16 queries x 64 compressed keys.  A wave owns one
// 16-key tile; the 32 heads are reduced in register order, so two executions
// with the same inputs write the same F32 bits.
template <int KF, int LD>
__global__ void __launch_bounds__(256) score_kernel(
    const float* __restrict__ q, const float* __restrict__ hw, const uint8_t* __restrict__ k,
    int T, int n, Rows lim, float* __restrict__ out, long long out_stride) {
    __shared__ __align__(16) uint16_t qhi[QT][LD];
    __shared__ __align__(16) uint16_t qlo[QT][LD];
    __shared__ __align__(16) uint16_t ks[KB][LD];

    const int tid = (int) threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    const int qt0 = (int) blockIdx.y * QT;
    const int kb0 = (int) blockIdx.x * KB;

    // One thread loads eight contiguous values of one query row.
    const int qr = tid >> 4, qc = (tid & 15) * 8;
    const int t = qt0 + qr;
    const bool load_live = t < T;
    const bool out_live = qt0 + rr < T;

    // Index keys are shared by all 32 heads: stage this CTA's 64 keys once.
    for (int item = tid; item < KB * 16; item += 256) {
        const int r = item >> 4, c = (item & 15) * 8;
        const int p = kb0 + r;
        if (p < n) {
            uint16_t packed[8];
            load_key8<KF>(k, p, c, packed);
#pragma unroll
            for (int x = 0; x < 8; ++x) ks[r][c + x] = packed[x];
        } else {
#pragma unroll
            for (int x = 0; x < 8; ++x) ks[r][c + x] = 0;
        }
    }
    __syncthreads();

    floatx4 total{0.f, 0.f, 0.f, 0.f};
    for (int head = 0; head < 32; ++head) {
        // Load this head's query tile.
        const float* qsrc = q + (size_t) (load_live ? t : 0) * 32 * 128 +
                            (size_t) head * 128 + qc;
#pragma unroll
        for (int x = 0; x < 8; ++x) {
            const float value = load_live ? qsrc[x] : 0.0f;
            const uint16_t hi = bf16_bits(value);
            qhi[qr][qc + x] = hi;
            qlo[qr][qc + x] = bf16_bits(value - bf16f(hi));
        }

        __syncthreads();

        floatx4 acc{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kb = 0; kb < 128; kb += 16) {
            const bf16x4 qhi_v = *reinterpret_cast<const bf16x4*>(&qhi[rr][kb + 4 * gg]);
            const bf16x4 qlo_v = *reinterpret_cast<const bf16x4*>(&qlo[rr][kb + 4 * gg]);
            const int key_row = wave * 16 + rr;
            const bf16x4 kv = *reinterpret_cast<const bf16x4*>(&ks[key_row][kb + 4 * gg]);
            acc = mmac(qhi_v, kv, acc);
            acc = mmac(qlo_v, kv, acc);
        }

        const int token = qt0 + rr;
        const float weight = out_live ? hw[(size_t) token * 64 + head] : 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            total[j] += fmaxf(acc[j], 0.0f) * weight;
        __syncthreads();
    }

    if (!out_live) return;
    const int token = qt0 + rr;
    int ln, lk;
    rows_at(lim, token, ln, lk);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int p = kb0 + wave * 16 + gg + 4 * j;
        float* dst = out + (size_t) token * out_stride;
        dst[p] = p < ln ? total[j] : -INFINITY;
    }
}

// The candidate-block path gives each query a different set of compressed
// keys, so a 16-query x 16-key matrix tile cannot be reused.  The generic
// fallback gives one thread one key and all 32 heads, which keeps 32
// accumulators live.  Split each key across a four-lane group instead: each
// lane computes eight heads, while one lane sums the 32 ReLU/weight terms in
// the original head order after the group's dot products finish.
template <int KF>
__global__ void __launch_bounds__(256) candidate_score_kernel(
    const float* __restrict__ q, const float* __restrict__ hw, const uint8_t* __restrict__ k,
    int T, int n, const int* __restrict__ cand, int cand_stride, int blk,
    Rows lim, float* __restrict__ out, long long out_stride) {
    extern __shared__ float smem[];
    float* qs = smem;                    // 32 * 128 query values, then 32 weights
    float* part = qs + 32 * 128 + 32;    // 64 candidate positions x 32 heads
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < 32 * 128; i += blockDim.x) qs[i] = q[(size_t) t * 32 * 128 + i];
    for (int i = threadIdx.x; i < 32; i += blockDim.x) qs[32 * 128 + i] = hw[(size_t) t * 64 + i];
    __syncthreads();

    const int group = threadIdx.x >> 2;  // one candidate position
    const int lane = threadIdx.x & 3;    // eight heads
    const int i = blockIdx.x * 64 + group;
    int ln = 0, lk = 0;
    rows_at(lim, t, ln, lk);
    int p = i;
    bool live = i < n;
    if (live && cand != nullptr) {
        live = i < lk * blk;
        if (live) p = cand[(size_t) t * cand_stride + i / blk] * blk + i % blk;
    }
    live = live && p < ln;

    const int h0 = lane * 8;
    float acc[8] = {};
    if (live) {
        const uint8_t* kr = k + (size_t) p * kv_row_bytes(KF, 128);
#pragma unroll 2
        for (int c = 0; c < 128; c += 8) {
            float kv[8];
            if constexpr (KF == kKvFp4p) {
                const uint32_t cw = *reinterpret_cast<const uint32_t*>(kr + c / 2);
                const float ks = ue8m0_val(kr[64 + c / 32]);
#pragma unroll
                for (int x = 0; x < 8; ++x)
                    kv[x] = e2m1_val14(cw >> (4 * x)) * 16384.0f * ks;
            } else {
                const uint4 raw = *reinterpret_cast<const uint4*>(kr + c * 2);
                const uint16_t* bits = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
                for (int x = 0; x < 8; ++x) kv[x] = bf16f(bits[x]);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float* qh = qs + (h0 + j) * 128 + c;
#pragma unroll
                for (int x = 0; x < 8; ++x) acc[j] += qh[x] * kv[x];
            }
        }
    }
    for (int j = 0; j < 8; ++j)
        part[group * 32 + h0 + j] = fmaxf(acc[j], 0.0f) * qs[32 * 128 + h0 + j];
    __syncthreads();
    if (lane == 0 && i < n) {
        float s = 0.0f;
        for (int h = 0; h < 32; ++h) s += part[group * 32 + h];
        out[(size_t) t * out_stride + i] = live ? s : -INFINITY;
    }
}

inline bool device_ok() {
    static std::atomic<int> cached[16];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    int value = cached[dev].load(std::memory_order_relaxed);
    if (value == 0) {
        cudaDeviceProp prop{};
        value = cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                        std::strncmp(prop.gcnArchName, "gfx928", 6) == 0
                    ? 2
                    : 1;
        cached[dev].store(value, std::memory_order_relaxed);
    }
    return value == 2;
}

inline bool enabled() {
    static const bool on = [] {
        const char* value = std::getenv("STRATA_DS4_QSA_MMAC928");
        return value != nullptr && value[0] == '1';
    }();
    return on;
}

inline bool candidate_enabled() {
    const char* value = std::getenv("STRATA_DS4_QSA_MMAC928_CANDIDATE");
    return value != nullptr && value[0] == '1';
}

template <int KF>
inline bool launch(const float* q, const float* w, const uint8_t* k, int heads, int n,
                   int T, const int* cand, int cand_stride, int blk, const Rows& lim,
                   float* out, long long out_stride, cudaStream_t s) {
    if (!enabled() || !device_ok() || heads != 32 || n <= 0) return false;
    if (cand != nullptr) {
        if (!candidate_enabled()) return false;
        const dim3 grid((unsigned) ((n + 63) / 64), (unsigned) T);
        const size_t smem = (size_t) (32 * 128 + 32 + 64 * 32) * sizeof(float);
        candidate_score_kernel<KF><<<grid, 256, smem, s>>>(q, w, k, T, n, cand, cand_stride, blk,
                                                           lim, out, out_stride);
        return cudaPeekAtLastError() == cudaSuccess;
    }
    const dim3 grid((unsigned) ((n + KB - 1) / KB), (unsigned) ((T + QT - 1) / QT));
    static const int ld = [] {
        const char* value = std::getenv("STRATA_DS4_QSA_MMAC928_LD");
        return value != nullptr ? std::atoi(value) : 136;
    }();
    if (ld == 132)
        score_kernel<KF, 132><<<grid, 256, 0, s>>>(q, w, k, T, n, lim, out, out_stride);
    else
        score_kernel<KF, 136><<<grid, 256, 0, s>>>(q, w, k, T, n, lim, out, out_stride);
    return cudaPeekAtLastError() == cudaSuccess;
}

inline bool launch(const float* q, const float* w, const uint8_t* k, int heads, int n,
                   int T, const int* cand, int cand_stride, int blk, const Rows& lim,
                   float* out, long long out_stride, cudaStream_t s, int kfmt) {
    if (kfmt == kKvFp4p)
        return launch<kKvFp4p>(q, w, k, heads, n, T, cand, cand_stride, blk, lim, out,
                               out_stride, s);
    if (kfmt == kKvBf16)
        return launch<kKvBf16>(q, w, k, heads, n, T, cand, cand_stride, blk, lim, out,
                               out_stride, s);
    return false;
}

}  // namespace qsa_mmac928
#endif
