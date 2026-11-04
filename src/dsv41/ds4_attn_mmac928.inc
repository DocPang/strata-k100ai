// Opt-in K100AI/gfx928 sparse-attention kernel for chunked prompts.
//
// A CTA handles 16 heads of one query.  One 64-lane wave computes the
// 16-head x 16-key score tile with BF16 MMAC inputs.
// The four waves then own 128 output dimensions each and compute the
// probability-value product.  Window rows are already BF16; packed cache rows
// are decoded to the same BF16 grid.  Query and probability high/low splits
// retain the F32 inputs' useful precision without changing the cache format.
// The exact scalar path remains the default.
#if defined(STRATA_USE_HIP)
namespace attn_mmac928 {

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float floatx4 __attribute__((ext_vector_type(4)));

constexpr int HEADS = 16;
constexpr int KEYS = 16;
constexpr int HD = 512;
constexpr int LD = 520;
constexpr int OUT_TILES = HD / 16;
constexpr int TILES_PER_WAVE = OUT_TILES / 4;

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

template <bool BF16_CKV, bool QSPLIT, bool PSPLIT>
__global__ void __launch_bounds__(256) attn_kernel(
    const float* __restrict__ q, const uint16_t* __restrict__ win,
    const uint8_t* __restrict__ ckv, const int* __restrict__ sel,
    int sel_stride, AttnRows ar, const float* __restrict__ sink,
    int heads, float scale, float* __restrict__ out) {
    __shared__ __align__(16) uint16_t qhi[HEADS][LD];
    __shared__ __align__(16) uint16_t qlo[QSPLIT ? HEADS : 1][LD];
    __shared__ __align__(16) uint16_t ks[KEYS][LD];
    __shared__ float sc[HEADS][KEYS];
    __shared__ float score_part[4][HEADS][KEYS];
    __shared__ uint16_t phi[HEADS][KEYS];
    __shared__ uint16_t plo[PSPLIT ? HEADS : 1][KEYS];
    __shared__ const uint8_t* rows[KEYS];
    __shared__ float m_run[HEADS], den[HEADS], tile_scale[HEADS];

    const int tid = (int) threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int head = lane & 15;
    const int group = lane >> 4;
    const int t = (int) blockIdx.y;
    const int hb = (int) blockIdx.x * HEADS;

    int n_win, n_sel, w0;
    if (ar.gen) {
        const long long ta = ar.t0 + t, pa = ar.p0 + ta;
        const long long first = max((long long) ar.win - ar.p0, ta + 1);
        w0 = (int) first;
        n_win = (int) (ar.win + ta - first + 1);
        n_sel = ar.R > 0 ? (int) min((long long) ar.kcap, (pa + 1) / ar.R) : 0;
    } else {
        n_win = ar.nwin[t];
        n_sel = ar.nsel[t];
        w0 = ar.win0[t];
    }
    const uint16_t* wbase = win + (size_t) w0 * HD;
    const int* sl = sel + (size_t) t * sel_stride;
    const size_t packed_row = kv_row_bytes(kKvFp4, HD);

    const int qr = tid >> 4, qc = (tid & 15) * 32;
#pragma unroll
    for (int x = 0; x < 32; ++x) {
        const float value = q[((size_t) t * heads + hb + qr) * HD + qc + x];
        const uint16_t hi = bf16_bits(value);
        qhi[qr][qc + x] = hi;
        if constexpr (QSPLIT) qlo[qr][qc + x] = bf16_bits(value - bf16f(hi));
    }
    if (tid < HEADS) {
        m_run[tid] = -1e30f;
        den[tid] = 0.0f;
    }

    floatx4 acc[TILES_PER_WAVE];
#pragma unroll
    for (int tile = 0; tile < TILES_PER_WAVE; ++tile)
        acc[tile] = floatx4{0.f, 0.f, 0.f, 0.f};
    __syncthreads();

    const int total = n_win + n_sel;
    for (int k0 = 0; k0 < total; k0 += KEYS) {
        const int nt = min(KEYS, total - k0);
        if (tid < nt) {
            const int j = k0 + tid;
            if (j < n_win) {
                rows[tid] = reinterpret_cast<const uint8_t*>(wbase + (size_t) j * HD);
            } else {
                const int p = sl[j - n_win];
                if constexpr (BF16_CKV) {
                    rows[tid] = reinterpret_cast<const uint8_t*>(
                        reinterpret_cast<const uint16_t*>(ckv) + (size_t) p * HD);
                } else {
                    rows[tid] = ckv + (size_t) p * packed_row;
                }
            }
        }
        __syncthreads();

        const int sr = tid >> 4, part = tid & 15;
        uint16_t values[32] = {};
        if (sr < nt) {
            if (k0 + sr < n_win || BF16_CKV) {
                const uint4* src = reinterpret_cast<const uint4*>(rows[sr] + part * 64);
                reinterpret_cast<uint4*>(values)[0] = src[0];
                reinterpret_cast<uint4*>(values)[1] = src[1];
                reinterpret_cast<uint4*>(values)[2] = src[2];
                reinterpret_cast<uint4*>(values)[3] = src[3];
            } else {
                unpack32(rows[sr], part, reinterpret_cast<uint4*>(values));
            }
        }
#pragma unroll
        for (int x = 0; x < 32; ++x) ks[sr][part * 32 + x] = values[x];
        __syncthreads();

        floatx4 scores{0.f, 0.f, 0.f, 0.f};
        constexpr int KPW = HD / 4;
        const int kd0 = wave * KPW;
#pragma unroll
        for (int kd = kd0; kd < kd0 + KPW; kd += 16) {
            const bf16x4 ah = *reinterpret_cast<const bf16x4*>(&qhi[head][kd + 4 * group]);
            const bf16x4 b = *reinterpret_cast<const bf16x4*>(&ks[head][kd + 4 * group]);
            scores = mmac(ah, b, scores);
            if constexpr (QSPLIT) {
                const bf16x4 al = *reinterpret_cast<const bf16x4*>(&qlo[head][kd + 4 * group]);
                scores = mmac(al, b, scores);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = group + 4 * j;
            score_part[wave][head][key] = scores[j];
        }
        __syncthreads();
        if (tid < HEADS * KEYS) {
            const int h = tid >> 4, key = tid & 15;
            float score = score_part[0][h][key];
#pragma unroll
            for (int w = 1; w < 4; ++w) score += score_part[w][h][key];
            sc[h][key] = key < nt ? score * scale : -INFINITY;
        }
        __syncthreads();

        if (tid < HEADS) {
            float tile_max = sc[tid][0];
#pragma unroll
            for (int key = 1; key < KEYS; ++key) tile_max = fmaxf(tile_max, sc[tid][key]);
            const float old_max = m_run[tid];
            const float new_max = fmaxf(old_max, tile_max);
            float tile_den = 0.0f;
#pragma unroll
            for (int key = 0; key < KEYS; ++key) {
                const float p = expf(sc[tid][key] - new_max);
                const uint16_t hi = bf16_bits(p);
                phi[tid][key] = hi;
                if constexpr (PSPLIT) plo[tid][key] = bf16_bits(p - bf16f(hi));
                tile_den += p;
            }
            const float ts = expf(old_max - new_max);
            tile_scale[tid] = ts;
            den[tid] = den[tid] * ts + tile_den;
            m_run[tid] = new_max;
        }
        __syncthreads();

        const float residue = tile_scale[head];
#pragma unroll
        for (int tile = 0; tile < TILES_PER_WAVE; ++tile) {
            acc[tile][0] *= residue;
            acc[tile][1] *= residue;
            acc[tile][2] *= residue;
            acc[tile][3] *= residue;
        }

#pragma unroll
        for (int local = 0; local < TILES_PER_WAVE; ++local) {
            const int dim0 = 128 * wave + 16 * local;
#pragma unroll
            for (int kk = 0; kk < KEYS; kk += 16) {
                const bf16x4 ah = *reinterpret_cast<const bf16x4*>(&phi[head][kk + 4 * group]);
                bf16x4 b;
#pragma unroll
                for (int x = 0; x < 4; ++x)
                    b[x] = __builtin_bit_cast(__bf16, ks[kk + 4 * group + x][dim0 + head]);
                acc[local] = mmac(ah, b, acc[local]);
                if constexpr (PSPLIT) {
                    const bf16x4 al = *reinterpret_cast<const bf16x4*>(&plo[head][kk + 4 * group]);
                    acc[local] = mmac(al, b, acc[local]);
                }
            }
        }
        __syncthreads();
    }

    const float denominator = den[head] + expf(sink[hb + head] - m_run[head]);
#pragma unroll
    for (int local = 0; local < TILES_PER_WAVE; ++local) {
        const int dim0 = 128 * wave + 16 * local;
        float* dst = out + ((size_t) t * heads + hb + head) * HD + dim0;
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[group + 4 * j] = acc[local][j] / denominator;
    }
}

inline bool device_ok() {
    static std::atomic<int> cache[16];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    int value = cache[dev].load(std::memory_order_relaxed);
    if (value == 0) {
        cudaDeviceProp prop{};
        value = cudaGetDeviceProperties(&prop, dev) == cudaSuccess &&
                        std::strncmp(prop.gcnArchName, "gfx928", 6) == 0
                    ? 2
                    : 1;
        cache[dev].store(value, std::memory_order_relaxed);
    }
    return value == 2;
}

inline bool enabled() {
    static const bool on = [] {
        const char* value = std::getenv("STRATA_DS4_ATTN_MMAC928");
        return value != nullptr && value[0] == '1';
    }();
    return on;
}

inline bool launch(const float* q, const uint16_t* win, const void* ckv,
                   const int* sel, int sel_stride, const AttnRows& ar,
                   const float* sink, int heads, int d, int T, float scale,
                   float* out, int ckv_fmt, int heads_per_block,
                   cudaStream_t s) {
    if (!enabled() || !device_ok() || d != HD || heads <= 0 ||
        heads % HEADS != 0 || T < KEYS || ar.gen == 0 ||
        heads_per_block != 0)
        return false;
    const dim3 grid((unsigned) (heads / HEADS), (unsigned) T);
    const char* qsplit_env = std::getenv("STRATA_DS4_ATTN_MMAC928_QSPLIT");
    const char* psplit_env = std::getenv("STRATA_DS4_ATTN_MMAC928_PSPLIT");
    const bool qsplit = qsplit_env == nullptr || qsplit_env[0] != '0';
    const bool psplit = psplit_env != nullptr && psplit_env[0] == '1';
    if (ckv_fmt == kKvBf16 && qsplit && psplit) {
        attn_kernel<true, true, true><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out);
    } else if (ckv_fmt == kKvBf16 && qsplit) {
        attn_kernel<true, true, false><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out);
    } else if (ckv_fmt == kKvBf16 && psplit) {
        attn_kernel<true, false, true><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out);
    } else if (ckv_fmt == kKvBf16) {
        attn_kernel<true, false, false><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out);
    } else if (ckv_fmt == kKvFp4 && qsplit && psplit) {
        attn_kernel<false, true, true><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out);
    } else if (ckv_fmt == kKvFp4 && qsplit) {
        attn_kernel<false, true, false><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out);
    } else if (ckv_fmt == kKvFp4 && psplit) {
        attn_kernel<false, false, true><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out);
    } else if (ckv_fmt == kKvFp4) {
        attn_kernel<false, false, false><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out);
    } else {
        return false;
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace attn_mmac928
#endif
