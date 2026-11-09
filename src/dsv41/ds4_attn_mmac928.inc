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

template <bool BF16_CKV, bool QSPLIT, bool PSPLIT, bool FINISH_F16, bool DIRECT_FINISH>
__global__ void __launch_bounds__(256) attn_kernel(
    const float* __restrict__ q, const uint16_t* __restrict__ win,
    const uint8_t* __restrict__ ckv, const int* __restrict__ sel,
    int sel_stride, AttnRows ar, const float* __restrict__ sink,
    int heads, float scale, float* __restrict__ out,
    uint16_t* __restrict__ out_f16, const float* __restrict__ rope_freqs,
    int rope_dim, int rope_pos, int rope_vpp) {
    __shared__ __align__(16) uint16_t qhi[HEADS][LD];
    __shared__ __align__(16) uint16_t qlo[QSPLIT ? HEADS : 1][LD];
    __shared__ __align__(16) uint16_t ks[KEYS][LD];
    __shared__ float sc[HEADS][KEYS];
    __shared__ float score_part[4][HEADS][KEYS];
    __shared__ uint16_t phi[HEADS][KEYS];
    __shared__ uint16_t plo[PSPLIT ? HEADS : 1][KEYS];
    __shared__ const uint8_t* rows[KEYS];
    __shared__ float m_run[HEADS], den[HEADS], tile_scale[HEADS];
    __shared__ float epilogue[FINISH_F16 && !DIRECT_FINISH ? HEADS * 128 : 1];

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
    if constexpr (FINISH_F16) {
        const int rope0 = HD - rope_dim;
        if constexpr (DIRECT_FINISH) {
            float* rope_s = &score_part[0][0][0];
            const float inv_den = 1.0f / denominator;
#pragma unroll
            for (int local = 0; local < TILES_PER_WAVE; ++local)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[local][j] *= inv_den;

            // Stage wave 3's 64 inverse-RoPE values in natural row-major
            // order.  This keeps the exact scalar formula while leaving only
            // two block barriers in the direct epilogue.
            if (wave == 3) {
#pragma unroll
                for (int local = 0; local < TILES_PER_WAVE; ++local) {
                    const int dim0 = 128 * wave + 16 * local;
                    if (dim0 < rope0 || dim0 + 16 > rope0 + rope_dim) continue;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        rope_s[head * rope_dim + dim0 - rope0 + group + 4 * j] =
                            acc[local][j];
                }
            }
            __syncthreads();
            for (int item = tid; item < HEADS * (rope_dim / 2); item += 256) {
                const int h = item / (rope_dim / 2), i = item % (rope_dim / 2);
                const int vector = t * heads + hb + h;
                const int vector_pos = rope_pos + vector / rope_vpp;
                const float ang = (float) vector_pos * rope_freqs[i];
                const float c = cosf(ang), sn = -sinf(ang);
                const int at = h * rope_dim + 2 * i;
                const float a = rope_s[at], b = rope_s[at + 1];
                rope_s[at] = a * c - b * sn;
                rope_s[at + 1] = a * sn + b * c;
            }
            __syncthreads();
            if (wave == 3) {
#pragma unroll
                for (int local = 0; local < TILES_PER_WAVE; ++local) {
                    const int dim0 = 128 * wave + 16 * local;
                    if (dim0 < rope0 || dim0 + 16 > rope0 + rope_dim) continue;
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        acc[local][j] =
                            rope_s[head * rope_dim + dim0 - rope0 + group + 4 * j];
                }
            }

            // One scale a 32-value group.  Four lanes, one per value residue,
            // reduce the maxima across xor 16 and xor 32 before each rounds
            // its own registers to the FP8 grid and stores F16.
#pragma unroll
            for (int q = 0; q < TILES_PER_WAVE / 2; ++q) {
                float am = 0.0f;
#pragma unroll
                for (int local = 2 * q; local < 2 * q + 2; ++local)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        am = fmaxf(am, fabsf(acc[local][j]));
                am = fmaxf(am, __shfl_xor(am, 16, 64));
                am = fmaxf(am, __shfl_xor(am, 32, 64));
                const float scale =
                    pow2_ceil(fmaxf(am, 1e-4f) * (1.0f / 448.0f));
#pragma unroll
                for (int local = 2 * q; local < 2 * q + 2; ++local) {
                    const int dim0 = 128 * wave + 16 * local;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float value =
                            fp8_rt(fminf(fmaxf(acc[local][j] / scale, -448.0f), 448.0f)) *
                            scale;
                        out_f16[((size_t) t * heads + hb + head) * HD +
                                dim0 + group + 4 * j] =
                            __half_as_ushort(__float2half_rn(value));
                    }
                }
            }
        } else {
        float* scale_s = &score_part[0][0][0];
        // Stage one 128-value slice for all 16 heads, then apply the scalar
        // finish semantics in that natural row-major layout.  The four waves
        // take turns; the barriers keep the reused 8 KiB slice race-free.
        for (int pw = 0; pw < 4; ++pw) {
            if (wave == pw) {
#pragma unroll
                for (int local = 0; local < TILES_PER_WAVE; ++local)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        epilogue[head * 128 + 16 * local + group + 4 * j] = acc[local][j] / denominator;
            }
            __syncthreads();
            for (int item = tid; item < HEADS * 32; item += 256) {
                const int h = item / 32, i = item % 32;
                if (pw == rope0 / 128) {
                    const int vector = t * heads + hb + h;
                    const int vector_pos = rope_pos + vector / rope_vpp;
                    const float ang = (float) vector_pos * rope_freqs[i];
                    const float c = cosf(ang), sn = -sinf(ang);
                    const int at = h * 128 + (rope0 - 128 * pw) + 2 * i;
                    const float a = epilogue[at], b = epilogue[at + 1];
                    epilogue[at] = a * c - b * sn;
                    epilogue[at + 1] = a * sn + b * c;
                }
            }
            __syncthreads();
            if (tid < HEADS * 4) {
                const int h = tid >> 2, qg = tid & 3;
                float am = 0.0f;
#pragma unroll
                for (int d = 0; d < 32; ++d) am = fmaxf(am, fabsf(epilogue[h * 128 + qg * 32 + d]));
                scale_s[tid] = pow2_ceil(fmaxf(am, 1e-4f) * (1.0f / 448.0f));
            }
            __syncthreads();
            for (int item = tid; item < HEADS * 128; item += 256) {
                const int h = item / 128, d = item % 128;
                const float scale = scale_s[h * 4 + d / 32];
                const float value =
                    fp8_rt(fminf(fmaxf(epilogue[item] / scale, -448.0f), 448.0f)) * scale;
                out_f16[((size_t) t * heads + hb + h) * HD + 128 * pw + d] =
                    __half_as_ushort(__float2half_rn(value));
            }
            __syncthreads();
        }
        }
    } else {
#pragma unroll
        for (int local = 0; local < TILES_PER_WAVE; ++local) {
            const int dim0 = 128 * wave + 16 * local;
            float* dst = out + ((size_t) t * heads + hb + head) * HD + dim0;
#pragma unroll
            for (int j = 0; j < 4; ++j) dst[group + 4 * j] = acc[local][j] / denominator;
        }
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

template <bool FINISH_F16, bool DIRECT_FINISH>
inline bool dispatch(const float* q, const uint16_t* win, const void* ckv,
                     const int* sel, int sel_stride, const AttnRows& ar,
                     const float* sink, int heads, int d, int T, float scale,
                     float* out, uint16_t* out_f16, int ckv_fmt,
                     const float* rope_freqs, int rope_dim, int rope_pos,
                     int rope_vpp, cudaStream_t s) {
    if (!enabled() || !device_ok() || d != HD || heads <= 0 ||
        heads % HEADS != 0 || T < KEYS || ar.gen == 0)
        return false;
    if constexpr (FINISH_F16)
        if (out_f16 == nullptr || rope_freqs == nullptr || rope_dim <= 0 ||
            rope_dim > HD || (rope_dim & 1) != 0 || rope_vpp <= 0)
            return false;
    const dim3 grid((unsigned) (heads / HEADS), (unsigned) T);
    const char* qsplit_env = std::getenv("STRATA_DS4_ATTN_MMAC928_QSPLIT");
    const char* psplit_env = std::getenv("STRATA_DS4_ATTN_MMAC928_PSPLIT");
    const bool qsplit = qsplit_env == nullptr || qsplit_env[0] != '0';
    const bool psplit = psplit_env != nullptr && psplit_env[0] == '1';
    if (ckv_fmt == kKvBf16 && qsplit && psplit) {
        attn_kernel<true, true, true, FINISH_F16, DIRECT_FINISH><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out, out_f16, rope_freqs, rope_dim, rope_pos, rope_vpp);
    } else if (ckv_fmt == kKvBf16 && qsplit) {
        attn_kernel<true, true, false, FINISH_F16, DIRECT_FINISH><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out, out_f16, rope_freqs, rope_dim, rope_pos, rope_vpp);
    } else if (ckv_fmt == kKvBf16 && psplit) {
        attn_kernel<true, false, true, FINISH_F16, DIRECT_FINISH><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out, out_f16, rope_freqs, rope_dim, rope_pos, rope_vpp);
    } else if (ckv_fmt == kKvBf16) {
        attn_kernel<true, false, false, FINISH_F16, DIRECT_FINISH><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out, out_f16, rope_freqs, rope_dim, rope_pos, rope_vpp);
    } else if (ckv_fmt == kKvFp4 && qsplit && psplit) {
        attn_kernel<false, true, true, FINISH_F16, DIRECT_FINISH><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out, out_f16, rope_freqs, rope_dim, rope_pos, rope_vpp);
    } else if (ckv_fmt == kKvFp4 && qsplit) {
        attn_kernel<false, true, false, FINISH_F16, DIRECT_FINISH><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out, out_f16, rope_freqs, rope_dim, rope_pos, rope_vpp);
    } else if (ckv_fmt == kKvFp4 && psplit) {
        attn_kernel<false, false, true, FINISH_F16, DIRECT_FINISH><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out, out_f16, rope_freqs, rope_dim, rope_pos, rope_vpp);
    } else if (ckv_fmt == kKvFp4) {
        attn_kernel<false, false, false, FINISH_F16, DIRECT_FINISH><<<grid, 256, 0, s>>>(
            q, win, reinterpret_cast<const uint8_t*>(ckv), sel, sel_stride,
            ar, sink, heads, scale, out, out_f16, rope_freqs, rope_dim, rope_pos, rope_vpp);
    } else {
        return false;
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

inline bool launch(const float* q, const uint16_t* win, const void* ckv,
                   const int* sel, int sel_stride, const AttnRows& ar,
                   const float* sink, int heads, int d, int T, float scale,
                   float* out, int ckv_fmt, int heads_per_block,
                   cudaStream_t s) {
    if (heads_per_block != 0) return false;
    return dispatch<false, false>(q, win, ckv, sel, sel_stride, ar, sink, heads, d, T,
                                  scale, out, nullptr, ckv_fmt, nullptr, 0, 0, 0, s);
}

inline bool launch_finish_f16(const float* q, const uint16_t* win, const void* ckv,
                              const int* sel, int sel_stride, const AttnRows& ar,
                              const float* sink, int heads, int d, int T, float scale,
                              uint16_t* out_f16, const float* rope_freqs,
                              int rope_dim, int rope_pos, int rope_vpp,
                              int ckv_fmt, cudaStream_t s) {
    static const bool direct_finish = [] {
        const char* value = std::getenv("STRATA_DS4_ATTN_EPILOGUE_DIRECT");
        return value != nullptr && value[0] == '1';
    }();
    if (direct_finish && rope_dim == 64 && rope_dim <= HD &&
        ((HD - rope_dim) % 128) == 64)
        return dispatch<true, true>(q, win, ckv, sel, sel_stride, ar, sink, heads, d, T,
                                    scale, nullptr, out_f16, ckv_fmt, rope_freqs, rope_dim,
                                    rope_pos, rope_vpp, s);
    return dispatch<true, false>(q, win, ckv, sel, sel_stride, ar, sink, heads, d, T,
                                 scale, nullptr, out_f16, ckv_fmt, rope_freqs, rope_dim,
                                 rope_pos, rope_vpp, s);
}

}  // namespace attn_mmac928
#endif
