// Opt-in K100AI/gfx928 prompt-experts path for IQ1_M and IQ2.
//
// The generic >16-row expert tile uses scalar F32 FMAs.  This path decodes
// IQ1_M weights to BF16 64-value tiles in LDS and multiplies them with BF16
// activation tiles on gfx928's native 16x16x16 MMAC instruction.  It keeps the
// original tile as the default and is enabled only by
// STRATA_DS4_MMAC928=1.

#if defined(STRATA_USE_HIP)
namespace mmac928 {

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float floatx4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float apply_swiglu(float gate, float up, float limit) {
    if (limit > 0.0f) {
        gate = fminf(gate, limit);
        up = fminf(fmaxf(up, -limit), limit);
    }
    return gate / (1.0f + __expf(-gate)) * up;
}

__device__ __forceinline__ uint16_t bf16_bits(float v) {
    uint32_t u = __float_as_uint(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

__device__ __forceinline__ bf16x4 bf16_pack(float a, float b, float c, float d) {
    const uint16_t h[4] = {bf16_bits(a), bf16_bits(b), bf16_bits(c), bf16_bits(d)};
    return *reinterpret_cast<const bf16x4*>(h);
}

__device__ __forceinline__ bf16x4 bf16_load_f16_four(const uint16_t* p) {
    return bf16_pack(__half2float(__ushort_as_half(p[0])),
                     __half2float(__ushort_as_half(p[1])),
                     __half2float(__ushort_as_half(p[2])),
                     __half2float(__ushort_as_half(p[3])));
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

template <int T, bool GATE_UP, bool ACTIVE_WAVES, bool SPLIT_GU>
__global__ void __launch_bounds__(256) expert_kernel(
    const SetArgs a, const int8_t* __restrict__ unused_q = nullptr,
    const float* __restrict__ unused_d = nullptr) {
    (void) unused_q;
    (void) unused_d;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int LD = BN + 8;
    constexpr int NT = BN / 16;
    __shared__ uint32_t tab[Ty<T>::TAB];
    __shared__ __align__(16) uint16_t sw[GATE_UP ? 2 : 1][BN][LD];

    const int expert = blockIdx.z;
    const int lo = a.bounds[expert] - a.row0;
    const int hi = a.bounds[expert + 1] - a.row0;
    const int rows = hi - lo;
    if (rows <= 0 || rows <= a.skip_le ||
        (a.skip_ge > 0 && rows >= a.skip_ge)) {
        return;
    }
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi) return;

    const int tid = (int) threadIdx.x;
    load_tab<T>(tab);
    __syncthreads();

    const int K = GATE_UP ? a.n_embd : a.n_ff;
    const int N = GATE_UP ? a.n_ff : a.n_embd;
    const size_t rb = row_bytes_t<T>(K);
    const uint8_t* blob = a.wbase + (size_t) expert * a.stride;
    const int c0 = blockIdx.y * BN;

    // Two threads decode one weight row's two 32-value sub-blocks.
    const int wrow = tid >> 1;
    const int wsub = tid & 1;
    const int nr = min(BM, hi - r0);
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    const int wave = tid >> 6;
    const int token0 = r0 + wave * 16;
    // The down path may skip MMAC issue for waves whose 16 output rows are all
    // past this expert's row range.  Every wave still reaches both barriers.
    const bool wave_active = token0 < hi;

    floatx4 accg[NT], accu[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        accg[nt] = floatx4{0.f, 0.f, 0.f, 0.f};
        accu[nt] = floatx4{0.f, 0.f, 0.f, 0.f};
    }

    for (int k0 = 0; k0 < K; k0 += 64) {
        // gate/up has two independent weight matrices.  Above 16 rows, use one
        // 128-thread half for each decode instead of serialising them in one
        // thread; the single-wave small-expert case keeps the original path.
        bool did_split = false;
        if constexpr (GATE_UP && SPLIT_GU) {
            if (rows > 16) {
                did_split = true;
                const int matrix = tid >> 7;
                const int decode_tid = tid & 127;
                const int decode_row = decode_tid >> 1;
                const int decode_sub = decode_tid & 1;
                const uint8_t* row = blob + a.tr[matrix] + (size_t) (c0 + decode_row) * rb;
                Sub sb;
                decode<T>(row, k0 / 32 + decode_sub, tab, sb);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float scale = sb.d * (float) (j < 4 ? sb.s0 : sb.s1);
                    const int kk = decode_sub * 32 + 4 * j;
                    float value[4];
#pragma unroll
                    for (int x = 0; x < 4; ++x) {
                        value[x] = scale * (float) (int8_t) (sb.q[j] >> (8 * x));
                        if constexpr (Ty<T>::MIN) value[x] -= j < 4 ? sb.m0 : sb.m1;
                    }
                    *reinterpret_cast<bf16x4*>(&sw[matrix][decode_row][kk]) =
                        bf16_pack(value[0], value[1], value[2], value[3]);
                }
            }
        }
        if (!did_split && tid < 128) {
            const uint8_t* row = blob + a.tr[GATE_UP ? 0 : 2] + (size_t) (c0 + wrow) * rb;
            Sub sb;
            decode<T>(row, k0 / 32 + wsub, tab, sb);
            Sub ub;
            if constexpr (GATE_UP) {
                row = blob + a.tr[1] + (size_t) (c0 + wrow) * rb;
                decode<T>(row, k0 / 32 + wsub, tab, ub);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float scale = sb.d * (float) (j < 4 ? sb.s0 : sb.s1);
                const int kk = wsub * 32 + 4 * j;
                float value[4];
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    value[x] = scale * (float) (int8_t) (sb.q[j] >> (8 * x));
                    if constexpr (Ty<T>::MIN) value[x] -= j < 4 ? sb.m0 : sb.m1;
                }
                *reinterpret_cast<bf16x4*>(&sw[0][wrow][kk]) =
                    bf16_pack(value[0], value[1], value[2], value[3]);
                if constexpr (GATE_UP) {
                    const float uscale = ub.d * (float) (j < 4 ? ub.s0 : ub.s1);
#pragma unroll
                    for (int x = 0; x < 4; ++x) {
                        value[x] = uscale * (float) (int8_t) (ub.q[j] >> (8 * x));
                        if constexpr (Ty<T>::MIN) value[x] -= j < 4 ? ub.m0 : ub.m1;
                    }
                    *reinterpret_cast<bf16x4*>(&sw[1][wrow][kk]) =
                        bf16_pack(value[0], value[1], value[2], value[3]);
                }
            }
        }
        __syncthreads();

        if (!ACTIVE_WAVES || wave_active) {
#pragma unroll
            for (int ks = 0; ks < 64; ks += 16) {
                // Measured gfx928 fragment layout:
                //   A lane l: A[l%16][4*(l/16)+k]
                //   B lane l: B[4*(l/16)+k][l%16]
                //   C lane l: C[l%16][l/16+4*j]
                const int token = min(token0 + rr, hi - 1);
                const int at = k0 + ks + 4 * gg;
                bf16x4 x;
                if constexpr (GATE_UP) {
                    const int token_row = a.idx[token];
                    const float* p = a.xf + (size_t) token_row * a.ldx + at;
                    x = bf16_pack(p[0], p[1], p[2], p[3]);
                } else {
                    const uint16_t* p = a.xh + (size_t) token * a.ldx + at;
                    x = bf16_load_f16_four(p);
                }
                for (int nt = 0; nt < NT; ++nt) {
                    const int weight_col = nt * 16 + rr;
                    const bf16x4 wgv =
                        *reinterpret_cast<const bf16x4*>(&sw[0][weight_col][ks + 4 * gg]);
                    accg[nt] = mmac(x, wgv, accg[nt]);
                    if constexpr (GATE_UP) {
                        const bf16x4 wuv =
                            *reinterpret_cast<const bf16x4*>(&sw[1][weight_col][ks + 4 * gg]);
                        accu[nt] = mmac(x, wuv, accu[nt]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int local = wave * 16 + rr;
            if (local >= nr) continue;
            const int col = c0 + nt * 16 + gg + 4 * j;
            if (GATE_UP) {
                a.h[(size_t) (r0 + local) * a.ldo + col] =
                    __half_as_ushort(__float2half_rn(apply_swiglu(accg[nt][j], accu[nt][j], a.limit)));
            } else {
                a.y[(size_t) (r0 + local) * a.ldo + col] = accg[nt][j];
            }
        }
    }
}

// Experts of 17-32 rows otherwise leave half their waves idle: one wave owns
// every output column for rows 0-15, and another does the same for rows 16-31.
// Spread the 64 output columns across all four waves instead.  Each output
// cell keeps the original K order; only the wave that computes it changes.
template <int T>
__global__ void __launch_bounds__(256) gate_up_low_nsplit_kernel(const SetArgs a) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int LD = BN + 8;
    __shared__ uint32_t tab[Ty<T>::TAB];
    __shared__ __align__(16) uint16_t sw[2][BN][LD];

    const int expert = blockIdx.z;
    const int lo = a.bounds[expert] - a.row0;
    const int hi = a.bounds[expert + 1] - a.row0;
    const int rows = hi - lo;
    if (rows <= 0 || rows <= a.skip_le ||
        (a.skip_ge > 0 && rows >= a.skip_ge)) {
        return;
    }
    const int r0 = lo + blockIdx.x * BM;
    if (r0 >= hi || rows > 32) return;

    const int tid = (int) threadIdx.x;
    load_tab<T>(tab);
    __syncthreads();

    const int K = a.n_embd;
    const size_t rb = row_bytes_t<T>(K);
    const uint8_t* blob = a.wbase + (size_t) expert * a.stride;
    const int c0 = blockIdx.y * BN;
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    const int wave = tid >> 6;
    const int row_group = wave & 1;
    const int col_group = wave >> 1;
    const int token0 = r0 + row_group * 16;

    floatx4 accg[2], accu[2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        accg[nt] = floatx4{0.f, 0.f, 0.f, 0.f};
        accu[nt] = floatx4{0.f, 0.f, 0.f, 0.f};
    }

    for (int k0 = 0; k0 < K; k0 += 64) {
        const int matrix = tid >> 7;
        const int decode_tid = tid & 127;
        const int decode_row = decode_tid >> 1;
        const int decode_sub = decode_tid & 1;
        const uint8_t* row = blob + a.tr[matrix] + (size_t) (c0 + decode_row) * rb;
        Sub sb;
        decode<T>(row, k0 / 32 + decode_sub, tab, sb);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float scale = sb.d * (float) (j < 4 ? sb.s0 : sb.s1);
            const int kk = decode_sub * 32 + 4 * j;
            float value[4];
#pragma unroll
            for (int x = 0; x < 4; ++x) {
                value[x] = scale * (float) (int8_t) (sb.q[j] >> (8 * x));
                if constexpr (Ty<T>::MIN) value[x] -= j < 4 ? sb.m0 : sb.m1;
            }
            *reinterpret_cast<bf16x4*>(&sw[matrix][decode_row][kk]) =
                bf16_pack(value[0], value[1], value[2], value[3]);
        }
        __syncthreads();

        if (token0 < hi) {
#pragma unroll
            for (int ks = 0; ks < 64; ks += 16) {
                const int token = min(token0 + rr, hi - 1);
                const int at = k0 + ks + 4 * gg;
                const int token_row = a.idx[token];
                const float* p = a.xf + (size_t) token_row * a.ldx + at;
                const bf16x4 x = bf16_pack(p[0], p[1], p[2], p[3]);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const int tile = col_group * 2 + nt;
                    const int weight_col = tile * 16 + rr;
                    const bf16x4 wgv =
                        *reinterpret_cast<const bf16x4*>(&sw[0][weight_col][ks + 4 * gg]);
                    const bf16x4 wuv =
                        *reinterpret_cast<const bf16x4*>(&sw[1][weight_col][ks + 4 * gg]);
                    accg[nt] = mmac(x, wgv, accg[nt]);
                    accu[nt] = mmac(x, wuv, accu[nt]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int local = row_group * 16 + rr;
            if (r0 + local >= hi) continue;
            const int tile = col_group * 2 + nt;
            const int col = c0 + tile * 16 + gg + 4 * j;
            a.h[(size_t) (r0 + local) * a.ldo + col] =
                __half_as_ushort(__float2half_rn(apply_swiglu(accg[nt][j], accu[nt][j], a.limit)));
        }
    }
}

constexpr int PBN = 64, PLD = PBN + 8, PNT = PBN / 16;

template <int T>
__device__ __forceinline__ void decode_down_sub(const SetArgs& a, const uint8_t* blob, int c0, int k0,
                                                int wrow, int wsub, uint32_t* tab, uint16_t* dst) {
    const size_t rb = row_bytes_t<T>(a.n_ff);
    const uint8_t* row = blob + a.tr[2] + (size_t) (c0 + wrow) * rb;
    Sub sb;
    decode<T>(row, k0 / 32 + wsub, tab, sb);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float scale = sb.d * (float) (j < 4 ? sb.s0 : sb.s1);
        const int kk = wsub * 32 + 4 * j;
        float value[4];
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            value[x] = scale * (float) (int8_t) (sb.q[j] >> (8 * x));
            if constexpr (Ty<T>::MIN) value[x] -= j < 4 ? sb.m0 : sb.m1;
        }
        *reinterpret_cast<bf16x4*>(dst + kk) = bf16_pack(value[0], value[1], value[2], value[3]);
    }
}

__device__ __forceinline__ void down_tile(
    const SetArgs& a, const uint16_t (&w)[PBN][PLD], int k0, int token0, int hi,
    int rr, int gg, floatx4 (&acc)[PNT]) {
#pragma unroll
    for (int ks = 0; ks < 64; ks += 16) {
        const int token = min(token0 + rr, hi - 1);
        const int at = k0 + ks + 4 * gg;
        const uint16_t* p = a.xh + (size_t) token * a.ldx + at;
        const bf16x4 x = bf16_load_f16_four(p);
#pragma unroll
        for (int nt = 0; nt < PNT; ++nt) {
            const int weight_col = nt * 16 + rr;
            const bf16x4 wv =
                *reinterpret_cast<const bf16x4*>(&w[weight_col][ks + 4 * gg]);
            acc[nt] = mmac(x, wv, acc[nt]);
        }
    }
}

// Low-row down sets can overlap the next weight tile's decode with the current
// tile's MMAC.  Wave 0/1 own the first 32 output rows; wave 2/3 decode into the
// alternate shared buffer.  launch() sends experts above this range to the
// single-buffer active-wave kernel, whose shared footprint does not regress.
template <int T>
__global__ void __launch_bounds__(256) down_pipeline_kernel(const SetArgs a) {
    __shared__ uint32_t tab[Ty<T>::TAB];
    __shared__ __align__(16) uint16_t sw[2][PBN][PLD];

    const int expert = blockIdx.z;
    const int lo = a.bounds[expert] - a.row0;
    const int hi = a.bounds[expert + 1] - a.row0;
    const int rows = hi - lo;
    if (rows <= 0 || rows <= a.skip_le ||
        (a.skip_ge > 0 && rows >= a.skip_ge)) {
        return;
    }
    const int r0 = lo + blockIdx.x * 64;
    if (r0 >= hi) return;

    const int tid = (int) threadIdx.x;
    const int lane = tid & 63;
    const int rr = lane & 15;
    const int gg = lane >> 4;
    const int wave = tid >> 6;
    const int token0 = r0 + wave * 16;
    const int remaining = hi - r0;
    const int c0 = blockIdx.y * PBN;
    const uint8_t* blob = a.wbase + (size_t) expert * a.stride;
    floatx4 acc[PNT];
#pragma unroll
    for (int nt = 0; nt < PNT; ++nt) acc[nt] = floatx4{0.f, 0.f, 0.f, 0.f};

    load_tab<T>(tab);
    __syncthreads();

    if (remaining > 32) return;
    if (tid < 128) {
        const int decode_tid = tid;
        decode_down_sub<T>(a, blob, c0, 0, decode_tid >> 1, decode_tid & 1, tab,
                           &sw[0][decode_tid >> 1][0]);
    }
    __syncthreads();

    const int ntiles = (a.n_ff + 63) / 64;
    for (int ti = 0; ti < ntiles; ++ti) {
        const int k0 = ti * 64;
        if (token0 < hi)
            down_tile(a, sw[ti & 1], k0, token0, hi, rr, gg, acc);
        const int next = ti + 1;
        if (tid >= 128 && next < ntiles) {
            const int decode_tid = tid - 128;
            decode_down_sub<T>(a, blob, c0, next * 64, decode_tid >> 1, decode_tid & 1, tab,
                               &sw[next & 1][decode_tid >> 1][0]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < PNT; ++nt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int local = wave * 16 + rr;
            if (local >= remaining) continue;
            const int col = c0 + nt * 16 + gg + 4 * j;
            a.y[(size_t) (r0 + local) * a.ldo + col] = acc[nt][j];
        }
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
        const char* value = std::getenv("STRATA_DS4_MMAC928");
        return value != nullptr && value[0] == '1';
    }();
    return on;
}

template <int T, bool GATE_UP>
inline bool launch(const SetArgs& a, cudaStream_t s) {
    if constexpr (T != kIQ1_M && T != kIQ2_XXS && T != kIQ2_XS) {
        return false;
    } else {
        if (!enabled() || !device_ok() || a.max_rows <= 16 || a.n_exp <= 0) {
            return false;
        }
        if constexpr (GATE_UP) {
            if (a.xf == nullptr || a.idx == nullptr || a.h == nullptr) return false;
        } else if (a.xh == nullptr || a.y == nullptr) {
            return false;
        }
        const int K = GATE_UP ? a.n_embd : a.n_ff;
        const int N = GATE_UP ? a.n_ff : a.n_embd;
        if (K % 64 != 0 || N % 64 != 0 || a.ldx % 4 != 0 || a.ldo % 2 != 0) return false;
        if ((uintptr_t) (GATE_UP ? (const void*) a.xf : (const void*) a.xh) & 15u) return false;
        const dim3 grid((unsigned) ((a.max_rows + 63) / 64), (unsigned) (N / 64), (unsigned) a.n_exp);
        static const bool active_waves = [] {
            const char* value = std::getenv("STRATA_DS4_MMAC928_ACTIVE_WAVES");
            return value != nullptr && value[0] == '1';
        }();
        // These are separate opt-ins so each scheduling change can be A/B'd.
        static const bool split_gu = [] {
            const char* value = std::getenv("STRATA_DS4_MMAC928_SPLIT_GU");
            return value != nullptr && value[0] == '1';
        }();
        static const bool gu_active_waves = [] {
            const char* value = std::getenv("STRATA_DS4_MMAC928_GU_ACTIVE_WAVES");
            return value != nullptr && value[0] == '1';
        }();
        static const bool down_pipe = [] {
            const char* value = std::getenv("STRATA_DS4_MMAC928_DOWN_PIPE");
            return value != nullptr && value[0] == '1';
        }();
        static const bool gu_low_nsplit = [] {
            const char* value = std::getenv("STRATA_DS4_MMAC928_GU_LOW_NSPLIT");
            return value != nullptr && value[0] == '1';
        }();
        if constexpr (GATE_UP) {
            auto launch_high = [&](const SetArgs& args) {
                if (split_gu && gu_active_waves)
                    expert_kernel<T, GATE_UP, true, true><<<grid, 256, 0, s>>>(args);
                else if (split_gu)
                    expert_kernel<T, GATE_UP, false, true><<<grid, 256, 0, s>>>(args);
                else if (gu_active_waves)
                    expert_kernel<T, GATE_UP, true, false><<<grid, 256, 0, s>>>(args);
                else
                    expert_kernel<T, GATE_UP, false, false><<<grid, 256, 0, s>>>(args);
            };
            if (gu_low_nsplit && T != kIQ2_XXS) {
                SetArgs low = a;
                low.skip_ge = a.skip_ge > 0 ? std::min(a.skip_ge, 33) : 33;
                const dim3 low_grid(1, grid.y, grid.z);
                gate_up_low_nsplit_kernel<T><<<low_grid, 256, 0, s>>>(low);
                if (a.max_rows > 32 && (a.skip_ge == 0 || a.skip_ge > 33)) {
                    SetArgs high = a;
                    high.skip_le = std::max(a.skip_le, 32);
                    launch_high(high);
                }
            } else {
                launch_high(a);
            }
        } else if (down_pipe) {
            SetArgs low = a;
            low.skip_ge = a.skip_ge > 0 ? std::min(a.skip_ge, 33) : 33;
            const dim3 low_grid(1, grid.y, grid.z);
            down_pipeline_kernel<T><<<low_grid, 256, 0, s>>>(low);
            if (a.max_rows > 32 && (a.skip_ge == 0 || a.skip_ge > 33)) {
                SetArgs high = a;
                high.skip_le = std::max(a.skip_le, 32);
                if (active_waves)
                    expert_kernel<T, GATE_UP, true, false><<<grid, 256, 0, s>>>(high);
                else
                    expert_kernel<T, GATE_UP, false, false><<<grid, 256, 0, s>>>(high);
            }
        } else if (active_waves) {
            expert_kernel<T, GATE_UP, true, false><<<grid, 256, 0, s>>>(a);
        } else {
            expert_kernel<T, GATE_UP, false, false><<<grid, 256, 0, s>>>(a);
        }
        return cudaPeekAtLastError() == cudaSuccess;
    }
}

}  // namespace mmac928
#endif
