#include <immintrin.h>

#include <cstring>
#include <vector>

#include "lut_build.h"
#include "ternary_mm.h"

// AVX-512 (F+BW) kernel exploiting the +/- mirror symmetry of balanced
// ternary: a packed byte v in [-121, 121] resolves as sign(v) * T[|v|],
// and the 122 magnitudes fit the 128-entry index space of a vpermt2w pair.
// The lookup therefore runs entirely out of registers: two vpermt2w + one
// blend give 32 int16 lookups per ~3 instructions.
//
// Per weight group the table T[m] = lut243[121 + m] is assembled without
// any scalar arithmetic: with n = 121 + m = 9*(h+13) + (l+4), where h is
// the value of the top 3 trits and l of the bottom 2, T[m] = H[n/9] +
// L[n%9]. H (27 entries) and L (9 entries) are themselves built as
// broadcast-multiply-adds against constant digit vectors (H = a0*D0 +
// a1*D1 + a2*D2 lane-wise), and n/9, n%9 per table lane are compile-time
// constants, so the four zmm table registers cost ~10 multiplies/adds plus
// eight constant-index vpermw.
//
// Rows are processed in blocks of four: the index decode (widen, abs,
// sign mask, bit-6 mask) depends only on the packed weights, so it is
// shared between the rows' lookups. A trailing N % 32 columns are handled
// by the same SIMD body under AVX-512BW masked loads/stores, so there is
// no scalar tail and no performance cliff at vector boundaries.

namespace {

// Digit-constant vectors for the table build. H lane t (t < 27) holds the
// balanced-ternary digits of t: D0=t/9-1, D1=(t/3)%3-1, D2=t%3-1; L lane
// t (t < 9): D3=t/3-1, D4=t%3-1. Lanes past the table size stay zero.
struct BuildConsts {
    __m512i digits[5];
    __m512i idx_h[4];
    __m512i idx_l[4];
};

BuildConsts make_build_consts() {
    alignas(64) int16_t d[5][32] = {};
    for (int t = 0; t < 27; ++t) {
        d[0][t] = (int16_t)(t / 9 - 1);
        d[1][t] = (int16_t)(t / 3 % 3 - 1);
        d[2][t] = (int16_t)(t % 3 - 1);
    }
    for (int t = 0; t < 9; ++t) {
        d[3][t] = (int16_t)(t / 3 - 1);
        d[4][t] = (int16_t)(t % 3 - 1);
    }
    // Table lane t of register k holds lut243[121 + 32k + t]; precompute
    // the H and L source lane for each. (Lanes past m=121 read the zero
    // padding of H/L and are never selected, since |v| <= 121.)
    alignas(64) int16_t ih[4][32], il[4][32];
    for (int k = 0; k < 4; ++k) {
        for (int t = 0; t < 32; ++t) {
            const int n = 121 + 32 * k + t;
            ih[k][t] = (int16_t)(n / 9);
            il[k][t] = (int16_t)(n % 9);
        }
    }
    BuildConsts bc;
    for (int w = 0; w < 5; ++w) bc.digits[w] = _mm512_loadu_si512(d[w]);
    for (int k = 0; k < 4; ++k) {
        bc.idx_h[k] = _mm512_loadu_si512(ih[k]);
        bc.idx_l[k] = _mm512_loadu_si512(il[k]);
    }
    return bc;
}

const BuildConsts& build_consts() {
    static const BuildConsts value = make_build_consts();
    return value;
}

inline void build_tables(const int8_t* a, const BuildConsts& bc,
                         __m512i T[4]) {
    const __m512i hv = _mm512_add_epi16(
        _mm512_add_epi16(
            _mm512_mullo_epi16(_mm512_set1_epi16(a[0]), bc.digits[0]),
            _mm512_mullo_epi16(_mm512_set1_epi16(a[1]), bc.digits[1])),
        _mm512_mullo_epi16(_mm512_set1_epi16(a[2]), bc.digits[2]));
    const __m512i lv = _mm512_add_epi16(
        _mm512_mullo_epi16(_mm512_set1_epi16(a[3]), bc.digits[3]),
        _mm512_mullo_epi16(_mm512_set1_epi16(a[4]), bc.digits[4]));
    for (int k = 0; k < 4; ++k) {
        T[k] = _mm512_add_epi16(_mm512_permutexvar_epi16(bc.idx_h[k], hv),
                                _mm512_permutexvar_epi16(bc.idx_l[k], lv));
    }
}

// crow = acc16 (first flush) or crow += acc16; acc16 = 0.
inline void flush_row(int16_t* acc16, int32_t* crow, int N, bool first) {
    int j = 0;
    for (; j + 32 <= N; j += 32) {
        const __m512i a = _mm512_loadu_si512(acc16 + j);
        __m512i lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a));
        __m512i hi =
            _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a, 1));
        if (!first) {
            lo = _mm512_add_epi32(_mm512_loadu_si512(crow + j), lo);
            hi = _mm512_add_epi32(_mm512_loadu_si512(crow + j + 16), hi);
        }
        _mm512_storeu_si512(crow + j, lo);
        _mm512_storeu_si512(crow + j + 16, hi);
        _mm512_storeu_si512(acc16 + j, _mm512_setzero_si512());
    }
    for (; j < N; ++j) {
        crow[j] = (first ? 0 : crow[j]) + acc16[j];
        acc16[j] = 0;
    }
}

// Processes R rows (activation rows a_base + r*K, outputs c_base + r*N).
// acc16 must hold R*N zeroed int16 and is left zeroed on return.
template <int R>
void lut_rows(const int8_t* __restrict a_base, const int8_t* __restrict P,
              int K, int N, int32_t* __restrict c_base,
              int16_t* __restrict acc16, const BuildConsts& bc) {
    const int G = K / 5;
    // int16 partial sums are flushed to int32 before they can overflow:
    // 48 groups * max |entry| 640 = 30720 < 32767.
    constexpr int kFlushGroups = 48;
    // The last N % 32 columns run the same body under these masks.
    const int rem = N & 31;
    const __mmask64 tail_b = rem ? (((uint64_t)1 << rem) - 1) : 0;
    const __mmask32 tail_w = rem ? (((uint32_t)1 << rem) - 1) : 0;
    const __m512i zero = _mm512_setzero_si512();
    const __m512i c64 = _mm512_set1_epi16(64);

    bool first = true;
    int pending = 0;
    __m512i T[R][4];
    for (int g = 0; g < G; ++g) {
        for (int r = 0; r < R; ++r)
            build_tables(a_base + (size_t)r * K + 5 * g, bc, T[r]);
        const int8_t* pw = P + (size_t)g * N;
        int j = 0;
        for (; j + 32 <= N; j += 32) {
            const __m256i pb = _mm256_loadu_si256((const __m256i*)(pw + j));
            const __m512i v = _mm512_cvtepi8_epi16(pb);
            const __mmask32 kneg = _mm512_movepi16_mask(v);
            const __m512i m = _mm512_abs_epi16(v);
            const __mmask32 khi = _mm512_test_epi16_mask(m, c64);
            for (int r = 0; r < R; ++r) {
                const __m512i r0 = _mm512_permutex2var_epi16(T[r][0], m,
                                                             T[r][1]);
                const __m512i r1 = _mm512_permutex2var_epi16(T[r][2], m,
                                                             T[r][3]);
                __m512i rr = _mm512_mask_blend_epi16(khi, r0, r1);
                rr = _mm512_mask_sub_epi16(rr, kneg, zero, rr);
                int16_t* acc = acc16 + (size_t)r * N + j;
                _mm512_storeu_si512(
                    acc, _mm512_add_epi16(_mm512_loadu_si512(acc), rr));
            }
        }
        if (rem) {
            const __m512i pb = _mm512_maskz_loadu_epi8(tail_b, pw + j);
            const __m512i v =
                _mm512_cvtepi8_epi16(_mm512_castsi512_si256(pb));
            const __mmask32 kneg = _mm512_movepi16_mask(v);
            const __m512i m = _mm512_abs_epi16(v);
            const __mmask32 khi = _mm512_test_epi16_mask(m, c64);
            for (int r = 0; r < R; ++r) {
                const __m512i r0 = _mm512_permutex2var_epi16(T[r][0], m,
                                                             T[r][1]);
                const __m512i r1 = _mm512_permutex2var_epi16(T[r][2], m,
                                                             T[r][3]);
                __m512i rr = _mm512_mask_blend_epi16(khi, r0, r1);
                rr = _mm512_mask_sub_epi16(rr, kneg, zero, rr);
                int16_t* acc = acc16 + (size_t)r * N + j;
                const __m512i cur = _mm512_maskz_loadu_epi16(tail_w, acc);
                _mm512_mask_storeu_epi16(acc, tail_w,
                                         _mm512_add_epi16(cur, rr));
            }
        }
        if (++pending == kFlushGroups || g + 1 == G) {
            for (int r = 0; r < R; ++r)
                flush_row(acc16 + (size_t)r * N, c_base + (size_t)r * N, N,
                          first);
            first = false;
            pending = 0;
        }
    }
}

}  // namespace

void lut_mm_avx512(const int8_t* __restrict A, const int8_t* __restrict P,
                   int M, int K, int N, int32_t* __restrict C) {
    const int G = K / 5;
    if (G == 0) {
        // No groups: the first-flush-store path would never write C.
        std::memset(C, 0, (size_t)M * N * sizeof(int32_t));
        return;
    }
    // Row blocks of 4: measured best on Zen 5 — R=8's 32 table registers
    // spill and give back the decode sharing.
    const BuildConsts& bc = build_consts();
    std::vector<int16_t> acc16((size_t)(M < 4 ? M : 4) * N, 0);
    int i = 0;
    for (; i + 4 <= M; i += 4) {
        lut_rows<4>(A + (size_t)i * K, P, K, N, C + (size_t)i * N,
                    acc16.data(), bc);
    }
    for (; i + 2 <= M; i += 2) {
        lut_rows<2>(A + (size_t)i * K, P, K, N, C + (size_t)i * N,
                    acc16.data(), bc);
    }
    for (; i < M; ++i) {
        lut_rows<1>(A + (size_t)i * K, P, K, N, C + (size_t)i * N,
                    acc16.data(), bc);
    }
}

// ---------------------------------------------------------------------------
// 256-bit control variant (AVX-512VL), used to separate vector width from
// the 5-trit/122-entry design when comparing against BitNet's 256-bit TL2
// kernel. Deliberately kept at the original (v1) sweep design so the
// width-scaling comparison against the TL2@512 port stays matched; see the
// README's width-vs-design section. The 128 int16 entries span eight ymm
// registers and a two-register vpermt2w covers only 32 of them, so each
// 16-lane lookup needs four permutes plus a three-blend select tree on
// index bits 5-6 — permute table capacity scales with the vector width,
// which is part of what 512 bits buys this design.

namespace {

// out[idx] = dot(a[0..count), digits(idx)), idx in base 3, digit-1 per
// trit, a[0] paired with the most significant trit. Writes 3^count
// entries. (v1 scalar build, used by the 256-bit control.)
inline void build_half_lut(const int8_t* a, int count, int16_t* out) {
    out[0] = 0;
    int len = 1;
    for (int t = 0; t < count; ++t) {
        const int16_t av = a[t];
        for (int s = len - 1; s >= 0; --s) {
            const int16_t base = out[s];
            out[3 * s + 0] = (int16_t)(base - av);
            out[3 * s + 1] = base;
            out[3 * s + 2] = (int16_t)(base + av);
        }
        len *= 3;
    }
}

// crow += acc16; acc16 = 0. (v1 flush, used by the 256-bit control.)
inline void flush_acc(int16_t* acc16, int32_t* crow, int N) {
    int j = 0;
    for (; j + 32 <= N; j += 32) {
        const __m512i a = _mm512_loadu_si512(acc16 + j);
        const __m512i lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a));
        const __m512i hi =
            _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a, 1));
        _mm512_storeu_si512(
            crow + j,
            _mm512_add_epi32(_mm512_loadu_si512(crow + j), lo));
        _mm512_storeu_si512(
            crow + j + 16,
            _mm512_add_epi32(_mm512_loadu_si512(crow + j + 16), hi));
        _mm512_storeu_si512(acc16 + j, _mm512_setzero_si512());
    }
    for (; j < N; ++j) {
        crow[j] += acc16[j];
        acc16[j] = 0;
    }
}

}  // namespace

void lut_mm_avx512_256(const int8_t* A, const int8_t* P, int M, int K, int N,
                       int32_t* C) {
    const int G = K / 5;
    constexpr int kFlushGroups = 32;  // 32 * 640 = 20480 < 32767

    alignas(64) int16_t idx_h[4][32], idx_l[4][32];
    for (int k = 0; k < 4; ++k) {
        for (int t = 0; t < 32; ++t) {
            const int n = 121 + 32 * k + t;
            idx_h[k][t] = (int16_t)(n / 9);
            idx_l[k][t] = (int16_t)(n % 9);
        }
    }
    __m512i vidx_h[4], vidx_l[4];
    for (int k = 0; k < 4; ++k) {
        vidx_h[k] = _mm512_loadu_si512(idx_h[k]);
        vidx_l[k] = _mm512_loadu_si512(idx_l[k]);
    }

    alignas(64) int16_t h_arr[32] = {0};  // 27 entries used, rest stays 0
    alignas(64) int16_t l_arr[32] = {0};  // 9 entries used, rest stays 0
    alignas(64) int16_t tbuf[128];
    const __m256i zero = _mm256_setzero_si256();
    const __m256i c32 = _mm256_set1_epi16(32);
    const __m256i c64 = _mm256_set1_epi16(64);

    std::vector<int16_t> acc16_buf((size_t)N, 0);
    int16_t* acc16 = acc16_buf.data();

    for (int i = 0; i < M; ++i) {
        const int8_t* a_row = A + (size_t)i * K;
        int32_t* crow = C + (size_t)i * N;
        std::memset(crow, 0, (size_t)N * sizeof(int32_t));
        for (int g = 0; g < G; ++g) {
            const int8_t* a = a_row + 5 * g;
            build_half_lut(a, 3, h_arr);
            build_half_lut(a + 3, 2, l_arr);
            const __m512i hv = _mm512_loadu_si512(h_arr);
            const __m512i lv = _mm512_loadu_si512(l_arr);
            for (int k = 0; k < 4; ++k) {
                _mm512_store_si512(
                    tbuf + 32 * k,
                    _mm512_add_epi16(
                        _mm512_permutexvar_epi16(vidx_h[k], hv),
                        _mm512_permutexvar_epi16(vidx_l[k], lv)));
            }
            __m256i T[8];
            for (int k = 0; k < 8; ++k)
                T[k] = _mm256_load_si256((const __m256i*)(tbuf + 16 * k));

            const int8_t* pw = P + (size_t)g * N;
            int j = 0;
            for (; j + 16 <= N; j += 16) {
                const __m128i pb =
                    _mm_loadu_si128((const __m128i*)(pw + j));
                const __m256i v = _mm256_cvtepi8_epi16(pb);
                const __mmask16 kneg = _mm256_movepi16_mask(v);
                const __m256i m = _mm256_abs_epi16(v);
                const __mmask16 k5 = _mm256_test_epi16_mask(m, c32);
                const __mmask16 k6 = _mm256_test_epi16_mask(m, c64);
                const __m256i r0 = _mm256_permutex2var_epi16(T[0], m, T[1]);
                const __m256i r1 = _mm256_permutex2var_epi16(T[2], m, T[3]);
                const __m256i r2 = _mm256_permutex2var_epi16(T[4], m, T[5]);
                const __m256i r3 = _mm256_permutex2var_epi16(T[6], m, T[7]);
                const __m256i ra = _mm256_mask_blend_epi16(k5, r0, r1);
                const __m256i rb = _mm256_mask_blend_epi16(k5, r2, r3);
                __m256i r = _mm256_mask_blend_epi16(k6, ra, rb);
                r = _mm256_mask_sub_epi16(r, kneg, zero, r);
                _mm256_storeu_si256(
                    (__m256i*)(acc16 + j),
                    _mm256_add_epi16(
                        _mm256_loadu_si256((const __m256i*)(acc16 + j)), r));
            }
            for (; j < N; ++j) {
                const int v = pw[j];
                const int n = 121 + (v < 0 ? -v : v);
                const int16_t val = (int16_t)(h_arr[n / 9] + l_arr[n % 9]);
                acc16[j] = (int16_t)(acc16[j] + (v < 0 ? -val : val));
            }
            if ((g + 1) % kFlushGroups == 0 || g + 1 == G)
                flush_acc(acc16, crow, N);
        }
    }
}
