#include <immintrin.h>

#include <cstring>

#include "lut_build.h"
#include "ternary_mm.h"

namespace {

// Gathers 8 table values for packed bytes pw[j..j+7]. The bytes are signed
// indices in [-121, 121], so the caller passes the table center (lut + 121)
// as the base; vpgatherdd indices are signed, which makes the +121 bias free.
inline __m256i gather_group(const int32_t* lut_center, const int8_t* pw,
                            int j) {
    const __m128i bytes = _mm_loadl_epi64((const __m128i*)(pw + j));
    return _mm256_i32gather_epi32(lut_center, _mm256_cvtepi8_epi32(bytes), 4);
}

}  // namespace

// For each row i and weight group g, build the 243-entry dot-product table
// for that group's 5 activations, then sweep the packed row: 8 outputs per
// iteration, with vpgatherdd doing the 8 table lookups. Four groups are
// processed per sweep so their gathers overlap and the accumulator
// read-modify-write is amortized; the four tables (4 KB) stay L1-resident.
void lut_mm_avx2(const int8_t* A, const int8_t* P, int M, int K, int N,
                 int32_t* C) {
    const int G = K / 5;
    alignas(32) int32_t lut[4][243];
    for (int i = 0; i < M; ++i) {
        const int8_t* a_row = A + (size_t)i * K;
        int32_t* acc = C + (size_t)i * N;
        std::memset(acc, 0, (size_t)N * sizeof(int32_t));
        int g = 0;
        for (; g + 4 <= G; g += 4) {
            for (int u = 0; u < 4; ++u)
                build_group_lut(a_row + 5 * (g + u), lut[u]);
            const int8_t* pw0 = P + (size_t)(g + 0) * N;
            const int8_t* pw1 = P + (size_t)(g + 1) * N;
            const int8_t* pw2 = P + (size_t)(g + 2) * N;
            const int8_t* pw3 = P + (size_t)(g + 3) * N;
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                const __m256i v0 = gather_group(lut[0] + 121, pw0, j);
                const __m256i v1 = gather_group(lut[1] + 121, pw1, j);
                const __m256i v2 = gather_group(lut[2] + 121, pw2, j);
                const __m256i v3 = gather_group(lut[3] + 121, pw3, j);
                const __m256i sum = _mm256_add_epi32(
                    _mm256_add_epi32(v0, v1), _mm256_add_epi32(v2, v3));
                _mm256_storeu_si256(
                    (__m256i*)(acc + j),
                    _mm256_add_epi32(
                        _mm256_loadu_si256((const __m256i*)(acc + j)), sum));
            }
            for (; j < N; ++j) {
                acc[j] += lut[0][pw0[j] + 121] + lut[1][pw1[j] + 121] +
                          lut[2][pw2[j] + 121] + lut[3][pw3[j] + 121];
            }
        }
        for (; g < G; ++g) {
            build_group_lut(a_row + 5 * g, lut[0]);
            const int8_t* pw = P + (size_t)g * N;
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                const __m256i v = gather_group(lut[0] + 121, pw, j);
                _mm256_storeu_si256(
                    (__m256i*)(acc + j),
                    _mm256_add_epi32(
                        _mm256_loadu_si256((const __m256i*)(acc + j)), v));
            }
            for (; j < N; ++j) acc[j] += lut[0][pw[j] + 121];
        }
    }
}
