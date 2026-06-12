// SPDX-License-Identifier: GPL-3.0-or-later
#include "ternary_mm.h"

#include <immintrin.h>

#include <algorithm>
#include <cstring>

// Dense AVX-512 VNNI baseline for C = A * W with int8 activations and
// ternary int8 weights. We store weights as bytes, not 5-trit packed bytes.
// The only layout change is K4 x N16 interleaving:
//
//   WV[g, jb, 4*c + t] = W[4*g + t, 16*jb + c]
//
// which matches vpdpbusd's per-int32-lane dot-product grouping. Since
// vpdpbusd is uint8*sint8, each activation byte is biased with xor 0x80
// and the output starts at -128 * sum_k W[k,j].

namespace {

int ceil_div(int x, int y) { return (x + y - 1) / y; }

__mmask16 lane_mask16(int cols) {
    return cols == 16 ? (__mmask16)0xFFFF : (__mmask16)((1u << cols) - 1u);
}

uint32_t biased_activation_word4(const int8_t* a) {
    uint32_t word;
    std::memcpy(&word, a, sizeof(word));
    return word ^ 0x80808080u;
}

uint32_t biased_activation_word_tail(const int8_t* a, int valid) {
    uint32_t word = 0;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&word);
    for (int t = 0; t < valid; ++t) {
        bytes[t] = (uint8_t)a[t] ^ 0x80u;
    }
    return word;
}

template <int R>
void vnni_rows(const int8_t* __restrict A, const int8_t* __restrict WV,
               const int32_t* __restrict col_sums, int K, int N,
               int32_t* __restrict C) {
    const int full_kgroups = K / 4;
    const int k_tail = K & 3;
    const int nblocks = ceil_div(N, 16);
    const __m512i neg_128 = _mm512_set1_epi32(-128);

    for (int jb = 0; jb < nblocks; ++jb) {
        const int j0 = jb * 16;
        const int cols = std::min(16, N - j0);
        const __mmask16 mask = lane_mask16(cols);
        const __m512i sums =
            cols == 16 ? _mm512_loadu_si512(col_sums + j0)
                       : _mm512_maskz_loadu_epi32(mask, col_sums + j0);
        const __m512i bias = _mm512_mullo_epi32(sums, neg_128);

        __m512i acc[R];
        for (int r = 0; r < R; ++r) acc[r] = bias;

        int g = 0;
        for (; g < full_kgroups; ++g) {
            const __m512i wv =
                _mm512_loadu_si512(WV + ((size_t)g * nblocks + jb) * 64);
            for (int r = 0; r < R; ++r) {
                const uint32_t aw =
                    biased_activation_word4(A + (size_t)r * K + 4 * g);
                const __m512i av = _mm512_set1_epi32((int)aw);
                acc[r] = _mm512_dpbusd_epi32(acc[r], av, wv);
            }
        }

        if (k_tail) {
            const __m512i wv =
                _mm512_loadu_si512(WV + ((size_t)g * nblocks + jb) * 64);
            for (int r = 0; r < R; ++r) {
                const uint32_t aw = biased_activation_word_tail(
                    A + (size_t)r * K + 4 * g, k_tail);
                const __m512i av = _mm512_set1_epi32((int)aw);
                acc[r] = _mm512_dpbusd_epi32(acc[r], av, wv);
            }
        }

        for (int r = 0; r < R; ++r) {
            int32_t* out = C + (size_t)r * N + j0;
            if (cols == 16) {
                _mm512_storeu_si512(out, acc[r]);
            } else {
                _mm512_mask_storeu_epi32(out, mask, acc[r]);
            }
        }
    }
}

}  // namespace

std::size_t vnni_weight_bytes(int K, int N) {
    return (std::size_t)ceil_div(K, 4) * ceil_div(N, 16) * 64;
}

void pack_weights_vnni(const int8_t* W, int K, int N, int8_t* WV,
                       int32_t* col_sums) {
    const int kgroups = ceil_div(K, 4);
    const int nblocks = ceil_div(N, 16);
    std::memset(WV, 0, vnni_weight_bytes(K, N));
    std::memset(col_sums, 0, (size_t)N * sizeof(int32_t));

    for (int g = 0; g < kgroups; ++g) {
        for (int jb = 0; jb < nblocks; ++jb) {
            const int j0 = jb * 16;
            const int cols = std::min(16, N - j0);
            int8_t* dst = WV + ((size_t)g * nblocks + jb) * 64;
            for (int c = 0; c < cols; ++c) {
                for (int t = 0; t < 4; ++t) {
                    const int k = 4 * g + t;
                    if (k >= K) continue;
                    const int8_t w = W[(size_t)k * N + j0 + c];
                    dst[4 * c + t] = w;
                    col_sums[j0 + c] += w;
                }
            }
        }
    }
}

void naive_mm_avx512_vnni(const int8_t* __restrict A,
                          const int8_t* __restrict WV,
                          const int32_t* __restrict col_sums, int M, int K,
                          int N, int32_t* __restrict C) {
    int i = 0;
    for (; i + 8 <= M; i += 8) {
        vnni_rows<8>(A + (size_t)i * K, WV, col_sums, K, N,
                     C + (size_t)i * N);
    }
    for (; i + 4 <= M; i += 4) {
        vnni_rows<4>(A + (size_t)i * K, WV, col_sums, K, N,
                     C + (size_t)i * N);
    }
    for (; i + 2 <= M; i += 2) {
        vnni_rows<2>(A + (size_t)i * K, WV, col_sums, K, N,
                     C + (size_t)i * N);
    }
    for (; i < M; ++i) {
        vnni_rows<1>(A + (size_t)i * K, WV, col_sums, K, N,
                     C + (size_t)i * N);
    }
}
