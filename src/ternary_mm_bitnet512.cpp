// SPDX-License-Identifier: GPL-3.0-or-later
// Derived from microsoft/BitNet's generated TL2 kernels (MIT License,
// Copyright (c) Microsoft Corporation); the widening is this project's.
//
// A direct 512-bit widening of BitNet's TL2 three-trit sweep, so the
// "width vs design" comparison has a measured TL2@512 instead of an
// estimate. The packed weight blob, LUT construction, sign convention and
// two-trit tail are unchanged (the latter two run BitNet's original
// 256-bit code via ternary_mm_bitnet.cpp); only three_tbl_impl is widened.
//
// The widening exploits their layout directly: consecutive index-byte
// positions of a 32-row block are adjacent in memory, so one 64-byte load
// covers a pair of positions. vpshufb zmm shuffles per 128-bit lane, so
// the pair's two different 16-entry tables are placed in lanes 0-1 and
// 2-3 respectively (one vshufi32x4 each from the two contiguous 64-byte
// table records). Sign extraction needs a per-lane shift amount (the two
// positions use different bit offsets), which vpsllvw provides. At the end
// the two 256-bit halves — the even and odd positions' partial sums for
// the same 32 rows — fold with one add, and the original widen/store
// epilogue runs unchanged.
#include <immintrin.h>

#include <cstring>
#include <vector>

#include "ternary_mm.h"

namespace {

template <int BMt, int BBK>
void three_tbl_impl_512(int32_t* c, const int8_t* lut, const uint8_t* a,
                        const uint8_t* sign) {
    constexpr int KK = BBK / 3;  // 3-trit groups per row per BBK block
    const __m512i vec_mask = _mm512_set1_epi8(0x0f);

    // Shift counts for sign-bit extraction: low 256 bits serve position
    // 2*jj (shift 4*(2jj)+t), high 256 bits position 2*jj+1.
    __m512i cnt[2][4];
    for (int jj = 0; jj < 2; ++jj) {
        for (int t = 0; t < 4; ++t) {
            cnt[jj][t] = _mm512_inserti64x4(
                _mm512_castsi256_si512(
                    _mm256_set1_epi16((short)(4 * (2 * jj) + t))),
                _mm256_set1_epi16((short)(4 * (2 * jj + 1) + t)), 1);
        }
    }

    for (int i = 0; i < BMt; i += 32) {
        __m512i vec_as[KK / 4];
        for (int ai = 0; ai < KK / 4; ++ai)
            vec_as[ai] =
                _mm512_loadu_si512(a + (size_t)i * KK / 2 + ai * 64);
        const uint8_t* sgnrow = sign + (size_t)i * KK / 8;
        __m512i c0 = _mm512_setzero_si512();
        __m512i c1 = _mm512_setzero_si512();
        for (int k = 0; k < KK / 8; ++k) {
            const __m512i vsign = _mm512_broadcast_i64x4(
                _mm256_loadu_si256((const __m256i*)(sgnrow + k * 32)));
            for (int jj = 0; jj < 2; ++jj) {
                const __m512i va = vec_as[k * 2 + jj];
                // 64-byte table records [k1 k2 k3 k4] for the two positions
                const __m512i zj0 = _mm512_loadu_si512(
                    lut + (size_t)k * 256 + (2 * jj + 0) * 64);
                const __m512i zj1 = _mm512_loadu_si512(
                    lut + (size_t)k * 256 + (2 * jj + 1) * 64);
                const __m512i tk1 = _mm512_shuffle_i32x4(zj0, zj1, 0x00);
                const __m512i tk2 = _mm512_shuffle_i32x4(zj0, zj1, 0x55);
                const __m512i tk3 = _mm512_shuffle_i32x4(zj0, zj1, 0xAA);
                const __m512i tk4 = _mm512_shuffle_i32x4(zj0, zj1, 0xFF);
                const __m512i top = _mm512_and_si512(
                    _mm512_srli_epi16(va, 4), vec_mask);
                const __m512i bot = _mm512_and_si512(va, vec_mask);
                const __m512i fir = _mm512_shuffle_epi8(tk1, top);
                const __m512i sec = _mm512_shuffle_epi8(tk2, top);
                const __m512i bfir = _mm512_shuffle_epi8(tk3, bot);
                const __m512i bsec = _mm512_shuffle_epi8(tk4, bot);
                const __m512i s_lh = _mm512_srai_epi16(
                    _mm512_sllv_epi16(vsign, cnt[jj][0]), 15);
                const __m512i s_ll = _mm512_srai_epi16(
                    _mm512_sllv_epi16(vsign, cnt[jj][1]), 15);
                const __m512i s_rh = _mm512_srai_epi16(
                    _mm512_sllv_epi16(vsign, cnt[jj][2]), 15);
                const __m512i s_rl = _mm512_srai_epi16(
                    _mm512_sllv_epi16(vsign, cnt[jj][3]), 15);
                const __m512i top_hi = _mm512_xor_si512(
                    _mm512_add_epi16(_mm512_unpacklo_epi8(fir, sec), s_lh),
                    s_lh);
                const __m512i top_lo = _mm512_xor_si512(
                    _mm512_add_epi16(_mm512_unpackhi_epi8(fir, sec), s_ll),
                    s_ll);
                const __m512i bot_hi = _mm512_xor_si512(
                    _mm512_add_epi16(_mm512_unpacklo_epi8(bfir, bsec), s_rh),
                    s_rh);
                const __m512i bot_lo = _mm512_xor_si512(
                    _mm512_add_epi16(_mm512_unpackhi_epi8(bfir, bsec), s_rl),
                    s_rl);
                c0 = _mm512_add_epi16(c0, _mm512_add_epi16(top_hi, bot_hi));
                c1 = _mm512_add_epi16(c1, _mm512_add_epi16(top_lo, bot_lo));
            }
        }
        // Fold even/odd-position halves, then BitNet's widen/store epilogue.
        const __m256i vc0 = _mm256_add_epi16(_mm512_castsi512_si256(c0),
                                             _mm512_extracti64x4_epi64(c0, 1));
        const __m256i vc1 = _mm256_add_epi16(_mm512_castsi512_si256(c1),
                                             _mm512_extracti64x4_epi64(c1, 1));
        __m256i gc0 = _mm256_loadu_si256((__m256i*)(c + i));
        __m256i gc1 = _mm256_loadu_si256((__m256i*)(c + i + 8));
        __m256i gc2 = _mm256_loadu_si256((__m256i*)(c + i + 16));
        __m256i gc3 = _mm256_loadu_si256((__m256i*)(c + i + 24));
        gc0 = _mm256_add_epi32(
            gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vc0)));
        gc1 = _mm256_add_epi32(
            gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vc0, 1)));
        gc2 = _mm256_add_epi32(
            gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vc1)));
        gc3 = _mm256_add_epi32(
            gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vc1, 1)));
        _mm256_storeu_si256((__m256i*)(c + i), gc0);
        _mm256_storeu_si256((__m256i*)(c + i + 8), gc1);
        _mm256_storeu_si256((__m256i*)(c + i + 16), gc2);
        _mm256_storeu_si256((__m256i*)(c + i + 24), gc3);
    }
}

template <int BMt, int BBK, int K3>
void three_qgemm_lut_512(const uint8_t* A, const uint8_t* sign,
                         const int8_t* LUT, int32_t* Ct) {
    alignas(64) int32_t CBits[BMt];
    std::memset(CBits, 0, sizeof(CBits));
    for (int k_outer = 0; k_outer < K3 / BBK; ++k_outer) {
        three_tbl_impl_512<BMt, BBK>(
            CBits, LUT + (size_t)k_outer * BBK / 3 * 32,
            A + (size_t)k_outer * BBK / 3 / 2 * BMt,
            sign + (size_t)k_outer * BBK / 3 / 8 * BMt);
    }
    for (int i = 0; i < BMt; ++i) Ct[i] = CBits[i];
}

}  // namespace

void lut_mm_bitnet_tl2_512(const float* B, const uint8_t* qw, int M, int K,
                           int N, int32_t* C) {
    const int three_k = K - K % 96;
    const int two_k = K - three_k;

    std::vector<int8_t> qlut3((size_t)three_k / 3 * 32);
    std::vector<int8_t> qlut2((size_t)two_k / 2 * 32);
    float lut_scales[1];
    float weight_scales[1] = {1.0f};

    const size_t idx3_tile = (size_t)256 * three_k / 6;
    const uint8_t* sgn = qw + (size_t)N * three_k / 6;
    const size_t sgn_tile = (size_t)256 * three_k / 24;
    const uint8_t* idx2 = sgn + (size_t)N * three_k / 24;
    const size_t idx2_tile = (size_t)256 * two_k / 4;

    for (int i = 0; i < M; ++i) {
        const float* b = B + (size_t)i * K;
        int32_t* crow = C + (size_t)i * N;

        bitnet_tl2_prep_row(b, K, qlut3.data(), qlut2.data(), lut_scales);

        for (int t = 0; t < N / 256; ++t) {
            int32_t* ct = crow + (size_t)t * 256;
            if (K == 2080) {
                three_qgemm_lut_512<256, 96, 2016>(
                    qw + t * idx3_tile, sgn + t * sgn_tile, qlut3.data(), ct);
            } else {
                three_qgemm_lut_512<256, 96, 4128>(
                    qw + t * idx3_tile, sgn + t * sgn_tile, qlut3.data(), ct);
            }
            bitnet_tl2_two_qgemm(K, idx2 + t * idx2_tile, qlut2.data(),
                                 weight_scales, lut_scales, ct);
        }

        // Float result left in C (BitNet's native output); the int32
        // readback runs outside timing via bitnet_tl2_to_int32.
    }
}
