// SPDX-License-Identifier: GPL-3.0-or-later
// Includes (via src/generated/) kernel code from microsoft/BitNet,
// MIT License, Copyright (c) Microsoft Corporation.
//
// Adapter running microsoft/BitNet's TL2 AVX2 kernels (generated verbatim
// by tools/gen_bitnet_tl2.py into src/generated/) inside this harness.
//
// Their pipeline: float activations are absmax-quantized while building the
// per-3-trit-group LUTs (three_lut_ctor / two_lut_ctor), then qgemm sweeps
// nibble-packed magnitude indices + a separate sign bitfield with pshufb.
// We force the activation quantization scale to 1.0 — our activations are
// already int8, so every LUT entry and the final int32 sums are exact and
// the harness can require a bit-exact match against the naive GEMM. The
// per_tensor_quant scan is still executed so its cost stays in the timing.
//
// Weights must be prepacked with tools/pack_tl2.py. Layout (their python,
// per BitNet's "M" = our N): all tiles' 3-trit index nibbles, then all sign
// bitfields, then all 2-trit index nibbles; each section tiled by BM=256
// output rows.

#define GGML_BITNET_X86_TL2
#include "generated/bitnet-lut-kernels.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#include "ternary_mm.h"

bool bitnet_tl2_supported(int K, int N) {
    return (K == 2080 && N == 2048) || (K == 4160 && N == 4096);
}

// Shared with the 512-bit port (ternary_mm_bitnet512.cpp), which cannot
// include the generated header itself: that would duplicate its non-inline
// functions. The prep keeps BitNet's quant scan + LUT constructors; the
// two-trit tail reuses BitNet's AVX2 table sweep but leaves exact int32
// sums instead of running the generated float dequantization epilogue.
void bitnet_tl2_prep_row(const float* b, int K, int8_t* qlut3, int8_t* qlut2,
                         float* lut_scales) {
    bitnet_float_type* bm = const_cast<float*>(b);
    partial_max_reset(1, lut_scales);
    per_tensor_quant(K, lut_scales, bm);
    lut_scales[0] = 1.0f;  // exact mode
    if (K == 2080) {
        three_lut_ctor<2016>(qlut3, bm, lut_scales);
        two_lut_ctor<64>(qlut2, bm + 2016, lut_scales);
    } else if (K == 4160) {
        three_lut_ctor<4128>(qlut3, bm, lut_scales);
        two_lut_ctor<32>(qlut2, bm + 4128, lut_scales);
    }
}

void bitnet_tl2_two_qgemm_int32(int K, const uint8_t* idx2_tile,
                                const int8_t* qlut2, int32_t* ct) {
    if (K == 2080) {
        alignas(32) int32_t bits[BM2048_2080];
        std::memset(bits, 0, sizeof(bits));
        for (int k_outer = 0; k_outer < 64 / 32; ++k_outer) {
            two_tbl_impl2048_2080<1, 64>(
                bits,
                const_cast<int8_t*>(qlut2 + k_outer * BK2 / 2 * 32),
                const_cast<uint8_t*>(idx2_tile +
                                     k_outer * BK2 / 2 / 2 * BM2048_2080));
        }
        for (int i = 0; i < BM2048_2080; ++i) ct[i] += bits[i];
    } else if (K == 4160) {
        alignas(32) int32_t bits[BM4096_4160];
        std::memset(bits, 0, sizeof(bits));
        for (int k_outer = 0; k_outer < 32 / 32; ++k_outer) {
            two_tbl_impl4096_4160<1, 32>(
                bits,
                const_cast<int8_t*>(qlut2 + k_outer * BK2 / 2 * 32),
                const_cast<uint8_t*>(idx2_tile +
                                     k_outer * BK2 / 2 / 2 * BM4096_4160));
        }
        for (int i = 0; i < BM4096_4160; ++i) ct[i] += bits[i];
    }
}

void lut_mm_bitnet_tl2(const float* B, const uint8_t* qw, int M, int K,
                       int N, int32_t* C) {
    const int three_k = K - K % 96;
    const int two_k = K - three_k;

    std::vector<int8_t> qlut3((size_t)three_k / 3 * 32);
    std::vector<int8_t> qlut2((size_t)two_k / 2 * 32);
    bitnet_float_type lut_scales[1];

    const size_t idx3_tile = (size_t)256 * three_k / 6;
    const uint8_t* sgn = qw + (size_t)N * three_k / 6;
    const size_t sgn_tile = (size_t)256 * three_k / 24;
    const uint8_t* idx2 = sgn + (size_t)N * three_k / 24;
    const size_t idx2_tile = (size_t)256 * two_k / 4;

    for (int i = 0; i < M; ++i) {
        const float* b = B + (size_t)i * K;
        int32_t* crow = C + (size_t)i * N;

        bitnet_tl2_prep_row(b, K, qlut3.data(), qlut2.data(), lut_scales);

        if (K == 2080 && N == 2048) {
            for (int t = 0; t < N / 256; ++t) {
                int32_t* ct = crow + (size_t)t * 256;
                three_qgemm_lut_2048_2080<1>(
                    (void*)(qw + t * idx3_tile), (void*)(sgn + t * sgn_tile),
                    qlut3.data(), nullptr, lut_scales, ct);
                bitnet_tl2_two_qgemm_int32(K, idx2 + t * idx2_tile,
                                           qlut2.data(), ct);
            }
        } else if (K == 4160 && N == 4096) {
            for (int t = 0; t < N / 256; ++t) {
                int32_t* ct = crow + (size_t)t * 256;
                three_qgemm_lut_4096_4160<1>(
                    (void*)(qw + t * idx3_tile), (void*)(sgn + t * sgn_tile),
                    qlut3.data(), nullptr, lut_scales, ct);
                bitnet_tl2_two_qgemm_int32(K, idx2 + t * idx2_tile,
                                           qlut2.data(), ct);
            }
        }
    }
}

void lut_mm_bitnet_tl2_mt(const float* B, const uint8_t* qw, int M, int K,
                          int N, int32_t* C, int num_threads) {
    if (num_threads <= 1 || M <= 1) {
        lut_mm_bitnet_tl2(B, qw, M, K, N, C);
        return;
    }

    const int threads = std::min(num_threads, M);
    std::vector<std::thread> workers;
    workers.reserve((size_t)threads - 1);

    int start = 0;
    for (int t = 0; t < threads; ++t) {
        const int remaining_rows = M - start;
        const int remaining_threads = threads - t;
        const int rows = (remaining_rows + remaining_threads - 1) /
                         remaining_threads;
        const int row0 = start;
        start += rows;

        auto run_chunk = [=] {
            lut_mm_bitnet_tl2(B + (size_t)row0 * K, qw, rows, K, N,
                              C + (size_t)row0 * N);
        };
        if (t + 1 == threads) {
            run_chunk();
        } else {
            workers.emplace_back(run_chunk);
        }
    }

    for (auto& worker : workers) worker.join();
}
