// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstddef>
#include <cstdint>

// Ternary-weight matrix multiplication via packed weights and lookup tables.
//
// Shapes and layouts (all row-major):
//   activations A : M x K, int8
//   weights W     : K x N, int8, every element in {-1, 0, 1}
//   packed P      : (K/5) x N, int8. P[g][j] encodes the 5 weights
//                   W[5g+0..5g+4][j] as a balanced-ternary number
//                   sum_t W[5g+t][j] * 3^(4-t), range [-121, 121].
//   output C      : M x N, int32
//
// K must be a multiple of 5.

// Compress W (K x N) into P ((K/5) x N), 5 ternary weights per byte.
void pack_weights(const int8_t* W, int K, int N, int8_t* P);

// Ground-truth dense GEMM on the unpacked weights: C = A * W.
void naive_mm(const int8_t* A, const int8_t* W, int M, int K, int N,
              int32_t* C);

// Dense one-byte-per-weight AVX-512 VNNI baseline. The VNNI layout is not
// compressed: it only interleaves each K block of 4 weights for 16 columns,
// so vpdpbusd can compute 16 int32 dot products per instruction. col_sums
// holds sum_k W[k, j], used to undo the unsigned-activation bias required by
// AVX-512 VNNI's uint8*sint8 dot-product instruction.
std::size_t vnni_weight_bytes(int K, int N);
void pack_weights_vnni(const int8_t* W, int K, int N, int8_t* WV,
                       int32_t* col_sums);
void dense_mm_avx512_vnni(const int8_t* __restrict A,
                          const int8_t* __restrict WV,
                          const int32_t* __restrict col_sums, int M, int K,
                          int N, int32_t* __restrict C);

// LUT-based GEMM on packed weights: AVX-512 (F+BW) vpermt2w lookups over
// the 122 mirror-reduced magnitudes. See ternary_mm_avx512.cpp for the
// full design.
void lut_mm_avx512(const int8_t* __restrict A, const int8_t* __restrict P,
                   int M, int K, int N, int32_t* __restrict C);
void lut_mm_avx512_mt(const int8_t* A, const int8_t* P, int M, int K, int N,
                      int32_t* C, int num_threads);

// microsoft/BitNet TL2 kernels (AVX2), for comparison. B is the float copy
// of the activations; qw is the weight blob from tools/pack_tl2.py. Only
// shapes baked into the generated kernels are supported. The adapter forces
// unit scales and leaves exact int32 sums in C instead of BitNet's native
// dequantized-float output.
bool bitnet_tl2_supported(int K, int N);
void lut_mm_bitnet_tl2(const float* B, const uint8_t* qw, int M, int K,
                       int N, int32_t* C);
void lut_mm_bitnet_tl2_mt(const float* B, const uint8_t* qw, int M, int K,
                          int N, int32_t* C, int num_threads);

// Our 512-bit widening of the TL2 three-trit sweep (same packed blob);
// measured counterpart to the "TL2 scaled to 512 bits" estimate.
void lut_mm_bitnet_tl2_512(const float* B, const uint8_t* qw, int M, int K,
                           int N, int32_t* C);

// Internal: TL2 pieces shared between the AVX2 adapter and the 512 port.
void bitnet_tl2_prep_row(const float* b, int K, int8_t* qlut3, int8_t* qlut2,
                         float* lut_scales);
void bitnet_tl2_two_qgemm_int32(int K, const uint8_t* idx2_tile,
                                const int8_t* qlut2, int32_t* ct);
