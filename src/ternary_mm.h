#pragma once

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

// LUT-based GEMM on packed weights, scalar implementation.
void lut_mm_scalar(const int8_t* A, const int8_t* P, int M, int K, int N,
                   int32_t* C);

// LUT-based GEMM on packed weights, AVX2 implementation (vpgatherdd lookups).
void lut_mm_avx2(const int8_t* A, const int8_t* P, int M, int K, int N,
                 int32_t* C);

// LUT-based GEMM on packed weights, AVX-512 implementation (vpermt2w
// lookups over the 122 mirror-reduced magnitudes). Requires AVX-512F+BW.
void lut_mm_avx512(const int8_t* __restrict A, const int8_t* __restrict P,
                   int M, int K, int N, int32_t* __restrict C);

// Width-control variant of lut_mm_avx512: identical design, 256-bit lookup
// sweep (vpermt2w on ymm; requires AVX-512VL). Exists to separate vector
// width from format when comparing against BitNet's 256-bit TL2.
void lut_mm_avx512_256(const int8_t* A, const int8_t* P, int M, int K, int N,
                       int32_t* C);

// microsoft/BitNet TL2 kernels (AVX2), for comparison. B is the float copy
// of the activations; qw is the weight blob from tools/pack_tl2.py. Only
// shapes baked into the generated kernels are supported.
bool bitnet_tl2_supported(int K, int N);
void lut_mm_bitnet_tl2(const float* B, const uint8_t* qw, int M, int K,
                       int N, int32_t* C);

// Our 512-bit widening of the TL2 three-trit sweep (same packed blob);
// measured counterpart to the "TL2 scaled to 512 bits" estimate.
void lut_mm_bitnet_tl2_512(const float* B, const uint8_t* qw, int M, int K,
                           int N, int32_t* C);

// Internal: TL2 pieces shared between the AVX2 adapter and the 512 port.
void bitnet_tl2_prep_row(const float* b, int K, int8_t* qlut3, int8_t* qlut2,
                         float* lut_scales);
void bitnet_tl2_two_qgemm(int K, const uint8_t* idx2_tile,
                          const int8_t* qlut2, const float* scales,
                          const float* lut_scales, void* ct);
