// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ternary_mm.h"

#include <cstring>

void pack_weights(const int8_t* W, int K, int N, int8_t* P) {
    const int G = K / 5;
    for (int g = 0; g < G; ++g) {
        for (int j = 0; j < N; ++j) {
            int v = 0;
            for (int t = 0; t < 5; ++t) v = v * 3 + W[(5 * g + t) * N + j];
            P[(size_t)g * N + j] = (int8_t)v;
        }
    }
}

void naive_mm(const int8_t* A, const int8_t* W, int M, int K, int N,
              int32_t* C) {
    for (int i = 0; i < M; ++i) {
        int32_t* out = C + (size_t)i * N;
        std::memset(out, 0, (size_t)N * sizeof(int32_t));
        for (int k = 0; k < K; ++k) {
            const int32_t a = A[(size_t)i * K + k];
            if (a == 0) continue;
            const int8_t* w = W + (size_t)k * N;
            for (int j = 0; j < N; ++j) out[j] += a * w[j];
        }
    }
}
