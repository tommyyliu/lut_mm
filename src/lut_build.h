#pragma once

#include <cstdint>

// Build the full 243-entry lookup table for one group of 5 activations:
// lut[v + 121] = dot(a[0..4], d[0..4]) for every ternary vector d in
// {-1,0,1}^5, where v is the balanced-ternary value of d (a[0] pairs with
// the most significant trit, matching pack_weights). A packed weight byte
// p therefore selects its dot product as lut[p + 121].
//
// Built incrementally one trit at a time: 3 + 9 + ... + 243 = 363 adds
// instead of 243 * 5 multiply-adds. The in-place descending loop is safe
// because step s writes only to [3s, 3s+2] and later iterations read only
// from [0, s-1].
inline void build_group_lut(const int8_t* a, int32_t* lut) {
    lut[0] = 0;
    int len = 1;
    for (int t = 0; t < 5; ++t) {
        const int32_t av = a[t];
        for (int s = len - 1; s >= 0; --s) {
            const int32_t base = lut[s];
            lut[3 * s + 0] = base - av;
            lut[3 * s + 1] = base;
            lut[3 * s + 2] = base + av;
        }
        len *= 3;
    }
}
