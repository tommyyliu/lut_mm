#ifndef LUT_MM_H
#define LUT_MM_H

#include <vector>
#include <cstdint>
#include "weight_packing.h"

// Pack 5 ternary weights (-1, 0, 1) into a single byte using ternary encoding
std::vector<std::vector<int8_t>> pack_weights_lut(const std::vector<std::vector<int8_t>>& weights);

// Standard matrix multiplication: activations (M×K) × weights (K×N) = result (M×N)
std::vector<std::vector<int>> basic_mm(const std::vector<std::vector<int8_t>>& activations,
                                      const std::vector<std::vector<int8_t>>& weights);

// Generate lookup table for activations to speed up matrix multiplication
std::vector<std::vector<std::vector<int>>> generate_lut(const std::vector<std::vector<int8_t>>& activations);

// Matrix multiplication using lookup tables and packed weights
template<typename T>
std::vector<std::vector<int>> lut_mm(const std::vector<std::vector<int8_t>>& activations,
                                    const std::vector<std::vector<T>>& packed_weights);

#endif // LUT_MM_H
