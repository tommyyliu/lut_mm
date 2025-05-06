#ifndef LUT_MM_OPTIMIZED_H
#define LUT_MM_OPTIMIZED_H

#include <vector>
#include <cstdint>
#include "weight_packing.h"

// Optimized standard matrix multiplication: activations (M×K) × weights (K×N) = result (M×N)
std::vector<std::vector<int>> basic_mm_optimized(const std::vector<std::vector<int8_t>>& activations,
                                               const std::vector<std::vector<int8_t>>& weights);

// Optimized lookup table generation for activations
std::vector<std::vector<std::vector<int>>> generate_lut_optimized(const std::vector<std::vector<int8_t>>& activations);

// Optimized matrix multiplication using lookup tables and packed weights
template<typename T>
std::vector<std::vector<int>> lut_mm_optimized(const std::vector<std::vector<int8_t>>& activations,
                                             const std::vector<std::vector<T>>& packed_weights);

#endif // LUT_MM_OPTIMIZED_H
