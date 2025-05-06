#include "lut_mm.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <iostream>

#include "weight_packing.h"
#include "permutations.h"

std::vector<std::vector<int8_t>> pack_weights_lut(const std::vector<std::vector<int8_t>>& weights) {
    int N = weights.size();    // Number of rows
    int K = weights[0].size(); // Number of columns

    // Calculate the number of packed columns
    int packed_cols = (K + GROUP_SIZE - 1) / GROUP_SIZE; // Ceiling division

    // Initialize packed weights matrix
    std::vector<std::vector<int8_t>> packed_weights(N, std::vector<int8_t>(packed_cols, 0));

    // Pack weights for each row
    for (int i = 0; i < N; i++) {
        // Process 5 weights at a time
        for (int j = 0; j < K; j += GROUP_SIZE) {
            int packed_value = 0;

            // Pack up to 5 weights (handle edge case where K is not divisible by 5)
            for (int k = 0; k < GROUP_SIZE && j + k < K; k++) {
                packed_value += weights[i][j + k] * DIGIT_WEIGHTS[k];
            }

            packed_weights[i][j / GROUP_SIZE] = packed_value;
        }
    }

    return packed_weights;
}


std::vector<std::vector<int>> basic_mm(const std::vector<std::vector<int8_t>>& activations,
                                     const std::vector<std::vector<int8_t>>& weights) {
    int M = activations.size();    // Number of rows in activations
    int K = activations[0].size(); // Number of columns in activations
    int N = weights[0].size();     // Number of columns in weights

    // Initialize result matrix
    std::vector<std::vector<int>> result(M, std::vector<int>(N, 0));

    // Perform matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                result[i][j] += activations[i][k] * weights[k][j];
            }
        }
    }

    return result;
}

std::vector<std::vector<std::vector<int>>> generate_lut(const std::vector<std::vector<int8_t>>& activations) {
    int M = activations.size();    // Number of rows in activations
    int K = activations[0].size(); // Number of columns in activations

    // Get non-negative permutations
    auto non_negative_perms = get_non_negative_perms();
    int P = non_negative_perms.size();

    // Calculate number of chunks
    int num_chunks = (K + GROUP_SIZE - 1) / GROUP_SIZE; // Ceiling division

    // Initialize lookup table
    std::vector<std::vector<std::vector<int>>> lut(M,
        std::vector<std::vector<int>>(num_chunks,
            std::vector<int>(P, 0)));

    // Fill the lookup table
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j += GROUP_SIZE) {
            for (int p = 0; p < P; p++) {
                int dot_product = 0;

                // Calculate dot product for this chunk
                for (int k = 0; k < GROUP_SIZE && j + k < K; k++) {
                    dot_product += activations[i][j + k] * non_negative_perms[p][k];
                }

                lut[i][j / GROUP_SIZE][p] = dot_product;
            }
        }
    }

    return lut;
}

// Template implementation
template<typename T>
std::vector<std::vector<int>> lut_mm(const std::vector<std::vector<int8_t>>& activations,
                                   const std::vector<std::vector<T>>& packed_weights) {
    // Check for empty inputs
    if (activations.empty() || packed_weights.empty()) {
        return {}; // Return empty result for empty inputs
    }

    int M = activations.size();        // Number of rows in activations
    int K = activations[0].size();     // Number of columns in activations
    int N = packed_weights.size();     // Number of rows in packed_weights (weight vectors)

    // Calculate number of chunks in the activations
    int num_chunks = (K + GROUP_SIZE - 1) / GROUP_SIZE; // Ceiling division

    // Generate lookup table
    auto lut = generate_lut(activations);

    // Initialize result matrix
    std::vector<std::vector<int>> result(M, std::vector<int>(N, 0));

    // Perform lookup-based matrix multiplication
    for (int i = 0; i < M; i++) {         // For each activation row
        for (int j = 0; j < N; j++) {     // For each weight vector
            for (int c = 0; c < num_chunks; c++) {
                // Get the packed weight value
                T packed_value = packed_weights[j][c];

                // Get the absolute value and sign
                int8_t sign;
                int8_t abs_value;

                // Handle different types
                if constexpr (std::is_same<T, sign_magnitude_8_t>::value) {
                    sign = sm_sign(packed_value);
                    abs_value = sm_magnitude(packed_value);
                } else {
                    sign = (packed_value >= 0) ? 1 : -1;
                    abs_value = std::abs(packed_value);
                }

                // Debug output for the first few values
                if (i == 0 && j == 0 && c < 3) {
                    std::cout << "Debug: packed_value=" << static_cast<int>(packed_value)
                              << ", sign=" << static_cast<int>(sign)
                              << ", abs_value=" << static_cast<int>(abs_value) << std::endl;
                }

                // Look up the dot product in the LUT and apply the sign
                // The LUT contains dot products for non-negative permutations
                // Make sure the abs_value is within the valid range for the LUT
                if (i < lut.size() && c < lut[i].size() && abs_value < lut[i][c].size()) {
                    int lut_value = lut[i][c][abs_value];
                    int contribution = lut_value * sign;
                    result[i][j] += contribution;

                    // Debug output for the first few values
                    if (i == 0 && j == 0 && c < 3) {
                        std::cout << "  chunk " << c << ": lut_value=" << lut_value
                                  << ", sign=" << static_cast<int>(sign)
                                  << ", contribution=" << contribution
                                  << ", running sum=" << result[i][j] << std::endl;
                    }
                }
            }
        }
    }

    return result;
}

// Explicit template instantiations
template std::vector<std::vector<int>> lut_mm<int8_t>(const std::vector<std::vector<int8_t>>& activations,
                                                    const std::vector<std::vector<int8_t>>& packed_weights);
template std::vector<std::vector<int>> lut_mm<sign_magnitude_8_t>(const std::vector<std::vector<int8_t>>& activations,
                                                     const std::vector<std::vector<sign_magnitude_8_t>>& packed_weights);