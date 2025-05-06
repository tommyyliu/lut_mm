#include "lut_mm_optimized.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <thread>
#include <vector>
#include <future>

#include "weight_packing.h"
#include "permutations.h"
#include "profiling.h"

// Helper function to determine optimal number of threads
inline unsigned int get_optimal_thread_count() {
    unsigned int thread_count = std::thread::hardware_concurrency();
    return thread_count > 0 ? thread_count : 4; // Default to 4 if detection fails
}

std::vector<std::vector<int>> basic_mm_optimized(const std::vector<std::vector<int8_t>>& activations,
                                               const std::vector<std::vector<int8_t>>& weights) {
    PROFILE_FUNCTION(basic_mm_optimized);

    int M = activations.size();    // Number of rows in activations
    int K = activations[0].size(); // Number of columns in activations
    int N = weights[0].size();     // Number of columns in weights

    // Initialize result matrix
    std::vector<std::vector<int>> result(M, std::vector<int>(N, 0));

    // Cache blocking parameters
    const int BLOCK_SIZE = 32; // Adjust based on cache size

    // Parallelize the computation using multiple threads
    unsigned int num_threads = get_optimal_thread_count();
    std::vector<std::thread> threads;

    auto worker = [&](int start_row, int end_row) {
        PROFILE_FUNCTION(basic_mm_worker);
        // Loop tiling for better cache utilization
        for (int i = start_row; i < end_row; i++) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                    // Process a block
                    for (int j = jj; j < std::min(jj + BLOCK_SIZE, N); j++) {
                        int sum = result[i][j]; // Load result once

                        // Inner loop with manual unrolling for better instruction-level parallelism
                        for (int k = kk; k < std::min(kk + BLOCK_SIZE, K); k += 4) {
                            if (k + 3 < K) {
                                // Process 4 elements at once
                                sum += activations[i][k] * weights[k][j] +
                                       activations[i][k+1] * weights[k+1][j] +
                                       activations[i][k+2] * weights[k+2][j] +
                                       activations[i][k+3] * weights[k+3][j];
                            } else {
                                // Handle remaining elements
                                for (int k_remainder = k; k_remainder < std::min(kk + BLOCK_SIZE, K); k_remainder++) {
                                    sum += activations[i][k_remainder] * weights[k_remainder][j];
                                }
                                break;
                            }
                        }

                        result[i][j] = sum; // Store result once
                    }
                }
            }
        }
    };

    // Divide work among threads
    int rows_per_thread = M / num_threads;
    int start_row = 0;

    for (unsigned int t = 0; t < num_threads - 1; t++) {
        int end_row = start_row + rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
        start_row = end_row;
    }

    // Last thread handles remaining rows
    threads.emplace_back(worker, start_row, M);

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    return result;
}

std::vector<std::vector<std::vector<int>>> generate_lut_optimized(const std::vector<std::vector<int8_t>>& activations) {
    PROFILE_FUNCTION(generate_lut_optimized);

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

    // Parallelize the computation using multiple threads
    unsigned int num_threads = get_optimal_thread_count();
    std::vector<std::thread> threads;

    auto worker = [&](int start_row, int end_row) {
        PROFILE_FUNCTION(generate_lut_worker);
        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < K; j += GROUP_SIZE) {
                // Pre-compute chunk index
                int chunk_idx = j / GROUP_SIZE;

                // Process permutations in batches for better cache locality
                for (int p = 0; p < P; p += 8) {
                    for (int p_offset = 0; p_offset < 8 && p + p_offset < P; p_offset++) {
                        int current_p = p + p_offset;
                        int dot_product = 0;

                        // Unroll the inner loop for better instruction-level parallelism
                        for (int k = 0; k < GROUP_SIZE && j + k < K; k++) {
                            dot_product += activations[i][j + k] * non_negative_perms[current_p][k];
                        }

                        lut[i][chunk_idx][current_p] = dot_product;
                    }
                }
            }
        }
    };

    // Divide work among threads
    int rows_per_thread = M / num_threads;
    int start_row = 0;

    for (unsigned int t = 0; t < num_threads - 1; t++) {
        int end_row = start_row + rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
        start_row = end_row;
    }

    // Last thread handles remaining rows
    threads.emplace_back(worker, start_row, M);

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    return lut;
}

// Template implementation
template<typename T>
std::vector<std::vector<int>> lut_mm_optimized(const std::vector<std::vector<int8_t>>& activations,
                                             const std::vector<std::vector<T>>& packed_weights) {
    PROFILE_FUNCTION(lut_mm_optimized);

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
    auto lut = generate_lut_optimized(activations);

    // Initialize result matrix
    std::vector<std::vector<int>> result(M, std::vector<int>(N, 0));

    // Parallelize the computation using multiple threads
    unsigned int num_threads = get_optimal_thread_count();
    std::vector<std::thread> threads;

    auto worker = [&](int start_row, int end_row) {
        PROFILE_FUNCTION(lut_mm_worker);
        for (int i = start_row; i < end_row; i++) {
            // Process columns in blocks for better cache locality
            for (int jj = 0; jj < N; jj += 32) {
                for (int j = jj; j < std::min(jj + 32, N); j++) {
                    int sum = 0;

                    // Process chunks in blocks
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

                        // Look up the dot product in the LUT and apply the sign
                        if (abs_value < lut[i][c].size()) {
                            int lut_value = lut[i][c][abs_value];
                            sum += lut_value * sign;
                        }
                    }

                    result[i][j] = sum;
                }
            }
        }
    };

    // Divide work among threads
    int rows_per_thread = M / num_threads;
    int start_row = 0;

    for (unsigned int t = 0; t < num_threads - 1; t++) {
        int end_row = start_row + rows_per_thread;
        threads.emplace_back(worker, start_row, end_row);
        start_row = end_row;
    }

    // Last thread handles remaining rows
    threads.emplace_back(worker, start_row, M);

    // Join all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Debug output for the first few values (only from main thread)
    if (M > 0 && N > 0 && num_chunks > 0) {
        T packed_value = packed_weights[0][0];
        int8_t sign, abs_value;

        if constexpr (std::is_same<T, sign_magnitude_8_t>::value) {
            sign = sm_sign(packed_value);
            abs_value = sm_magnitude(packed_value);
        } else {
            sign = (packed_value >= 0) ? 1 : -1;
            abs_value = std::abs(packed_value);
        }

        std::cout << "Debug: packed_value=" << static_cast<int>(packed_value)
                  << ", sign=" << static_cast<int>(sign)
                  << ", abs_value=" << static_cast<int>(abs_value) << std::endl;

        if (abs_value < lut[0][0].size()) {
            int lut_value = lut[0][0][abs_value];
            int contribution = lut_value * sign;
            std::cout << "  chunk 0: lut_value=" << lut_value
                      << ", sign=" << static_cast<int>(sign)
                      << ", contribution=" << contribution
                      << ", running sum=" << result[0][0] << std::endl;
        }
    }

    return result;
}

// Explicit template instantiations
template std::vector<std::vector<int>> lut_mm_optimized<int8_t>(const std::vector<std::vector<int8_t>>& activations,
                                                              const std::vector<std::vector<int8_t>>& packed_weights);
template std::vector<std::vector<int>> lut_mm_optimized<sign_magnitude_8_t>(const std::vector<std::vector<int8_t>>& activations,
                                                               const std::vector<std::vector<sign_magnitude_8_t>>& packed_weights);
