#include <vector>
#include <cstdint> // For int8_t, uint8_t
#include <iostream> // For debugging prints
#include <iomanip>  // For std::setw, std::setprecision
#include <algorithm> // For std::min
#include <chrono>   // For benchmarking
#include <random>   // For generating random matrices
#include <type_traits> // For std::is_same
#include <thread>   // For multi-threading
#include <sstream>  // For stringstream
#include "lut_mm.h" // Include our header file
#include "lut_mm_optimized.h" // Include optimized functions
#include "weight_packing.h" // Include weight packing header
#include "permutations.h" // Include permutations header
#include "profiling.h" // Include profiling header

int main() {
    // Print permutation statistics
    std::cout << "\n--- Permutation Statistics ---" << std::endl;
    print_permutation_stats();

    // Example dimensions
    const int M = 2; // Number of activation rows
    const int K = 5; // Number of features
    const int N = 2; // Number of weight columns

    // Example activations (M x K)
    std::vector<std::vector<int8_t>> activations = {
        {1, 2, 4, 3, 2},
        {4, 5, 6, 1, 2}
    };

    // Example weights (K x N)
    std::vector<std::vector<int8_t>> weights = {
        {-1, 1},
        {0, 0},
        {1, -1},
        {0, 0},
        {0, 0}
    };

    // 1. Basic matrix multiplication
    std::cout << "\n--- Basic Matrix Multiplication ---" << std::endl;
    auto basic_result = basic_mm(activations, weights);

    std::cout << "Result:" << std::endl;
    for (const auto& row : basic_result) {
        for (int val : row) {
            std::cout << std::setw(4) << val << " ";
        }
        std::cout << std::endl;
    }

    // 2. Pack the weights using our pack_weights_lut function
    std::cout << "\n--- Packing Weights ---" << std::endl;
    auto packed_weights = pack_weights_lut(weights);

    std::cout << "Original weights:" << std::endl;
    for (const auto& row : weights) {
        for (int8_t val : row) {
            std::cout << std::setw(4) << static_cast<int>(val) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nPacked weights:" << std::endl;
    for (const auto& row : packed_weights) {
        for (int8_t val : row) {
            std::cout << std::setw(4) << static_cast<int>(val) << " ";
        }
        std::cout << std::endl;
    }

    // 3. LUT-based matrix multiplication
    std::cout << "\n--- LUT-based Matrix Multiplication ---" << std::endl;

    // Print some non-negative permutations for reference
    std::cout << "Some non-negative permutations:" << std::endl;
    auto non_neg_perms = get_non_negative_perms();
    for (size_t i = 0; i < std::min(size_t(10), non_neg_perms.size()); i++) {
        std::cout << "Perm[" << i << "] (value " << calculate_perm_value(non_neg_perms[i]) << "): ";
        for (int8_t val : non_neg_perms[i]) {
            std::cout << std::setw(2) << static_cast<int>(val) << " ";
        }
        std::cout << std::endl;
    }

    // Generate a small lookup table for verification
    std::cout << "\nSmall LUT for verification:" << std::endl;
    auto small_lut = generate_lut(activations);

    // Print the activations for reference
    std::cout << "\nActivations:" << std::endl;
    for (size_t i = 0; i < activations.size(); i++) {
        std::cout << "Row " << i << ": ";
        for (size_t j = 0; j < activations[i].size(); j++) {
            std::cout << std::setw(2) << static_cast<int>(activations[i][j]) << " ";
        }
        std::cout << std::endl;
    }

    // Print the first few entries of the LUT for each activation row and chunk
    // LUT[i][j][k] = dot product of activation row i, chunk j with permutation k
    for (size_t i = 0; i < small_lut.size(); i++) {
        for (size_t j = 0; j < small_lut[i].size(); j++) {
            std::cout << "LUT[" << i << "][" << j << "] (first 10 entries): ";
            for (size_t k = 0; k < std::min(size_t(10), small_lut[i][j].size()); k++) {
                std::cout << std::setw(4) << small_lut[i][j][k] << " ";
            }
            std::cout << std::endl;

            // Verify a few entries to show how the LUT is calculated
            if (i == 0 && j == 0) {
                std::cout << "  Verification of first few entries:" << std::endl;
                for (size_t k = 0; k < std::min(size_t(3), small_lut[i][j].size()); k++) {
                    // Calculate dot product manually
                    int dot_product = 0;
                    for (size_t l = 0; l < GROUP_SIZE && l < activations[i].size(); l++) {
                        dot_product += activations[i][l] * non_neg_perms[k][l];
                    }
                    std::cout << "    Perm[" << k << "] dot Activations[" << i << "] = ";
                    for (size_t l = 0; l < GROUP_SIZE; l++) {
                        std::cout << static_cast<int>(activations[i][l]) << "*" << static_cast<int>(non_neg_perms[k][l]);
                        if (l < GROUP_SIZE - 1) std::cout << " + ";
                    }
                    std::cout << " = " << dot_product << " (LUT value: " << small_lut[i][j][k] << ")" << std::endl;
                }
            }
        }
    }

    // Prepare packed weights for LUT-MM
    // For LUT-MM, weights need to be transposed and packed
    std::vector<std::vector<int8_t>> weights_T(N, std::vector<int8_t>(K));
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            weights_T[j][i] = weights[i][j];
        }
    }

    // Pack the transposed weights
    auto packed_weights_T = pack_weights(weights_T);

    // Perform LUT-based matrix multiplication
    auto lut_result = lut_mm(activations, packed_weights_T);

    // Print the result of the LUT-based matrix multiplication
    std::cout << "\nLUT-MM Result:" << std::endl;
    for (const auto& row : lut_result) {
        for (int val : row) {
            std::cout << std::setw(4) << val << " ";
        }
        std::cout << std::endl;
    }

    // Verify that the results match the basic matrix multiplication
    bool results_match = true;
    for (size_t i = 0; i < basic_result.size(); i++) {
        for (size_t j = 0; j < basic_result[i].size(); j++) {
            if (basic_result[i][j] != lut_result[i][j]) {
                results_match = false;
                std::cout << "Mismatch at [" << i << "][" << j << "]: "
                          << basic_result[i][j] << " vs " << lut_result[i][j] << std::endl;
            }
        }
    }
    std::cout << "Results match basic matrix multiplication: " << (results_match ? "Yes" : "No") << std::endl;

    // 4. Additional example with weight packing
    std::cout << "\n--- Additional Weight Packing Example ---" << std::endl;

    // Create a larger example matrix for weight packing
    std::vector<std::vector<int8_t>> weights_larger = {
        {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1},
        {1, 1, 0, 0, -1, -1, 0, 0, 1, 1}
    };

    std::cout << "Original weights:" << std::endl;
    for (const auto& row : weights_larger) {
        for (int8_t val : row) {
            std::cout << std::setw(4) << static_cast<int>(val) << " ";
        }
        std::cout << std::endl;
    }

    // Pack the weights
    auto packed_weights_larger = pack_weights(weights_larger);

    std::cout << "\nPacked weights:" << std::endl;
    for (const auto& row : packed_weights_larger) {
        for (int8_t val : row) {
            std::cout << std::setw(4) << static_cast<int>(val) << " ";
        }
        std::cout << std::endl;
    }

    // Compression ratio
    double original_size = weights_larger.size() * weights_larger[0].size();
    double packed_size = packed_weights_larger.size() * packed_weights_larger[0].size();
    double compression_ratio = original_size / packed_size;

    std::cout << "\nCompression ratio: " << std::fixed << std::setprecision(2)
              << compression_ratio << ":1" << std::endl;
    std::cout << "(Each 5 weights packed into 1 byte)" << std::endl;

    // 5. Benchmarking
    std::cout << "\n--- Benchmarking Matrix Multiplication ---" << std::endl;

    // Define matrix sizes for benchmarking
    const int BENCH_M = 2000;  // Number of activation rows
    const int BENCH_K = 2000;  // Number of features
    const int BENCH_N = 2000;  // Number of weight columns

    std::cout << "Matrix sizes: " << BENCH_M << "x" << BENCH_K << " * " << BENCH_K << "x" << BENCH_N << std::endl;

    // Function to print a horizontal separator line
    auto print_separator = [](int width = 80) {
        std::cout << std::string(width, '-') << std::endl;
    };

    // Function to print a table header
    auto print_table_header = [&print_separator]() {
        std::cout << std::left << std::setw(30) << "Algorithm"
                  << std::right << std::setw(15) << "Time (s)"
                  << std::right << std::setw(15) << "Speedup vs Basic"
                  << std::right << std::setw(20) << "Speedup vs Category" << std::endl;
        print_separator();
    };

    // Create random matrices for benchmarking
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> ternary_dist(-1, 1); // Ternary distribution for weights
    std::uniform_int_distribution<> act_dist(0, 10);     // Activation distribution

    // Generate random activations
    std::vector<std::vector<int8_t>> bench_activations(BENCH_M, std::vector<int8_t>(BENCH_K));
    for (int i = 0; i < BENCH_M; i++) {
        for (int j = 0; j < BENCH_K; j++) {
            bench_activations[i][j] = act_dist(gen);
        }
    }

    // Generate random weights
    std::vector<std::vector<int8_t>> bench_weights(BENCH_K, std::vector<int8_t>(BENCH_N));
    for (int i = 0; i < BENCH_K; i++) {
        for (int j = 0; j < BENCH_N; j++) {
            bench_weights[i][j] = ternary_dist(gen);
        }
    }

    std::cout << "\n\nRunning benchmarks, please wait..." << std::endl;

    // Enable profiling
    Profiler::enable();
    Profiler::reset();

    // Benchmark basic matrix multiplication
    auto start_basic = std::chrono::high_resolution_clock::now();
    auto bench_basic_result = basic_mm(bench_activations, bench_weights);
    auto end_basic = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_basic = end_basic - start_basic;

    // Benchmark optimized basic matrix multiplication
    auto start_basic_opt = std::chrono::high_resolution_clock::now();
    auto bench_basic_opt_result = basic_mm_optimized(bench_activations, bench_weights);
    auto end_basic_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_basic_opt = end_basic_opt - start_basic_opt;

    // Verify optimized basic MM results match
    bool basic_opt_results_match = true;
    for (size_t i = 0; i < bench_basic_result.size() && i < bench_basic_opt_result.size(); i++) {
        for (size_t j = 0; j < bench_basic_result[i].size() && j < bench_basic_opt_result[i].size(); j++) {
            if (bench_basic_result[i][j] != bench_basic_opt_result[i][j]) {
                basic_opt_results_match = false;
                std::cout << "Basic MM mismatch at [" << i << "][" << j << "]: "
                          << bench_basic_result[i][j] << " vs " << bench_basic_opt_result[i][j] << std::endl;
                if (i * bench_basic_result[i].size() + j > 5) break;
            }
        }
        if (!basic_opt_results_match && i > 2) break;
    }

    // Calculate speedup for optimized basic MM
    double basic_speedup = elapsed_basic.count() / elapsed_basic_opt.count();

    // Prepare for LUT-MM
    // Transpose weights
    auto start_transpose = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int8_t>> bench_weights_T(BENCH_N, std::vector<int8_t>(BENCH_K));
    for (int i = 0; i < BENCH_K; i++) {
        for (int j = 0; j < BENCH_N; j++) {
            bench_weights_T[j][i] = bench_weights[i][j];
        }
    }
    auto end_transpose = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_transpose = end_transpose - start_transpose;

    // Pack the transposed weights
    auto start_packing = std::chrono::high_resolution_clock::now();
    auto bench_packed_weights_T = pack_weights(bench_weights_T);
    auto end_packing = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_packing = end_packing - start_packing;

    // Convert to sign_magnitude_8_t if needed
    std::vector<std::vector<sign_magnitude_8_t>> bench_packed_weights_T_sm;
    if (!std::is_same<decltype(bench_packed_weights_T[0][0]), sign_magnitude_8_t>::value) {
        bench_packed_weights_T_sm.resize(bench_packed_weights_T.size());
        for (size_t i = 0; i < bench_packed_weights_T.size(); i++) {
            bench_packed_weights_T_sm[i].resize(bench_packed_weights_T[i].size());
            for (size_t j = 0; j < bench_packed_weights_T[i].size(); j++) {
                bench_packed_weights_T_sm[i][j] = static_cast<sign_magnitude_8_t>(bench_packed_weights_T[i][j]);
            }
        }
    } else {
        bench_packed_weights_T_sm = bench_packed_weights_T;
    }

    // Generate LUT
    auto start_lut_gen = std::chrono::high_resolution_clock::now();
    auto bench_lut = generate_lut(bench_activations);
    auto end_lut_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_lut_gen = end_lut_gen - start_lut_gen;

    // Generate optimized LUT
    auto start_lut_gen_opt = std::chrono::high_resolution_clock::now();
    auto bench_lut_opt = generate_lut_optimized(bench_activations);
    auto end_lut_gen_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_lut_gen_opt = end_lut_gen_opt - start_lut_gen_opt;

    // Benchmark LUT-based matrix multiplication
    auto start_lut = std::chrono::high_resolution_clock::now();
    auto bench_lut_result = lut_mm(bench_activations, bench_packed_weights_T_sm);
    auto end_lut = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_lut = end_lut - start_lut;

    // Benchmark optimized LUT-based matrix multiplication
    auto start_lut_opt = std::chrono::high_resolution_clock::now();
    auto bench_lut_opt_result = lut_mm_optimized(bench_activations, bench_packed_weights_T_sm);
    auto end_lut_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_lut_opt = end_lut_opt - start_lut_opt;

    // Calculate total time for LUT-MM including preparation
    double total_lut_time = elapsed_transpose.count() + elapsed_packing.count() +
                           elapsed_lut_gen.count() + elapsed_lut.count();

    // Calculate total time for optimized LUT-MM including preparation
    double total_lut_opt_time = elapsed_transpose.count() + elapsed_packing.count() +
                               elapsed_lut_gen_opt.count() + elapsed_lut_opt.count();

    // Verify optimized LUT-MM results match
    bool lut_opt_results_match = true;
    for (size_t i = 0; i < bench_lut_result.size() && i < bench_lut_opt_result.size(); i++) {
        for (size_t j = 0; j < bench_lut_result[i].size() && j < bench_lut_opt_result[i].size(); j++) {
            if (bench_lut_result[i][j] != bench_lut_opt_result[i][j]) {
                lut_opt_results_match = false;
                std::cout << "LUT-MM mismatch at [" << i << "][" << j << "]: "
                          << bench_lut_result[i][j] << " vs " << bench_lut_opt_result[i][j] << std::endl;
                if (i * bench_lut_result[i].size() + j > 5) break;
            }
        }
        if (!lut_opt_results_match && i > 2) break;
    }

    // Verify results match between basic and LUT-MM
    bool bench_results_match = true;
    for (size_t i = 0; i < bench_basic_result.size() && i < bench_lut_result.size(); i++) {
        for (size_t j = 0; j < bench_basic_result[i].size() && j < bench_lut_result[i].size(); j++) {
            if (bench_basic_result[i][j] != bench_lut_result[i][j]) {
                bench_results_match = false;
                std::cout << "Mismatch at [" << i << "][" << j << "]: "
                          << bench_basic_result[i][j] << " vs " << bench_lut_result[i][j] << std::endl;
                // Only show a few mismatches to avoid flooding the output
                if (i * bench_basic_result[i].size() + j > 10) break;
            }
        }
        if (!bench_results_match && i > 5) break;
    }

    // Calculate all speedups
    double speedup_lut = elapsed_basic.count() / elapsed_lut.count();
    double total_speedup_lut = elapsed_basic.count() / total_lut_time;
    double opt_speedup_lut = elapsed_basic.count() / elapsed_lut_opt.count();
    double total_opt_speedup_lut = elapsed_basic.count() / total_lut_opt_time;
    double lut_opt_vs_normal = elapsed_lut.count() / elapsed_lut_opt.count();
    double total_lut_opt_vs_normal = total_lut_time / total_lut_opt_time;

    // Memory usage comparison
    size_t basic_memory = BENCH_K * BENCH_N * sizeof(int8_t); // Original weights
    size_t lut_memory = bench_packed_weights_T.size() * bench_packed_weights_T[0].size() * sizeof(int8_t); // Packed weights
    double memory_reduction = static_cast<double>(basic_memory) / lut_memory;

    // Print results
    std::cout << "\n\n--- BENCHMARK RESULTS ---" << std::endl;
    std::cout << "Matrix size: " << BENCH_M << "x" << BENCH_K << " * " << BENCH_K << "x" << BENCH_N << std::endl;
    std::cout << "Correctness checks:" << std::endl;
    std::cout << "  Basic MM vs Optimized Basic MM: " << (basic_opt_results_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "  LUT-MM vs Optimized LUT-MM: " << (lut_opt_results_match ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Basic MM vs LUT-MM: " << (bench_results_match ? "PASS" : "FAIL") << std::endl;

    // Print memory usage
    std::cout << "\nMemory usage:" << std::endl;
    std::cout << "  Basic weights: " << basic_memory << " bytes" << std::endl;
    std::cout << "  Packed weights: " << lut_memory << " bytes" << std::endl;
    std::cout << "  Memory reduction: " << std::fixed << std::setprecision(2) << memory_reduction << "x" << std::endl;

    // Print performance table
    std::cout << "\nPerformance Comparison:" << std::endl;
    print_separator();
    print_table_header();

    // Core algorithm performance (excluding preparation)
    std::cout << std::left << std::setw(30) << "Basic MM"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_basic.count()
              << std::right << std::setw(15) << "1.00x"
              << std::right << std::setw(20) << "baseline" << std::endl;

    std::cout << std::left << std::setw(30) << "Optimized Basic MM"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_basic_opt.count()
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << basic_speedup << "x"
              << std::right << std::setw(20) << basic_speedup << "x" << std::endl;

    std::cout << std::left << std::setw(30) << "LUT-MM"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_lut.count()
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << speedup_lut << "x"
              << std::right << std::setw(20) << "baseline" << std::endl;

    std::cout << std::left << std::setw(30) << "Optimized LUT-MM"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_lut_opt.count()
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << opt_speedup_lut << "x"
              << std::right << std::setw(20) << lut_opt_vs_normal << "x" << std::endl;

    print_separator();

    // Preparation times
    std::cout << "\nPreparation Times:" << std::endl;
    print_separator();
    std::cout << std::left << std::setw(30) << "Operation"
              << std::right << std::setw(15) << "Time (s)"
              << std::right << std::setw(15) << "% of Total" << std::endl;
    print_separator();

    std::cout << std::left << std::setw(30) << "Transpose weights"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_transpose.count()
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << (elapsed_transpose.count() / total_lut_time * 100) << "%" << std::endl;

    std::cout << std::left << std::setw(30) << "Pack weights"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_packing.count()
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << (elapsed_packing.count() / total_lut_time * 100) << "%" << std::endl;

    std::cout << std::left << std::setw(30) << "Generate LUT"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_lut_gen.count()
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << (elapsed_lut_gen.count() / total_lut_time * 100) << "%" << std::endl;

    std::cout << std::left << std::setw(30) << "Generate LUT (optimized)"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_lut_gen_opt.count()
              << std::right << std::setw(15) << std::fixed << std::setprecision(1) << (elapsed_lut_gen_opt.count() / total_lut_opt_time * 100) << "%" << std::endl;

    print_separator();

    // Total times (including preparation)
    std::cout << "\nTotal Times (including preparation):" << std::endl;
    print_separator();
    print_table_header();

    std::cout << std::left << std::setw(30) << "Basic MM"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_basic.count()
              << std::right << std::setw(15) << "1.00x"
              << std::right << std::setw(20) << "baseline" << std::endl;

    std::cout << std::left << std::setw(30) << "Optimized Basic MM"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << elapsed_basic_opt.count()
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << basic_speedup << "x"
              << std::right << std::setw(20) << basic_speedup << "x" << std::endl;

    std::cout << std::left << std::setw(30) << "LUT-MM with preparation"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << total_lut_time
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << total_speedup_lut << "x"
              << std::right << std::setw(20) << "baseline" << std::endl;

    std::cout << std::left << std::setw(30) << "Optimized LUT-MM with prep"
              << std::right << std::setw(15) << std::fixed << std::setprecision(3) << total_lut_opt_time
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << total_opt_speedup_lut << "x"
              << std::right << std::setw(20) << total_lut_opt_vs_normal << "x" << std::endl;

    // Compare optimized basic MM vs optimized LUT-MM with prep
    double opt_basic_vs_opt_lut_prep = elapsed_basic_opt.count() / total_lut_opt_time;
    std::string comparison_result;
    if (opt_basic_vs_opt_lut_prep > 1) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << opt_basic_vs_opt_lut_prep << "x (LUT better)";
        comparison_result = ss.str();
    } else {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << (1.0/opt_basic_vs_opt_lut_prep) << "x (Basic better)";
        comparison_result = ss.str();
    }

    std::cout << std::left << std::setw(30) << "Opt Basic MM vs Opt LUT-MM"
              << std::right << std::setw(15) << "--"
              << std::right << std::setw(15) << "--"
              << std::right << std::setw(20) << comparison_result << std::endl;

    print_separator();

    // Print profiling results
    std::cout << "\n\n--- DETAILED PROFILING RESULTS ---" << std::endl;
    Profiler::print_results();

    return 0;
}
