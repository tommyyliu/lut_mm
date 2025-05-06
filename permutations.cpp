#include "permutations.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <iostream>

std::vector<std::vector<int8_t>> generate_permutations() {
    std::vector<std::vector<int8_t>> perms;
    std::vector<int8_t> values = {-1, 0, 1};

    // Recursive function to generate all permutations
    std::function<void(std::vector<int8_t>&, int)> generate =
        [&](std::vector<int8_t>& current, int depth) {
            if (depth == GROUP_SIZE) {
                perms.push_back(current);
                return;
            }

            for (int val : values) {
                current[depth] = val;
                generate(current, depth + 1);
            }
        };

    std::vector<int8_t> current(GROUP_SIZE, 0);
    generate(current, 0);

    return perms;
}

int calculate_perm_value(const std::vector<int8_t>& perm) {
    // The return value will be normal twos complement and not sign-magnitude.
    // This is just for checking ordering and not for use in the lut.

    // Compute weighted sum using ternary digit weights
    int value = 0;
    for (int i = 0; i < GROUP_SIZE && i < perm.size(); i++) {
        value += perm[i] * DIGIT_WEIGHTS[i];
    }
    return value;
}

std::vector<std::vector<int8_t>> get_non_negative_perms() {
    // Get permutations with values 0 to 121 (second half of all permutations)
    auto all_perms = generate_permutations();

    // Permutations are already in order; non-negative start at index 121
    const size_t NON_NEGATIVE_START_INDEX = 121;
    std::vector<std::vector<int8_t>> non_negative_perms;
    non_negative_perms.assign(all_perms.begin() + NON_NEGATIVE_START_INDEX, all_perms.end());

    return non_negative_perms;
}

void print_permutation_stats() {
    auto all_perms = generate_permutations();

    // Count permutations and find value range
    std::cout << "Total number of permutations: " << all_perms.size() << std::endl;

    int min_value = std::numeric_limits<int>::max();
    int max_value = std::numeric_limits<int>::min();
    int non_negative_count = 0;

    for (const auto& perm : all_perms) {
        int value = calculate_perm_value(perm);
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        if (value >= 0) non_negative_count++;
    }

    std::cout << "Value range: [" << min_value << ", " << max_value << "]" << std::endl;
    std::cout << "Number of non-negative permutations: " << non_negative_count << std::endl;
    std::cout << "Expected index for non-negative permutations: " << (all_perms.size() - non_negative_count) << std::endl;

    // Verify permutations are already sorted
    bool is_sorted = true;
    for (size_t i = 1; i < all_perms.size(); i++) {
        if (calculate_perm_value(all_perms[i-1]) > calculate_perm_value(all_perms[i])) {
            is_sorted = false;
            break;
        }
    }
    std::cout << "Permutations are already sorted by value: " << (is_sorted ? "Yes" : "No") << std::endl;
}
