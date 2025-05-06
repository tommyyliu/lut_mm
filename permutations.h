#ifndef PERMUTATIONS_H
#define PERMUTATIONS_H

#include <vector>
#include <cstdint>
#include "weight_packing.h"

// Generate all 3^5 permutations of -1, 0, 1 with length 5
std::vector<std::vector<int8_t>> generate_permutations();

// Get only non-negative permutations (values 0 to 121)
std::vector<std::vector<int8_t>> get_non_negative_perms();

// Calculate the packed value for a permutation using ternary encoding
int calculate_perm_value(const std::vector<int8_t>& perm);

// Print statistics about permutation count, range, and ordering
void print_permutation_stats();

#endif // PERMUTATIONS_H
