#include "weight_packing.h"
#include <bitset>
#include <iostream>


sign_magnitude_8_t convert_sign_magnitude_8(int8_t value) {
    // Set magnitude
    sign_magnitude_8_t result = std::abs(value) & 0x7F;
    // Set sign bit (1 for negative, 0 for positive)
    if (value < 0) {
        result |= 0x80;
    }
    return result;
}

std::vector<std::vector<sign_magnitude_8_t>> pack_weights(
    const std::vector<std::vector<int8_t> > &weights) {

    // Get matrix dimensions
    int rows = weights.size();
    int cols = (rows > 0) ? weights[0].size() : 0;

    // Calculate packed columns (ceiling division)
    int packed_cols = (cols + GROUP_SIZE - 1) / GROUP_SIZE;

    // Initialize output matrix
    std::vector packed_weights(rows, std::vector<sign_magnitude_8_t>(packed_cols, 0));

    // Pack each row
    for (int r = 0; r < rows; r++) {
        // Process in groups of 5 weights
        for (int c = 0; c < cols; c += GROUP_SIZE) {
            int8_t packed_value = 0;

            for (int i = 0; i < GROUP_SIZE && (c + i) < cols; i++) {
                int8_t weight = weights[r][c + i];
                packed_value += weight * DIGIT_WEIGHTS[i];
            }

            sign_magnitude_8_t sm_value = convert_sign_magnitude_8(packed_value);
            packed_weights[r][c / GROUP_SIZE] = sm_value;
        }
    }

    return packed_weights;
}

int8_t sm_sign(sign_magnitude_8_t value) {
    return (value & 0x80) ? -1 : 1;
}

int8_t sm_magnitude(sign_magnitude_8_t value) {
    return 0x7F & value;
}
