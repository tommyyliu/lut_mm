#ifndef WEIGHT_PACKING_H
#define WEIGHT_PACKING_H

#include <cstdint> // For int8_t, uint8_t
#include <vector>

// Constants
const int GROUP_SIZE = 5;  // Number of ternary weights per group
const int DIGIT_WEIGHTS[] = {81, 27, 9, 3, 1};  // Powers of 3: 3^4, 3^3, 3^2, 3^1, 3^0

typedef unsigned char sign_magnitude_8_t;

// Pack ternary weights (-1, 0, 1) into single int8_t values using 5-trit encoding
std::vector<std::vector<unsigned char>> pack_weights(
    const std::vector<std::vector<int8_t> > &weights);

int8_t sm_sign(sign_magnitude_8_t value);
int8_t sm_magnitude(sign_magnitude_8_t value);

#endif // WEIGHT_PACKING_H