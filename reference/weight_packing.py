from itertools import product

import numpy as np

PERMS = np.array(list(product([-1, 0, 1], repeat=5)))


# Each 5 ternary weights are packed into a single byte
# The ternary weights are treated as a ternary number with possible digits of -1, 0, 1 in each position
def pack_weights(weights):
    digit_weights = np.array([81,27,9,3,1])
    packed_weights = np.zeros((weights.shape[0], weights.shape[1] // 5), dtype=np.int8)
    for i in range(weights.shape[0]):
        # Select 5 at a time
        for j in range(0, weights.shape[1], 5):
            packed_weights[i, j//5] = np.dot(weights[i, j:j+5], digit_weights)
    return packed_weights


if __name__ == '__main__':
    # Validate that the packed indices are correct
    for perm in PERMS:
        print(pack_weights(np.expand_dims(perm,0)))
    print(pack_weights(np.expand_dims(PERMS[121], 0)))