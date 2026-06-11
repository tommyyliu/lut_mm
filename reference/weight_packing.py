from itertools import product

import numpy as np

PERMS = np.array(list(product([-1, 0, 1], repeat=5)))


DIGIT_WEIGHTS = np.array([81, 27, 9, 3, 1])


# Each 5 ternary weights are packed into a single byte
# The ternary weights are treated as a ternary number with possible digits of -1, 0, 1 in each position
def pack_weights(weights):
    groups = weights.reshape(weights.shape[0], -1, 5)
    return (groups @ DIGIT_WEIGHTS).astype(np.int8)


if __name__ == '__main__':
    # Validate that the packed indices are correct
    for perm in PERMS:
        print(pack_weights(np.expand_dims(perm,0)))
    print(pack_weights(np.expand_dims(PERMS[121], 0)))