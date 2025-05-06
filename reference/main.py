import numpy as np
from itertools import product

from weight_packing import pack_weights

# Generate all length 5 permutations of -1, 0, 1
# This creates a list of all possible combinations
PERMS = np.array(list(product([-1, 0, 1], repeat=5)))
NON_NEGATIVE_PERMS = PERMS[121:]
DIGIT_WEIGHTS = np.array([81, 27, 9, 3, 1])

def basic_mm(activations, weights):
    """Basic matrix multiplication."""
    return np.dot(activations, weights)

def generate_lut(activations):
    # Create lut from activations
    lut = np.zeros((activations.shape[0], activations.shape[1] // 5, NON_NEGATIVE_PERMS.shape[0]))
    # Use the pre-generated permutations from PERMS
    for i in range(activations.shape[0]):
        for j in range(0, activations.shape[1], 5):
            for p, perm in enumerate(NON_NEGATIVE_PERMS):
                lut[i, j//5, p] = np.dot(activations[i, j:j+5], perm)
    return lut

def lut_mm(activations, packed_weights):
    """Look-up table based matrix multiplication."""
    lut = generate_lut(activations)
    mm_product = np.zeros((activations.shape[0], packed_weights.shape[1], activations.shape[1] // 5))
    for i in range(activations.shape[0]):
        for j in range(packed_weights.shape[1]):
            for k in range(activations.shape[1] // 5):
                mm_product[i, j, k] = lut[i, k, np.abs(packed_weights[k, j])] * np.sign(packed_weights[k, j])
    return mm_product.sum(axis=2)


if __name__ == "__main__":
    # Simple test case
    acts = np.array([[1, 2, 4, 3, 2], [4, 5, 6, 1, 2]], dtype=np.int8)
    weights = np.array([[-1, 0, 1, 0, 0], [1, 0, -1, 0, 0]], dtype=np.int8).T

    # Basic matrix multiply
    result = basic_mm(acts, weights)
    print("Basic matrix multiply result:")
    print(result)

    # LUT result
    lut_result = lut_mm(acts, pack_weights(weights.T).T)
    print(f"LUT result: {lut_result}")
    print("\nLUT result shape:", lut_result.shape)
