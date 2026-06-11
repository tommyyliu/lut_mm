import numpy as np
from itertools import product

from weight_packing import pack_weights

# All ternary vectors in {-1, 0, 1}^5, ordered so that PERMS[v + 121] is the
# vector whose balanced-ternary value (dot with [81, 27, 9, 3, 1]) is v.
# PERMS[121:] are therefore the 122 vectors with non-negative values; by
# mirror symmetry the rest are their negations.
PERMS = np.array(list(product([-1, 0, 1], repeat=5)))
NON_NEGATIVE_PERMS = PERMS[121:]

def basic_mm(activations, weights):
    """Basic matrix multiplication."""
    return activations @ weights

def generate_lut(activations):
    """Dot products of every 5-activation group with every non-negative perm."""
    lut = np.zeros((activations.shape[0], activations.shape[1] // 5, NON_NEGATIVE_PERMS.shape[0]), dtype=np.int32)
    for i in range(activations.shape[0]):
        for j in range(0, activations.shape[1], 5):
            for p, perm in enumerate(NON_NEGATIVE_PERMS):
                lut[i, j//5, p] = np.dot(activations[i, j:j+5], perm)
    return lut

def lut_mm(activations, packed_weights):
    """Look-up table based matrix multiplication."""
    lut = generate_lut(activations)
    mm_product = np.zeros((activations.shape[0], packed_weights.shape[1], activations.shape[1] // 5), dtype=np.int32)
    for i in range(activations.shape[0]):
        for j in range(packed_weights.shape[1]):
            for k in range(activations.shape[1] // 5):
                mm_product[i, j, k] = lut[i, k, np.abs(packed_weights[k, j])] * np.sign(packed_weights[k, j])
    return mm_product.sum(axis=2)


def generate_lut_vec(activations):
    """Vectorized generate_lut: one batched matmul against the permutations."""
    groups = activations.reshape(activations.shape[0], -1, 5).astype(np.int32)
    return groups @ NON_NEGATIVE_PERMS.T.astype(np.int32)  # (M, K//5, 122)


def lut_mm_vec(activations, packed_weights, group_chunk=32):
    """Vectorized lut_mm, bit-identical to the loop version.

    The lookup phase gathers lut[i, g, |packed[g, j]|] with take_along_axis
    and reduces over groups with the signs applied, chunked over groups so
    the (M, chunk, N) temporary stays modest.
    """
    lut = generate_lut_vec(activations)              # (M, G, 122)
    idx = np.abs(packed_weights).astype(np.intp)     # (G, N)
    sgn = np.sign(packed_weights).astype(np.int32)   # (G, N)
    G, N = packed_weights.shape
    out = np.zeros((activations.shape[0], N), dtype=np.int32)
    for g0 in range(0, G, group_chunk):
        g1 = min(g0 + group_chunk, G)
        gathered = np.take_along_axis(lut[:, g0:g1, :], idx[None, g0:g1, :],
                                      axis=2)        # (M, g1-g0, N)
        out += np.einsum('mgn,gn->mn', gathered, sgn[g0:g1])
    return out


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
