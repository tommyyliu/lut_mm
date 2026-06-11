"""Ternary matrix multiplication via packed weights and lookup tables.

The idea, in three steps:

1. Pack: every 5 ternary weights become one byte — their balanced-ternary
   value v in [-121, 121] (see weight_packing.py). 5x smaller weights.
2. Look up: for each group of 5 activations, precompute its dot products
   against ternary vectors once; a packed byte then resolves a 5-element
   dot product with a single table lookup.
3. Halve the table: negating all 5 trits negates both v and the dot
   product, so only the 122 non-negative values need entries:

       dot(v) = sign(v) * lut[|v|]

   (122 <= 128 is also what later lets the AVX-512 kernel keep the whole
   table in registers.)

generate_lut / lut_mm are the readable loop spec; the *_vec versions are
bit-identical NumPy vectorizations. Run this module for a self-check.
"""
import numpy as np
from itertools import product

from weight_packing import DIGIT_WEIGHTS, pack_weights

# All ternary vectors in {-1,0,1}^5, lexicographic. Lexicographic order on
# trits is numeric order on packed values, so PERMS[v + 121] is exactly the
# vector whose packed value is v:
PERMS = np.array(list(product([-1, 0, 1], repeat=5)))
assert (PERMS @ DIGIT_WEIGHTS == np.arange(-121, 122)).all()

# The non-negative half. Mirror symmetry, PERMS[121 + v] == -PERMS[121 - v],
# makes it sufficient.
NON_NEGATIVE_PERMS = PERMS[121:]
assert (NON_NEGATIVE_PERMS[1:] == -PERMS[120::-1]).all()


def basic_mm(activations, weights):
    """Plain matrix multiply (pass dtypes wide enough not to overflow)."""
    return activations @ weights


def generate_lut(activations):
    """Dot products of every 5-activation group with every non-negative perm.

    lut[i, g, p] = activations[i, 5g:5g+5] . NON_NEGATIVE_PERMS[p]
    """
    M, K = activations.shape
    lut = np.zeros((M, K // 5, len(NON_NEGATIVE_PERMS)), dtype=np.int32)
    for i in range(M):
        for g in range(K // 5):
            for p, perm in enumerate(NON_NEGATIVE_PERMS):
                lut[i, g, p] = np.dot(activations[i, 5 * g:5 * g + 5], perm)
    return lut


def lut_mm(activations, packed_weights):
    """Look-up table based matrix multiplication (loop spec)."""
    lut = generate_lut(activations)
    M = activations.shape[0]
    G, N = packed_weights.shape
    out = np.zeros((M, N, G), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            for g in range(G):
                v = packed_weights[g, j]
                out[i, j, g] = lut[i, g, np.abs(v)] * np.sign(v)
    return out.sum(axis=2)


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
    # Small enough to check by hand: K=5 is a single weight group, so each
    # output element is one packed byte resolved by one table lookup.
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 10, size=(2, 5), dtype=np.int8)
    weights = rng.integers(-1, 2, size=(5, 3), dtype=np.int8)
    packed = pack_weights(weights.T).T  # (K/5, N): one byte per 5 weights

    print("activations (2x5):", acts, sep="\n")
    print("ternary weights (5x3):", weights, sep="\n")
    print("packed bytes (81*w0 + 27*w1 + 9*w2 + 3*w3 + w4, per column):",
          packed.ravel())
    expected = basic_mm(acts.astype(np.int32), weights.astype(np.int32))
    print("activations @ weights:", expected, sep="\n")

    assert (lut_mm(acts, packed) == expected).all()
    assert (lut_mm_vec(acts, packed) == expected).all()
    print("lut_mm and lut_mm_vec agree exactly")
