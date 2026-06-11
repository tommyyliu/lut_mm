# SPDX-License-Identifier: GPL-3.0-or-later
"""Pack ternary weights (-1, 0, 1) five to a byte.

Five trits are read as a balanced-ternary number, w0*81 + w1*27 + w2*9 +
w3*3 + w4, which lands in [-121, 121] and therefore fits an int8: 1.6
bits per weight, against the information-theoretic floor of
log2(3) = 1.585. The packing is a bijection between {-1,0,1}^5 and
[-121, 121] (run this module to check).
"""
from itertools import product

import numpy as np

# Place values of the five trits: 3^4 .. 3^0 = [81, 27, 9, 3, 1].
DIGIT_WEIGHTS = 3 ** np.arange(4, -1, -1)


def pack_weights(weights):
    """Pack each row's ternary weights, 5 at a time, into int8 bytes.

    >>> pack_weights(np.array([[1, 0, -1, 0, 0]]))
    array([[72]], dtype=int8)
    """
    groups = weights.reshape(weights.shape[0], -1, 5)
    return (groups @ DIGIT_WEIGHTS).astype(np.int8)


if __name__ == "__main__":
    # Packing every ternary 5-vector, in lexicographic order, yields every
    # value in [-121, 121] exactly once: lexicographic order on the trits
    # is numeric order on the packed values, and the packing is a bijection.
    all_trit_vectors = np.array(list(product([-1, 0, 1], repeat=5)))
    packed = pack_weights(all_trit_vectors)
    assert (packed.ravel() == np.arange(-121, 122)).all()
    print("bijection verified: lexicographic {-1,0,1}^5 <-> [-121, 121]")
