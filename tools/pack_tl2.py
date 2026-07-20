# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Pack a ternary weight matrix into BitNet's TL2 on-disk format.

The two preprocess_* functions are copied verbatim (MIT license) from
microsoft/BitNet utils/convert-hf-to-gguf-bitnet.py so the packed layout is
exactly what their kernels expect. The driver replaces their gguf/config
plumbing: it reads the raw K x N int8 weight matrix dumped by
`bench.exe --dump-w`, transposes it to BitNet's (M, K) row-major convention
(their M is our N), and writes the packed bytes.

Usage: py tools/pack_tl2.py --weights W.bin -k 2080 -n 2048 --out W_tl2.bin
"""

import argparse

import numpy as np

BM, BY, BM_SIMD = 256, 96, 32  # must match tools/gen_bitnet_tl2.py


def preprocess_two_weights_tl2(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 2, 2))
    hi_weight = np.multiply(np.split(weight, 2, axis=1)[0], 3)
    lo_weight = np.split(weight, 2, axis=1)[1]

    weight = np.reshape((hi_weight + lo_weight), weight_num // 2)

    weight = weight + 4
    weight = np.reshape(weight, (M, K // 2)).astype(np.uint8)
    weight = weight.reshape((M // BM, BM, K // 2)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 2, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 2)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 2, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm, by // 2))
    weight_0 = weight[:, :, :, :, :, 0]
    weight_1 = weight[:, :, :, :, :, 1]
    weight_0 = weight_0 << 4
    weight_1 = weight_1
    weight = weight_0 + weight_1
    weight = weight.reshape((M * K // bm // by, bm // 8, 8))
    weight[:, [0, 1, 2, 3], :] = weight[:, [0, 2, 1, 3], :]
    weight = weight.reshape(M * K // bm // by, bm)

    for i in range(weight.shape[0]):
        final_weight.append(weight[i, :])


def preprocess_three_weights_tl2(M, K, weight_num, BM, BY, bm, by, weight, final_weight):
    weight = np.reshape(weight, (weight_num // 3, 3))
    split_weights = np.split(weight, 3, axis=1)
    first_weight = np.multiply(split_weights[0], 9)
    second_weight = np.multiply(split_weights[1], 3)
    third_weight = split_weights[2]

    weight = np.reshape((first_weight + second_weight + third_weight), weight_num // 3)
    sign_weight = np.sign(weight) + 2
    sign_weight = np.where(sign_weight > 1, 0, sign_weight)
    weight = np.abs(weight)

    weight = np.reshape(weight, (M, K // 3)).astype(np.uint8)
    sign_weight = np.reshape(sign_weight, (M, K // 3)).astype(np.uint8)

    weight = weight.reshape((M // BM, BM, K // 3)).transpose(0, 2, 1)
    weight = weight.reshape((M // BM, K // BY, BY // 3, BM)).transpose(0, 1, 3, 2)
    weight = weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 3)).transpose(0, 1, 2, 4, 3)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, by // 3, bm)).transpose(0, 1, 2, 3, 5, 4)
    weight = weight.reshape((M // BM, K // BY, BM // bm, BY // by, bm, by // 3))
    weight_0 = weight[:, :, :, :, :, 0]
    weight_1 = weight[:, :, :, :, :, 1]
    weight_0 = weight_0 << 4
    weight_1 = weight_1
    weight = weight_0 + weight_1
    weight = weight.reshape((M * K // bm // by, bm // 8, 8))
    weight[:, [0, 1, 2, 3], :] = weight[:, [0, 2, 1, 3], :]
    weight = weight.reshape(M * K // bm // by, bm)

    for i in range(weight.shape[0]):
        final_weight.append(weight[i, :])

    sign_weight = sign_weight.reshape((M // BM, BM, K // 3)).transpose(0, 2, 1)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BY // 3, BM)).transpose(0, 1, 3, 2)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, bm, BY // 3)).transpose(0, 1, 2, 4, 3)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), by // 3 * 4, bm)).transpose(0, 1, 2, 3, 5, 4)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), bm, by // 3 * 4)).transpose(0, 1, 2, 3, 5, 4)
    sign_weight = sign_weight.reshape((M // BM, K // BY, BM // bm, BY // (by * 4), by // 3 * 8, bm // 2)).astype(np.uint16)
    combine_weight = np.zeros((M // BM, K // BY, BM // bm, BY // (by * 4), bm // 2), dtype=np.uint16)
    for i in range(16):
        temp_weight = sign_weight[:, :, :, :, i, :] << 15 - i
        combine_weight += temp_weight
    combine_weight = combine_weight.view(np.uint8)
    combine_weight = combine_weight.reshape((M * K // bm // (by * 4)), bm)

    for i in range(combine_weight.shape[0]):
        final_weight.append(combine_weight[i, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="raw K x N int8 weights")
    ap.add_argument("-k", type=int, required=True)
    ap.add_argument("-n", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = np.fromfile(args.weights, dtype=np.int8)
    assert w.size == args.k * args.n, "file size does not match K x N"
    # ours: (K, N) row-major -> theirs: (M, K) with M = our N
    weight = w.reshape(args.k, args.n).T.astype(np.float32)
    M, K = weight.shape
    assert M % BM == 0, "N must be a multiple of 256"
    assert (K % BY) % 32 == 0, "(K % 96) must be a multiple of 32"

    by = 192 // BM_SIMD  # 6: two 3-trit groups per index byte
    three_k = K - K % BY

    final_weight = []
    preprocess_three_weights_tl2(M, three_k, M * three_k, BM, BY, BM_SIMD, by,
                                 weight[:, :three_k], final_weight)
    if K % BY != 0:
        two_k = K - three_k
        preprocess_two_weights_tl2(M, two_k, M * two_k, BM, 32, 32, 4,
                                   weight[:, three_k:], final_weight)

    packed = np.array(final_weight, dtype=np.uint8).reshape(-1)
    packed.tofile(args.out)
    print("wrote {} ({} bytes, {:.3f} bits/weight)".format(
        args.out, packed.size, packed.size * 8 / (M * K)))


if __name__ == "__main__":
    main()
