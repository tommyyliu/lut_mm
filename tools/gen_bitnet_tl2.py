"""Generate BitNet TL2 kernels for this project's benchmark shapes.

Drives microsoft/BitNet's own utils/codegen_tl2.py (cloned under
third_party/BitNet) to emit src/generated/bitnet-lut-kernels.h, so the
kernel code under benchmark is byte-for-byte what their generator produces.

Shapes are chosen to satisfy both their constraints (N % BM == 0,
(K % BK) % 32 == 0) and ours (K % 5 == 0). Note BitNet's "M" is the output
dimension, i.e. our N.
"""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODEGEN = os.path.join(ROOT, "third_party", "BitNet", "utils", "codegen_tl2.py")
OUT = os.path.join(ROOT, "src", "generated", "bitnet-lut-kernels.h")

# [their M (= our N), K]
KERNEL_SHAPES = [[2048, 2080], [4096, 4160]]
BM_LIST = [256, 256]
BK_LIST = [96, 96]
BM_SIMD_LIST = [32, 32]


def main():
    spec = importlib.util.spec_from_file_location("codegen_tl2", CODEGEN)
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)

    for i, (m, k) in enumerate(KERNEL_SHAPES):
        assert m % BM_LIST[i] == 0
        assert (k % BK_LIST[i]) % 32 == 0

    k_list = [
        cg.get_three_k_two_k(shape[1], BK_LIST[i])
        for i, shape in enumerate(KERNEL_SHAPES)
    ]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="\n") as f:
        f.write("#if defined(GGML_BITNET_X86_TL2)\n")
        f.write(cg.gen_ctor_code())
        for i, shape in enumerate(KERNEL_SHAPES):
            pre = "{}_{}".format(shape[0], shape[1])
            f.write(
                cg.gen_tbl_impl(pre, BM_LIST[i], BK_LIST[i], BM_SIMD_LIST[i],
                                k_list[i]))
        f.write(cg.gen_top_api(KERNEL_SHAPES, k_list))
        # gen_transform_code is skipped: it needs ggml tensor plumbing that
        # the benchmark replaces with tools/pack_tl2.py.
        f.write("#endif\n")
    print("wrote", OUT)
    for shape, (two_k, three_k) in zip(KERNEL_SHAPES, k_list):
        print("  shape m={} k={}: three_k={} two_k={}".format(
            shape[0], shape[1], three_k, two_k))


if __name__ == "__main__":
    main()
