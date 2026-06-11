"""Benchmark the NumPy reference implementation (reference/main.py).

Runs the reference basic_mm (@) and lut_mm on random data of the given
shape, verifies lut_mm exactly against an int64 matmul, and reports Gop/s
comparable to bench.exe. The reference lut_mm is a pure-Python triple loop,
so benchmark it at -m 1 (a GEMV row); larger M is infeasible and the rate
is M-independent anyway.

Usage: py tools/bench_python.py -m 1 -k 2080 -n 2048 -r 2
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference"))
import main as ref  # noqa: E402  (reference/main.py)
from weight_packing import pack_weights  # noqa: E402


def bench(fn, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", type=int, default=1)
    ap.add_argument("-k", type=int, default=2080)
    ap.add_argument("-n", type=int, default=2048)
    ap.add_argument("-r", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-lut", action="store_true",
                    help="skip the loop-based lut_mm (infeasible at large M)")
    args = ap.parse_args()
    M, K, N = args.m, args.k, args.n
    assert K % 5 == 0

    rng = np.random.default_rng(args.seed)
    acts = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    weights = rng.integers(-1, 2, size=(K, N), dtype=np.int8)
    gops = 2.0 * M * N * K / 1e9
    print(f"M={M} K={K} N={N} reps={args.r} ({gops:.4f} Gop)")

    ref_out = acts.astype(np.int64) @ weights.astype(np.int64)

    a32, w32 = acts.astype(np.int32), weights.astype(np.int32)
    ms = bench(lambda: ref.basic_mm(a32, w32), args.r)
    print(f"basic_mm (int32 matmul): {ms:10.2f} ms  {gops / (ms / 1e3):8.3f} Gop/s")

    t0 = time.perf_counter()
    packed = pack_weights(weights.T).T
    print(f"pack_weights (python): {(time.perf_counter() - t0) * 1e3:.0f} ms (one-time)")

    vec_out = ref.lut_mm_vec(acts, packed)
    exact = np.array_equal(vec_out.astype(np.int64), ref_out)
    print(f"lut_mm_vec accuracy: {'PASS' if exact else 'FAIL'}")
    if not exact:
        sys.exit(1)
    ms = bench(lambda: ref.lut_mm_vec(acts, packed), args.r)
    print(f"lut_mm_vec (vectorized): {ms:10.2f} ms  {gops / (ms / 1e3):8.3f} Gop/s")

    if args.skip_lut:
        return
    lut_out = ref.lut_mm(acts, packed)
    exact = np.array_equal(lut_out.astype(np.int64), ref_out)
    print(f"lut_mm accuracy: {'PASS' if exact else 'FAIL'}")
    if not exact:
        sys.exit(1)
    ms = bench(lambda: ref.lut_mm(acts, packed), args.r)
    print(f"lut_mm (python loops):   {ms:10.2f} ms  {gops / (ms / 1e3):8.3f} Gop/s")


if __name__ == "__main__":
    main()
