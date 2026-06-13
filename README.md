# LUT-MM — Ternary Matrix Multiplication, 5 Trits per Byte

Matrix multiplication where the weight matrix is ternary (-1, 0, 1),
built around three ideas:

1. **Packing**: every 5 ternary weights are stored as one byte — their
   balanced-ternary value `sum w[t] * 3^(4-t)`, range [-121, 121]. That is
   1.600 bits/weight against the information-theoretic floor of
   log2(3) = 1.585 (99.1% efficient; BitNet's TL2 format is 1.667).
2. **Lookup**: for each group of 5 activations, the dot products against
   ternary vectors are precomputed into a table, so each packed weight
   byte resolves a 5-element dot product with a single table lookup.
3. **Mirror symmetry**: negating all 5 trits negates both the packed value
   and the dot product, so only 122 non-negative magnitudes are needed:
   `dot(v) = sign(v) * T[|v|]`. **122 ≤ 128** is what makes the format
   fast: the whole table fits the two-register index space of AVX-512's
   `vpermt2w`, turning lookups into register-resident shuffles.

A NumPy sketch of the idea lives in `reference/` (loop spec plus a
bit-identical vectorized version). Two write-ups walk through it:
[Part 1](writeups/idea.md) explains the 5-trits-per-byte lookup idea,
and [Part 2](writeups/avx512.md) covers why AVX-512 makes it fast.

## Results

Ryzen 9 9950X (Zen 5), MSVC 19.51, single-threaded unless noted.
M=256, K=2080, N=2048, random int8 activations, random ternary weights:

| Implementation | min ms | Gop/s | vs naive |
|----------------|-------:|------:|---------:|
| naive_mm (C++, AVX2 auto-vectorized by MSVC) | 25.9 | 84.2 | 1.00x |
| bitnet_tl2 (microsoft/BitNet's AVX2 pshufb) | 12.6 | 173.4 | 2.06x |
| bitnet_tl2@512b (our 512-bit port of TL2) | 6.6 | 329.3 | 3.91x |
| dense_mm_vnni (dense int8, AVX-512 VNNI `vpdpbusd`) | 3.9 | 555.2 | 6.59x |
| **lut_mm_avx512 (this project's kernel)** | **3.1** | **704.3** | **8.37x** |

(Best of 9 reps; a bare `build\bench.exe` reproduces this shape.) So the
kernel is 4.1x faster than BitNet's AVX2 TL2 kernel and 2.1x faster than
our 512-bit TL2 port. BitNet's kernels emit float (their native
dequantized output); the harness converts that to int32 for the
bit-exact check *outside* the timed region, the same concession it makes
for the int8→float input — so their timings cover only their kernel.

At M=256, K=4160, N=4096: lut_mm_avx512 718 Gop/s vs bitnet_tl2@512b 352
and bitnet_tl2 180. GEMV (M=1, K=2080) runs in 0.02-0.03 ms (330-430
Gop/s; timer-noise limited at that scale). Every implementation,
including BitNet's, produces bit-identical int32 results vs the dense
GEMM — the harness refuses to benchmark anything that doesn't.

`dense_mm_vnni` is the strongest dense baseline this hardware offers —
unpacked int8 weights fed to the dual-pumped `vpdpbusd` dot-product
instruction. The LUT kernel beats it by 1.27x at the cache-resident
shape above, and the gap stays modest while the dense weights still fit
this CPU's 32 MB L3 (1.5x at K=4160, 17 MB; M=256, single thread, Gop/s
LUT vs VNNI: 718 vs 485). It blows open once they spill: at K=8320 the
68 MB of dense weights fall out of L3 while the 13.6 MB packed form
still fits, and VNNI craters to DRAM speed (690 vs 127, 5.4x); at
K=16640 both spill and the ratio settles near the 5x density advantage
(582 vs 123). The compression is not just a footprint feature; it is
what keeps the kernel compute-bound at LLM-layer sizes.

Both our kernel and BitNet's have row-parallel wrappers (`-t threads`):
packed weights stay shared, independent chunks of output rows go to
threads. Peak for ours is around 8 threads, after which it turns
memory-bandwidth-bound and degrades; BitNet's TL2 keeps scaling to 16
threads from its lower single-thread base (peak ~1400 Gop/s). Full sweep
in [results/thread_scaling.csv](results/thread_scaling.csv), plotted in
[results/thread_scaling.svg](results/thread_scaling.svg) (regenerate with
`tools/plot_scaling.py`):

| Shape | threads | min ms | Gop/s | speedup vs 1T AVX-512 |
|-------|--------:|-------:|------:|----------------------:|
| M=256 K=2080 N=2048 | 8 | 0.69 | 3141 | 4.5x |
| M=256 K=4160 N=4096 | 8 | 1.99 | 4394 | 6.2x |

For scale, the NumPy reference implementations (`tools/bench_python.py`,
same shape) land far below the C kernels: pure-Python loop lut_mm ~0.022
Gop/s, int32 matmul 0.53, vectorized-NumPy lut_mm 3.43. The loop-based
lut_mm was timed at M=1 (380 ms) and scaled; its rate is M-independent.
NumPy has no BLAS path for integer dtypes, so its matmul is a plain C
loop.

## The kernel (`src/ternary_mm_avx512.cpp`, AVX-512 F+BW)

- **Lookup**: packed byte v → `sign(v) * T[|v|]` with |v| ≤ 121. The 122
  int16 magnitudes live in four zmm registers; two `vpermt2w` (64 entries
  each) plus a blend on index bit 6 perform 32 lookups in ~3 shuffles.
  Sign is applied with a masked subtract from a `vpmovw2m` mask.
- **Row blocking**: the index decode (widen, abs, sign mask, bit-6 mask)
  depends only on the weights, so 4 activation rows share it per sweep;
  each row adds just its two permutes, blend, masked negate, and int16
  accumulate. (8 rows would need 32 table registers, which spill and give
  the win back.)
- **Masked tail**: the last N % 32 columns run the same SIMD body under
  AVX-512BW masked loads/stores. The int16 accumulator rows are padded to
  a 32-lane stride, so row starts stay vector-aligned even when N is not
  a multiple of 32 (N=2047 runs within ~5% of N=2048).
- **Table construction**: with n = |v| + 121 = 9(h+13) + (l+4), where h is
  the top-3-trit value and l the bottom-2-trit value, T[m] = H[n/9] +
  L[n%9]. H (27 entries) and L (9 entries) are built lane-wise as
  broadcast-multiply-adds against constant digit vectors (H = a0·D0 +
  a1·D1 + a2·D2), and n/9, n%9 per table lane are compile-time constants,
  so the four table registers cost ~10 arithmetic ops plus eight
  constant-index `vpermw` — no scalar work, no divide in the data path.
- **Accumulation**: int16 partial sums, flushed to the int32 output every
  51 groups (51 × max|entry| 640 = 32640 < 32767, no overflow); the first
  flush stores instead of adds, so the output is never pre-zeroed.

## The BitNet comparison

[microsoft/BitNet](https://github.com/microsoft/BitNet)'s x86 TL2 kernel
groups 3 trits into a 4-bit index plus a separate sign bit (1.667
bits/weight) so lookups fit AVX2 `pshufb`'s 16-entry tables. To compare
faithfully, this repo runs their actual code, not a reimplementation:

- `tools/gen_bitnet_tl2.py` drives their own `utils/codegen_tl2.py` to
  emit `src/generated/bitnet-lut-kernels.h` (checked in, so the default
  build needs neither Python nor the BitNet clone; pinned to BitNet
  commit `01eb4157`).
- `tools/pack_tl2.py` packs weights with their numpy preprocessing,
  copied verbatim.
- `src/ternary_mm_bitnet.cpp` forces their activation-quantization scale
  to 1.0 — our activations are already int8 — which makes their kernel
  exact, so it participates in the bit-exact accuracy check.
- `src/ternary_mm_bitnet512.cpp` is our faithful 512-bit widening of
  their three-trit sweep (same packed blob; their LUT constructor and
  small two-trit tail kept as shipped), so the width comparison is
  measured rather than estimated.

Their kernels are shape-specialized at codegen time; K=2080/N=2048 and
K=4160/N=4096 satisfy their constraints and our K%5 simultaneously:

```
git clone --depth 1 https://github.com/microsoft/BitNet third_party/BitNet
py tools/gen_bitnet_tl2.py                # regenerate src/generated/
build\bench.exe -k 2080 -n 2048 --dump-w W.bin
py tools/pack_tl2.py --weights W.bin -k 2080 -n 2048 --out W_tl2.bin
build\bench.exe -k 2080 -n 2048 --bitnet W_tl2.bin
```

## Findings along the way

These experiments shaped the final kernel; the control kernels they used
(scalar LUT, AVX2 gather, and a 256-bit variant of ours) were removed in
the cleanup but live at commit `b84eb3e`.

- **Gathers are not shuffles.** The obvious AVX2 port of the LUT idea —
  `vpgatherdd` into the 243-entry table — ties the plain scalar loop
  (~47 Gop/s both; gather is microcoded on Zen, ~1 lookup/cycle) and
  loses to the auto-vectorized dense GEMM. BitNet's pshufb design beats
  it 3.7x in the same ISA class. Lookup throughput is everything.
- **Width vs design, measured as a 2x2** (Gop/s at K=2080, matched v1
  sweep implementations on both sides):

  | | 256-bit | 512-bit | width scaling |
  |---|---:|---:|---:|
  | TL2 3-trit pshufb | 173 | 329 | 1.90x |
  | ours 5-trit vpermt2w | 155 | 355 | 2.29x |

  At equal 256-bit width TL2 is ~12% ahead — it does more, cheaper
  lookups, while ours does one lookup per 5 weights but pays a 4-permute
  + 3-blend select tree. What flips the result is scaling: the 5-trit
  format gains superlinearly with width (2.29x vs 1.90x), because the
  two-register permute's table capacity doubles along with the lanes,
  collapsing the select tree to 2 permutes + 1 blend. At matched 512-bit
  width ours pulls ~8% ahead while storing weights 4.6% denser. TL2
  remains the better design for AVX2-only hardware, where no shuffle can
  index 122 entries; the 5-trit format only wins once AVX-512 unlocks the
  wide table.
- **Optimization passes** took the kernel from 355 to 695 Gop/s: SIMD
  table build replacing a serial scalar build (~40% of runtime), 4-row
  decode sharing, masked tails, and first-flush-store.

## Layout

- `reference/` — the idea in NumPy: loop spec, vectorized version, packing
- `src/ternary_mm.h` — shapes, packing format, public API
- `src/ternary_mm_ref.cpp` — `pack_weights` and the naive ground-truth GEMM
- `src/ternary_mm_avx512.cpp` — the kernel
- `src/ternary_mm_vnni.cpp` — dense AVX-512 VNNI baseline (`vpdpbusd`)
- `src/ternary_mm_bitnet.cpp`, `src/ternary_mm_bitnet512.cpp`,
  `src/generated/` — the BitNet comparison
- `src/bench_main.cpp` — harness: bit-exact verification, then timing
- `tools/` — BitNet codegen/packing drivers, NumPy benchmarks

## Build and run

From a VS x64 developer prompt (or after `vcvars64.bat`):

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
build\bench.exe [-m M] [-k K] [-n N] [-r reps] [-t threads] [--seed S]
```

Defaults: M=256, K=2080, N=2048, 5 reps. K must be a multiple of 5.
The AVX-512 kernel is detected and skipped at runtime if the CPU lacks
AVX-512BW; the VNNI baseline likewise requires AVX-512 VNNI. `-t` adds a
multi-threaded AVX-512 implementation to the accuracy check and
benchmark; the default is `-t 1`.

## License

GPL-3.0-or-later (see `LICENSE`). The BitNet-derived components —
`src/generated/bitnet-lut-kernels.h` and the preprocessing functions in
`tools/pack_tl2.py` — originate from
[microsoft/BitNet](https://github.com/microsoft/BitNet) under the MIT
License, Copyright (c) Microsoft Corporation, and retain those terms.
