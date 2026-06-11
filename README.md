# LUT-MM — Ternary Matrix Multiplication via Packed Weights and Lookup Tables

Matrix multiplication where the weight matrix is ternary (-1, 0, 1):

1. **Packing**: every 5 ternary weights are stored as one byte — their
   balanced-ternary value `sum w[t] * 3^(4-t)`, range [-121, 121].
   5x memory reduction (1.6 bits/weight; the information-theoretic floor
   for ternary is log2(3) = 1.585).
2. **Lookup**: for each group of 5 activations, the dot products against
   ternary vectors are precomputed into a table. Each packed weight byte
   then resolves a 5-element dot product with a single table lookup.
3. **Mirror symmetry**: negating all 5 trits negates both the packed value
   and the dot product, so only the 122 non-negative magnitudes are needed:
   `dot(v) = sign(v) * T[|v|]`. This is what makes the format fast on
   AVX-512 — 122 entries fit the 128-entry index space of a `vpermt2w`
   pair, turning the lookup into register-resident shuffles.

A NumPy sketch of the idea lives in `reference/`; `tools/bench_python.py`
benchmarks it for the comparison table.

## Layout

- `src/ternary_mm.h` — shapes, packing format, public API
- `src/lut_build.h` — incremental group-LUT construction (DP, 363 adds)
- `src/ternary_mm_ref.cpp` — `pack_weights`, naive dense GEMM (ground
  truth), scalar LUT implementation
- `src/ternary_mm_avx2.cpp` — AVX2 kernel: `vpgatherdd` lookups into the
  full 243-entry table, 4 weight groups per sweep
- `src/ternary_mm_avx512.cpp` — AVX-512 (F+BW) kernel: mirror-reduced
  `vpermt2w` lookups (see below)
- `src/ternary_mm_bitnet.cpp` — adapter running microsoft/BitNet's TL2
  AVX2 kernels for comparison (generated into `src/generated/` by
  `tools/gen_bitnet_tl2.py` from a clone under `third_party/BitNet`)
- `src/bench_main.cpp` — experiment harness: exact-match accuracy check
  against the naive GEMM, then timing

## Build and run

From a VS x64 developer prompt (or after `vcvars64.bat`):

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
build\bench.exe [-m M] [-k K] [-n N] [-r reps] [--seed S]
```

Defaults: M=256, K=2000, N=2048, 5 reps. K must be a multiple of 5.
The AVX-512 kernel is detected and skipped at runtime if the CPU lacks
AVX-512BW.

### Comparing against microsoft/BitNet TL2

BitNet's kernels are shape-specialized at codegen time; K=2080/N=2048 and
K=4160/N=4096 satisfy both their constraints and ours and are pre-generated.

```
git clone --depth 1 https://github.com/microsoft/BitNet third_party/BitNet
py tools/gen_bitnet_tl2.py                # regenerate src/generated/ kernels
build\bench.exe -k 2080 -n 2048 --dump-w W.bin
py tools/pack_tl2.py --weights W.bin -k 2080 -n 2048 --out W_tl2.bin
build\bench.exe -k 2080 -n 2048 --bitnet W_tl2.bin
```

The packer reuses BitNet's own numpy preprocessing verbatim. The adapter
forces their activation-quantization scale to 1.0 (our activations are
already int8), which makes their kernel exact too — so it participates in
the bit-exact accuracy check like every other implementation.

## Results (Ryzen 9 9950X, MSVC 19.51, single-threaded)

M=256, K=2080, N=2048 (the BitNet-compatible shape):

| Implementation | min ms | Gop/s | vs naive |
|----------------|-------:|------:|---------:|
| python reference lut_mm (pure-Python loops) † | ~99,000 | 0.022 | 0.0003x |
| python reference basic_mm (int32 matmul) † | 4,104 | 0.53 | 0.006x |
| python lut_mm_vec (vectorized numpy) † | 637 | 3.43 | 0.040x |
| naive_mm (AVX2 auto-vectorized by MSVC) | 25.7 | 84.7 | 1.00x |
| lut_mm_scalar | 46.2 | 47.2 | 0.56x |
| lut_mm_avx2 (gather) | 47.1 | 46.4 | 0.55x |
| bitnet_tl2 (BitNet's AVX2 pshufb) | 13.4 | 163.3 | 1.93x |
| lut_mm_512@256b (width control, vpermt2w ymm) | 14.1 | 155.0 | 1.83x |
| bitnet_tl2@512b (our 512-bit port of TL2) | 7.5 | 292.1 | 3.48x |
| lut_mm_avx512 (vpermt2w zmm, v2) | 3.2 | 690.3 | 8.22x |

† measured by `tools/bench_python.py`. The reference lut_mm was timed at
M=1 (380 ms) and scaled to M=256; its per-element loop rate is
M-independent. NumPy does not use BLAS for integer dtypes, so its matmul
is a plain C loop; it and lut_mm_vec were timed at the full M=256. lut_mm_vec
(in `reference/main.py`) vectorizes the idea — LUT build as one batched
matmul against the 122 permutations, lookups via `take_along_axis`, signed
group reduction via einsum — and is bit-identical to the loop version.
Within numpy, the LUT idea beats dense int matmul 6.5x.

At M=256, K=4160, N=4096: lut_mm_avx512 611 Gop/s vs bitnet_tl2@512b 323
(1.9x) and bitnet_tl2 173 (3.5x); at GEMV (M=1, K=2080): 435 Gop/s. All
implementations, including BitNet's, produce bit-identical int32 results
vs the dense GEMM.

Takeaways:

- **Shuffles vs gathers**: BitNet's pshufb kernel beats our AVX2 gather
  kernel ~3.5x in the same ISA class — lookup throughput is everything.
- **Width vs design** — measured as a full 2x2 with two control kernels:
  `lut_mm_512@256b` (our format, 256-bit sweep) and `bitnet_tl2@512b`
  (our faithful 512-bit widening of their three-trit sweep; same packed
  blob, their LUT ctor and two-trit tail kept as shipped at 256-bit).
  Gop/s at M=256, K=2080 (K=4160 in parentheses):

  | | 256-bit | 512-bit | width scaling |
  |---|---:|---:|---:|
  | TL2 3-trit pshufb | 163 (167) | 292 (321) | 1.79-1.92x |
  | ours 5-trit vpermt2w (v1) | 155 (163) | 355 (391) | 2.29-2.40x |

  (The 2x2 uses the original v1 sweep on both of our kernels so the
  designs stay matched; the v2 optimizations below — shared decode across
  4 rows, SIMD table build — later pushed the 512-bit kernel to 690.)

  At 256 bits the designs tie (TL2 +5%): TL2 does more, cheaper lookups
  (a 16-entry pshufb pair per 3 weights plus nibble/sign unpacking); ours
  does one lookup per 5 weights but pays 4 permutes + 3 blends to select
  among 128 entries. So the headline win over BitNet-as-shipped is mostly
  the 512-bit datapath — but not all of it.
- **The 5-trit format scales superlinearly with width**: 256→512 bits
  gives our kernel 2.3x, because two-register permute capacity doubles
  along with the lanes (64 int16 entries per vpermt2w zmm), collapsing
  the select tree to 2 permutes + 1 blend. TL2's per-lookup structure is
  width-independent, so its port scales sublinearly (~1.8x). At matched
  512-bit width our kernel is ~21% faster than TL2, while also storing
  weights 4.6% denser (1.600 vs 1.677 bits/weight).
- BitNet TL2 remains the right design for AVX2-only hardware, where no
  shuffle can index 122 entries.

## How the AVX-512 kernel works

- **Lookup**: packed byte v → `sign(v) * T[|v|]` with |v| ≤ 121. The 122
  int16 magnitudes live in four zmm registers; two `vpermt2w` (64 entries
  each) plus a blend on index bit 6 perform 32 lookups in ~3 shuffles.
  Sign is applied with a masked subtract from a `vpmovw2m` mask.
- **Row blocking**: the index decode (widen, abs, sign mask, bit-6 mask)
  depends only on the weights, so 4 activation rows share it per sweep;
  each row adds just its two permutes, blend, masked negate, and int16
  accumulate. (R=8 spills the 32 table registers and gives the win back.)
- **Masked tail**: the last N % 32 columns run the same SIMD body under
  AVX-512BW masked loads/stores — no scalar tail, no table spills to
  memory, and no performance cliff at vector boundaries (N=2047 runs
  within ~20% of N=2048, vs ~60% slower with the earlier scalar tail).
- **Table construction**: with n = |v| + 121 = 9(h+13) + (l+4), where h is
  the top-3-trit value and l the bottom-2-trit value, T[m] = H[n/9] +
  L[n%9]. H (27 entries) and L (9 entries) are built lane-wise as
  broadcast-multiply-adds against constant digit vectors (H = a0·D0 +
  a1·D1 + a2·D2), and n/9, n%9 per table lane are compile-time constants,
  so the four table registers cost ~10 arithmetic ops plus eight
  constant-index `vpermw` — no scalar work, no divide in the data path.
- **Accumulation**: int16 partial sums, flushed to the int32 output every
  48 groups (48 × max|entry| 640 = 30720 < 32767, no overflow); the first
  flush stores instead of adds, so the output is never pre-zeroed.

### Why gathers lose and shuffles win (the BitNet connection)

Microsoft BitNet's TL2 kernel caps lookup indices at 4 bits (+1 sign bit)
so they fit AVX2 `pshufb`, paying 1.67 bits/weight. A 243-state byte can't
be shuffle-indexed on AVX2, and `vpgatherdd` manages only ~1 lookup/cycle
on Zen, which is why the gather kernel ties the scalar one. The mirror
trick brings the index space to 122 ≤ 128, which AVX-512's two-register
word permute handles natively — so the 5-trit format keeps its full 1.6
bits/weight density *and* gets shuffle-speed lookups (~10 lookups/cycle
measured), with a decode cost of just `vpabsw` + one mask per 32 weights.

On AVX2-only hardware the equivalent path is splitting n into n/9 (27
states → sign + 14 magnitudes ≤ 16) and n%9 (9 states → sign + 5) for
`pshufb`, at the cost of a multiply-by-57-shift-9 divmod in the kernel —
implemented here only for AVX-512, where it isn't needed.

## License

MIT
