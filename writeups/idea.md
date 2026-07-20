title: Key Idea of BitNet (ternary edition). Matrix Multiply using Lookup Tables
link: ternary-mm
published_date: 2026-06-11 15:11

*This example is inspired by [BitNet b1.58](https://arxiv.org/abs/2402.17764), which uses ternary weights: $-1$, $0$, and $+1$. Actual packing structure and details aren't meant to be the same.*

Lookup-based matrix multiplication is not a new idea, but it is useful here because it allows compact quantized weights to be used directly during CPU inference.

Key ideas:

- Multiple quantized weights can be packed into one byte, reducing storage and memory use.
- During matrix multiplication, the packed weights do not need to be unpacked. The byte can be used directly as an index into a lookup table.

In a follow-up post, we will show that the bigger tables shown here are actually feasible with AVX-512 and can get many times the performance of the AVX2 implementations in the official [BitNet Repository](https://github.com/microsoft/BitNet).

## Packing five ternary weights into one byte

### It can be packed!

Five ternary weights have $3^5 = 243$ possible combinations.

One byte has $2^8 = 256$ possible values. This means one byte is enough to represent 5 trits.

### Encoding a group of weights

Consider five weights:

```text
[-1, 0, 1, 1, -1]
```

Treat them as the digits of a balanced ternary number. The place values are powers of three:

| Position | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| Place value | 81 | 27 | 9 | 3 | 1 |
| Weight | -1 | 0 | 1 | 1 | -1 |

The packed value is

$$
(-1)(81) + (0)(27) + (1)(9) + (1)(3) + (-1)(1) = -70
$$

Each weight is one of $-1$, $0$, or $1$. The packed code therefore ranges from $-121$ to $121$, giving exactly 243 possible values.

## Using the packed byte as a lookup index

Suppose the current group of five input values is

```text
[1, 2, 3, 4, 5]
```

We build a table containing the dot product of these inputs with every possible five-weight pattern. A small section of that table looks like this:

| Packed code | Ternary weights        | Dot product |
| ----------: | ---------------------- | ----------: |
|      $-121$ | `[-1, -1, -1, -1, -1]` |       $-15$ |
|      $-120$ | `[-1, -1, -1, -1, 0]`  |       $-10$ |
|      $-119$ | `[-1, -1, -1, -1, 1]`  |        $-5$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|        $-1$ | `[0, 0, 0, 0, -1]`     |        $-5$ |
|         $0$ | `[0, 0, 0, 0, 0]`      |         $0$ |
|         $1$ | `[0, 0, 0, 0, 1]`      |         $5$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|       $119$ | `[1, 1, 1, 1, -1]`     |         $5$ |
|       $120$ | `[1, 1, 1, 1, 0]`      |        $10$ |
|       $121$ | `[1, 1, 1, 1, 1]`      |        $15$ |


## Applying this to matrix multiplication

Let's walk through a small example: $y = xW$ with 10 inputs and several output columns.

### Setup

Inputs, in two groups of five:

```text
x = [1 2 3 4 5 | 6 7 8 9 10]
      group 0      group 1
```

Weights, one column per output:

```text
            col 0   col 1   col 2   col 3   col 4   col 5    ...
group 0:     -1       0       1       1       0      -1      ...
              0       0      -1       1       1       1
              1       0       1       1      -1       0
              1       0      -1       1       0       1
             -1       0       1       1       1       0
group 1:      1       1      -1      -1       1       0      ...
              1       1       0      -1       0      -1
              0       1       0      -1       1       1
             -1       1       0      -1       0       1
              0       1       1      -1      -1       0
```

Each five-weight block is packed into one byte, and these bytes are the only form of the weights kept in memory:

| Packed codes | col 0 | col 1 | col 2 | col 3 | col 4 | col 5 | $\cdots$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| group 0 | $-70$ | $0$ | $61$ | $121$ | $19$ | $-51$ | $\cdots$ |
| group 1 | $105$ | $121$ | $-80$ | $-121$ | $89$ | $-15$ | $\cdots$ |

### Build one table per input group

When the input vector arrives, build the full 243-entry lookup table for each group of five inputs, exactly as in the previous section.

Table 0 contains the dot products with the group-0 inputs `[1, 2, 3, 4, 5]`:

| Packed code | Ternary weights        | Dot product |
| ----------: | ---------------------- | ----------: |
|      $-121$ | `[-1, -1, -1, -1, -1]` |       $-15$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|       $-70$ | `[-1, 0, 1, 1, -1]`    |         $1$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|       $-51$ | `[-1, 1, 0, 1, 0]`     |         $5$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|         $0$ | `[0, 0, 0, 0, 0]`      |         $0$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|        $19$ | `[0, 1, -1, 0, 1]`     |         $4$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|        $61$ | `[1, -1, 1, -1, 1]`    |         $3$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|       $121$ | `[1, 1, 1, 1, 1]`      |        $15$ |

Table 1 contains the dot products with the group-1 inputs `[6, 7, 8, 9, 10]`:

| Packed code | Ternary weights        | Dot product |
| ----------: | ---------------------- | ----------: |
|      $-121$ | `[-1, -1, -1, -1, -1]` |       $-40$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|       $-80$ | `[-1, 0, 0, 0, 1]`     |         $4$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|       $-15$ | `[0, -1, 1, 1, 0]`     |        $10$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|        $89$ | `[1, 0, 1, 0, -1]`     |         $4$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|       $105$ | `[1, 1, 0, -1, 0]`     |         $4$ |
|    $\vdots$ | `...`                  |    $\vdots$ |
|       $121$ | `[1, 1, 1, 1, 1]`      |        $40$ |

### Look up and accumulate

Now sweep across the packed bytes. Each byte costs one table lookup and one add:

```text
group 0 codes: [-70    0   61  121   19  -51]  →  table 0 gives [ 1   0   3   15   4   5]
group 1 codes: [105  121  -80 -121   89  -15]  →  table 1 gives [ 4  40   4  -40   4  10]
                                                       sum  y = [ 5  40   7  -25   8  15]
```

Sanity check on column 0, the slow way:

$$
\underbrace{-1+0+3+4-5}_{\text{group 0}} + \underbrace{6+7+0-9+0}_{\text{group 1}} = 1 + 4 = 5  \huge \checkmark
$$

The weights were never unpacked: each byte went straight from memory into a table index.

### A few questions you might have

**Isn't building a 243-entry table expensive?** It's built once per input group and then reused by *every* output column. A real weight matrix has thousands of columns, so the build cost is negligible.

**What about those negative indices?** The table has a symmetry: flipping all five weights negates both the code and the dot product, so $T[-c] = -T[c]$. The implementation stores only the non-negative half and applies the sign separately.

**How dense is the packing?** 8 bits for 5 weights is 1.6 bits per weight, close to the theoretical floor of $\log_2 3 \approx 1.585$ — and denser than the 2.0 bits of a straightforward 2-bits-per-trit encoding.
