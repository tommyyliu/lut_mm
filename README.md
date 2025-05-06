# BNet - Optimized Matrix Multiplication

This project implements and compares different approaches to matrix multiplication with a focus on performance optimization:

1. **Basic Matrix Multiplication**: Standard implementation of matrix multiplication
2. **Optimized Basic Matrix Multiplication**: Enhanced version with multi-threading, cache optimization, and loop tiling
3. **LUT-based Matrix Multiplication**: Uses lookup tables and weight packing for efficient computation
4. **Optimized LUT-based Matrix Multiplication**: Combines LUT approach with advanced optimization techniques

## Features

- Ternary weight matrices (-1, 0, 1) with efficient packing (5x memory reduction)
- Lookup table-based computation for faster matrix multiplication
- Multi-threaded implementation for parallel processing
- Detailed performance benchmarking and profiling
- Memory usage optimization

## Performance Results

For 2000x2000 matrices:

| Algorithm | Time (s) | Speedup vs Basic |
|-----------|----------|------------------|
| Basic MM | 26.722 | 1.00x |
| Optimized Basic MM | 4.543 | 5.88x |
| LUT-MM | 10.225 | 2.61x |
| Optimized LUT-MM | 2.028 | 13.18x |

Including preparation time:

| Algorithm | Time (s) | Speedup vs Basic |
|-----------|----------|------------------|
| Basic MM | 26.722 | 1.00x |
| Optimized Basic MM | 4.543 | 5.88x |
| LUT-MM with preparation | 11.164 | 2.39x |
| Optimized LUT-MM with prep | 2.651 | 10.08x |

## Building the Project

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Running the Benchmarks

```bash
cd build/Release
./bnet
```

## Implementation Details

- **Weight Packing**: Compresses 5 ternary weights into a single byte
- **Lookup Tables**: Pre-computes dot products for all possible permutations
- **Optimizations**: 
  - Multi-threading
  - Cache-friendly memory access
  - Loop tiling and unrolling
  - SIMD-friendly code structure

## License

MIT
