// SPDX-License-Identifier: GPL-3.0-or-later
//
// Experiment harness: verifies every LUT implementation against a naive
// dense GEMM (exact integer match) and benchmarks all of them.
//
// Usage: bench [-m M] [-k K] [-n N] [-r reps] [--seed S]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "ternary_mm.h"

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace {

bool cpu_has_avx512bw() {
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuid(info, 1);
    if (!(info[2] & (1 << 27))) return false;  // OSXSAVE
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0xE6) != 0xE6) return false;  // XMM, YMM, opmask, ZMM state
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 16)) && (info[1] & (1 << 30));  // AVX512F, BW
#else
    return __builtin_cpu_supports("avx512bw");
#endif
}

double time_ms(const std::function<void()>& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct Stats {
    double min_ms;
    double median_ms;
};

Stats bench(const std::function<void()>& fn, int reps) {
    fn();  // warmup
    std::vector<double> times(reps);
    for (auto& t : times) t = time_ms(fn);
    std::sort(times.begin(), times.end());
    return {times.front(), times[reps / 2]};
}

// Returns the number of mismatching elements; prints the first one.
size_t verify(const char* name, const std::vector<int32_t>& got,
              const std::vector<int32_t>& want, int N) {
    size_t bad = 0;
    for (size_t x = 0; x < got.size(); ++x) {
        if (got[x] != want[x]) {
            if (bad == 0) {
                std::printf("  %s MISMATCH at [%zu][%zu]: got %d, want %d\n",
                            name, x / N, x % N, got[x], want[x]);
            }
            ++bad;
        }
    }
    return bad;
}

int arg_int(int argc, char** argv, const char* flag, int def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return std::atoi(argv[i + 1]);
    return def;
}

const char* arg_str(int argc, char** argv, const char* flag) {
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    const int M = arg_int(argc, argv, "-m", 256);
    const int K = arg_int(argc, argv, "-k", 2000);
    const int N = arg_int(argc, argv, "-n", 2048);
    const int reps = arg_int(argc, argv, "-r", 5);
    const int seed = arg_int(argc, argv, "--seed", 42);

    if (K % 5 != 0) {
        std::fprintf(stderr, "K (%d) must be a multiple of 5\n", K);
        return 1;
    }
    if (M <= 0 || N <= 0 || K <= 0 || reps <= 0) {
        std::fprintf(stderr, "sizes and reps must be positive\n");
        return 1;
    }

    std::printf("M=%d K=%d N=%d reps=%d seed=%d\n", M, K, N, reps, seed);
    std::printf("packed weights: %d bytes vs %d unpacked (%.1fx smaller)\n\n",
                K / 5 * N, K * N, 5.0);

    // Separate streams so W depends only on (K, N, seed) — prepacked
    // weight files (--dump-w / --bitnet) stay valid across different M.
    std::mt19937 rng_a(seed);
    std::mt19937 rng_w(seed + 0x9e3779b9);
    std::uniform_int_distribution<int> act_dist(-128, 127);
    std::uniform_int_distribution<int> tern_dist(-1, 1);

    std::vector<int8_t> A((size_t)M * K);
    std::vector<int8_t> W((size_t)K * N);
    std::vector<int8_t> P((size_t)K / 5 * N);
    for (auto& v : A) v = (int8_t)act_dist(rng_a);
    for (auto& v : W) v = (int8_t)tern_dist(rng_w);

    const double pack_ms = time_ms([&] { pack_weights(W.data(), K, N, P.data()); });
    std::printf("pack_weights: %.2f ms (one-time)\n\n", pack_ms);

    // --dump-w: write the raw weight matrix for tools/pack_tl2.py and exit.
    // The same seed regenerates identical weights on the benchmark run.
    if (const char* dump = arg_str(argc, argv, "--dump-w")) {
        FILE* f = std::fopen(dump, "wb");
        if (!f || std::fwrite(W.data(), 1, W.size(), f) != W.size()) {
            std::fprintf(stderr, "failed writing %s\n", dump);
            return 1;
        }
        std::fclose(f);
        std::printf("dumped %zu weight bytes to %s\n", W.size(), dump);
        return 0;
    }

    // --bitnet <file>: add microsoft/BitNet's TL2 kernel using the packed
    // weight blob produced by tools/pack_tl2.py.
    std::vector<uint8_t> bitnet_qw;
    std::vector<float> Bf;
    if (const char* path = arg_str(argc, argv, "--bitnet")) {
        if (!bitnet_tl2_supported(K, N)) {
            std::fprintf(stderr,
                         "--bitnet: shape K=%d N=%d not in the generated "
                         "kernels (use K=2080 N=2048 or K=4160 N=4096)\n",
                         K, N);
            return 1;
        }
        FILE* f = std::fopen(path, "rb");
        if (!f) {
            std::fprintf(stderr, "cannot open %s\n", path);
            return 1;
        }
        std::fseek(f, 0, SEEK_END);
        bitnet_qw.resize((size_t)std::ftell(f));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(bitnet_qw.data(), 1, bitnet_qw.size(), f) !=
            bitnet_qw.size()) {
            std::fprintf(stderr, "failed reading %s\n", path);
            return 1;
        }
        std::fclose(f);
        // Their pipeline consumes float activations; converted outside the
        // timed region (a concession in their favor).
        Bf.resize(A.size());
        for (size_t x = 0; x < A.size(); ++x) Bf[x] = (float)A[x];
    }

    std::vector<int32_t> C_ref((size_t)M * N);
    std::vector<int32_t> C((size_t)M * N);

    struct Impl {
        const char* name;
        std::function<void(int32_t*)> run;
    };
    std::vector<Impl> impls = {
        {"naive_mm",
         [&](int32_t* out) { naive_mm(A.data(), W.data(), M, K, N, out); }},
    };
    if (cpu_has_avx512bw()) {
        impls.push_back({"lut_mm_avx512", [&](int32_t* out) {
                             lut_mm_avx512(A.data(), P.data(), M, K, N, out);
                         }});
    } else {
        std::printf("lut_mm_avx512: skipped (CPU lacks AVX-512BW)\n\n");
    }
    if (!bitnet_qw.empty()) {
        impls.push_back({"bitnet_tl2", [&](int32_t* out) {
                             lut_mm_bitnet_tl2(Bf.data(), bitnet_qw.data(),
                                               M, K, N, out);
                         }});
        if (cpu_has_avx512bw()) {
            impls.push_back({"bitnet_tl2@512b", [&](int32_t* out) {
                                 lut_mm_bitnet_tl2_512(
                                     Bf.data(), bitnet_qw.data(), M, K, N,
                                     out);
                             }});
        }
    }

    // Accuracy: naive_mm is ground truth; every other impl must match exactly.
    std::printf("accuracy check (vs naive_mm, exact int32 match):\n");
    impls[0].run(C_ref.data());
    bool all_ok = true;
    for (size_t s = 1; s < impls.size(); ++s) {
        impls[s].run(C.data());
        const size_t bad = verify(impls[s].name, C, C_ref, N);
        std::printf("  %-14s %s\n", impls[s].name,
                    bad ? "FAIL" : "PASS");
        if (bad) {
            std::printf("    (%zu of %zu elements wrong)\n", bad, C.size());
            all_ok = false;
        }
    }
    if (!all_ok) {
        std::printf("\naccuracy failed; skipping benchmark\n");
        return 1;
    }

    const double gops = 2.0 * M * N * K / 1e9;
    std::printf("\nbenchmark (%.2f Gop, best of %d):\n", gops, reps);
    std::printf("  %-14s %10s %10s %8s %9s\n", "impl", "min ms", "median ms",
                "Gop/s", "speedup");
    double naive_min = 0.0;
    for (const auto& impl : impls) {
        const Stats st = bench([&] { impl.run(C.data()); }, reps);
        if (impl.name == impls[0].name) naive_min = st.min_ms;
        std::printf("  %-14s %10.2f %10.2f %8.1f %8.2fx\n", impl.name,
                    st.min_ms, st.median_ms, gops / (st.min_ms / 1e3),
                    naive_min / st.min_ms);
    }
    return 0;
}
