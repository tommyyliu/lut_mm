cmake_minimum_required(VERSION 3.21)
project(lut_mm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bench
    src/bench_main.cpp
    src/ternary_mm_ref.cpp
    src/ternary_mm_avx512.cpp
    src/ternary_mm_vnni.cpp
    src/ternary_mm_bitnet.cpp
    src/ternary_mm_bitnet512.cpp
)

# Arch flags are per-file so each kernel is compiled exactly for its target
# ISA (and naive_mm's AVX2 auto-vectorized baseline stays comparable).
if(MSVC)
    target_compile_options(bench PRIVATE /W4)
    set_source_files_properties(src/ternary_mm_ref.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/ternary_mm_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    set_source_files_properties(src/ternary_mm_vnni.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
    # /wd4068: the generated BitNet code uses #pragma unroll
    set_source_files_properties(src/ternary_mm_bitnet.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/wd4068")
    set_source_files_properties(src/ternary_mm_bitnet512.cpp
        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    # TODO(non-MSVC): this GCC/Clang branch has never been built — the dev
    # machine is Windows/MSVC-only. Before relying on it, build on Linux
    # (gcc or clang, needs AVX-512 VNNI; the kernels gate at runtime so a
    # non-AVX-512 host still builds and skips), and consider a minimal CI
    # workflow (ubuntu gcc + windows msvc, build + `bench -r 1`) so it stays
    # green.
    target_compile_options(bench PRIVATE -Wall -Wextra)
    set_source_files_properties(src/ternary_mm_ref.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(src/ternary_mm_avx512.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw")
    set_source_files_properties(src/ternary_mm_vnni.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni")
    set_source_files_properties(src/ternary_mm_bitnet.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(src/ternary_mm_bitnet512.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw")
endif()
