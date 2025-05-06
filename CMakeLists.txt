cmake_minimum_required(VERSION 3.30.5)
project(bnet)

set(CMAKE_CXX_STANDARD 26)

# Add optimization flags
if(MSVC)
    # Visual Studio compiler flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Oi /Ot /GL /arch:AVX2")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /O2 /Ob1")
    # Enable parallel compilation
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    # Enable link-time optimization
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
else()
    # GCC/Clang compiler flags
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O2")
    # Enable link-time optimization
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
endif()

add_executable(bnet main.cpp
        lut_mm.cpp
        lut_mm.h
        lut_mm_optimized.cpp
        lut_mm_optimized.h
        permutations.cpp
        permutations.h
        weight_packing.cpp
        weight_packing.h
)
