// Minimal shim replacing BitNet's ggml-bitnet.h so the generated TL2
// kernels compile standalone, without the llama.cpp/ggml dependency.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#if defined(_WIN32)
#include <malloc.h>
#endif

typedef float bitnet_float_type;

enum ggml_type {
    GGML_TYPE_Q4_0,
    GGML_TYPE_TL2,
};

struct bitnet_tensor_extra {
    int lut_scales_size;
    int BK;
    int n_tile_num;
    uint8_t* qweights;
    bitnet_float_type* scales;
};
