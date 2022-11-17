#pragma once
#ifndef _FMHA_
#define _FMHA_

#include "fmha_flash_attention/include/fmha_flash_attention.h"
#include <stdio.h>
#include <stdlib.h>

namespace nvinfer1
{
namespace plugin
{

int run_fmha_v2_api(void* qkv_packed_d, void* cu_seqlens_d, void* o_packed_d, size_t total, int32_t sm,
    FusedMultiHeadFlashAttentionKernel const* kernels, size_t b = 2, size_t h = 8, size_t d = 64, size_t s = 4096,
    cudaStream_t stream = 0);

}
} // namespace nvinfer1

#endif
