#pragma once
#ifndef _FMHCA_
#define _FMHCA_

#include "fmha_cross_attention/include/fmha_cross_attention.h"
#include <stdio.h>
#include <stdlib.h>

namespace nvinfer1
{
namespace plugin
{
int32_t run_fmhca_api(void* q_packed_d, void* kv_packed_d, void* cu_seqlens_q_d, void* cu_seqlens_kv_d,
    void* o_packed_d, int32_t sm, FusedMultiHeadCrossAttentionKernel const* kernels, size_t b = 2, size_t h = 8,
    size_t d = 64, size_t s_q = 4096, size_t s_kv = 77, cudaStream_t stream = 0);
}
} // namespace nvinfer1

#endif
