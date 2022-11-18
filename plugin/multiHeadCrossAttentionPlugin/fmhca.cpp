/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fmhca.h"

namespace nvinfer1
{
namespace plugin
{
static void set_alpha(uint32_t& alpha, float norm, Data_type dtype)
{
    if (dtype == DATA_TYPE_FP16)
    {
        half x = __float2half_rn(norm);
        uint16_t h = reinterpret_cast<uint16_t const&>(x);
        ushort2 h2 = {h, h};
        alpha = reinterpret_cast<uint32_t const&>(h2);
    }
    else if (dtype == DATA_TYPE_FP32)
    {
        alpha = reinterpret_cast<uint32_t const&>(norm);
    }
    else if (dtype == DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = reinterpret_cast<uint32_t const&>(inorm);
    }
    else
    {
        assert(false);
    }
}

static void set_params(Fused_multihead_attention_params_mhca& params,
    // types
    Data_type data_type, Data_type acc_type,
    // sizes
    size_t b, size_t s_q, size_t s_kv, size_t h, size_t d, size_t total,
    // device pointers
    void* q_packed_d, void* kv_packed_d, void* cu_seqlens_q_d, void* cu_seqlens_kv_d, void* o_packed_d, void* p_d,
    void* s_d,
    // scale factors
    float scale_bmm1, float scale_softmax, float scale_bmm2,
    // flags
    bool interleaved, bool ignore_b1opt, bool force_unroll, bool use_int8_scale_max,
    bool use_tma)
{
    memset(&params, 0, sizeof(params));

    int32_t const d_padded = std::pow(2, std::ceil(std::log(d) / std::log(2)));

    // Set the pointers.
    params.o_ptr = o_packed_d;
    params.o_stride_in_bytes = get_size_in_bytes(h * d, data_type);

#if defined(STORE_P)
    params.p_ptr = p_d;
    params.p_stride_in_bytes = get_size_in_bytes(b * h * s_kv, acc_type);
#endif // defined(STORE_P)

#if defined(STORE_S)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s_kv, data_type);
#endif // defined(STORE_S)

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s_q = s_q;
    params.s = s_kv;
    params.d = d;
    params.d_padded = d_padded;

    // Set the different scale values.
    Data_type scale_type1 = data_type == DATA_TYPE_FP16 ? acc_type : DATA_TYPE_FP32;
    Data_type scale_type2 = data_type == DATA_TYPE_FP16 ? DATA_TYPE_FP16 : DATA_TYPE_FP32;

    set_alpha(params.scale_bmm1, scale_bmm1, scale_type1);
    set_alpha(params.scale_softmax, scale_softmax, scale_type1);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);

    // Set the pointers.
    params.gmem_q_params.ptr = q_packed_d;
    params.gmem_q_params.stride_in_bytes = get_size_in_bytes(h * d, data_type);
    params.gmem_q_params.h = h;
    params.gmem_q_params.d = d;
    params.gmem_q_params.cu_seqlens = static_cast<int32_t*>(cu_seqlens_q_d);

    params.gmem_kv_params.ptr = kv_packed_d;
    params.gmem_kv_params.stride_in_bytes = get_size_in_bytes(h * 2 * d, data_type);
    params.gmem_kv_params.h = h;
    params.gmem_kv_params.d = d;
    params.gmem_kv_params.cu_seqlens = static_cast<int32_t*>(cu_seqlens_kv_d);

    // Set flags
    params.interleaved = interleaved;
    params.ignore_b1opt = ignore_b1opt;
    params.force_unroll = force_unroll;
    params.use_int8_scale_max = use_int8_scale_max;

    // Do we enable the trick to replace I2F with FP math in the 2nd GEMM?
    if (data_type == DATA_TYPE_INT8)
    {
        params.enable_i2f_trick
            = -double(1 << 22) * double(scale_bmm2) <= -128.f && double(1 << 22) * double(scale_bmm2) >= 127.f;
    }
}

int32_t run_fmhca_api(void* q_packed_d, void* kv_packed_d, void* cu_seqlens_q_d, void* cu_seqlens_kv_d,
    void* o_packed_d, int32_t sm, FusedMultiHeadCrossAttentionKernel const* kernels, size_t b, size_t h, size_t d,
    size_t s_q, size_t s_kv, cudaStream_t stream)
{

    // The data type of the kernel.
    Data_type data_type = DATA_TYPE_FP16;
    // The type of the intermediate P matrix.
    Data_type acc_type = DATA_TYPE_FP16;

    bool const force_unroll = true;
    bool const interleaved = false;
    bool const ignore_b1opt = false;
    bool const use_int8_scale_max = false;
    bool const use_tma = false;
    void* p_d = nullptr;
    void* s_d = nullptr;
    size_t total = 0; // used only for interleaved, which is false

    float scale_bmm1 = 1.f / sqrtf(d);
    float scale_softmax = 1.f;
    float scale_bmm2 = 1.f;

    // Set the params.
    Fused_multihead_attention_params_mhca params{};
    set_params(params, data_type, acc_type, b, s_q, s_kv, h, d, total, q_packed_d, kv_packed_d, cu_seqlens_q_d,
        cu_seqlens_kv_d, o_packed_d, p_d, s_d, scale_bmm1, scale_softmax, scale_bmm2, interleaved, ignore_b1opt,
        force_unroll, use_int8_scale_max, use_tma);

    kernels->run(params, stream);

    return 0;
}
} // namespace plugin
} // namespace nvinfer1
