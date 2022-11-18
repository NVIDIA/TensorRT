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

#include "fmhaPlugin.h"
#include "fmha.h"

namespace nvinfer1
{
namespace plugin
{
PluginFieldCollection fmhaPluginCreator::mFc{};
std::vector<PluginField> fmhaPluginCreator::mPluginAttributes;

int32_t fmhaPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // input[ 0]:  [float16],  (b, s, h, 3, d)
    // output[0]:  [float16],  (b,s,h,d)
    int32_t result{-1};
    try
    {
        PLUGIN_ASSERT(mKernels);
        PLUGIN_ASSERT(mSM);
        PLUGIN_ASSERT(mCuSeqLen);

        // update cuseqlens when bs or seq changed.
        int32_t const batchSize = inputDesc[0].dims.d[0];
        int32_t const seqLen = inputDesc[0].dims.d[1];
        if (batchSize != m_.mOptBatchSize || seqLen != m_.mOptSeqLen)
        {
            initializeSeqlens(batchSize, seqLen, mCuSeqLen.get(), stream);
        }

        // launch kernel.
        int32_t const head_num = inputDesc[0].dims.d[2];
        int32_t const size_per_head = inputDesc[0].dims.d[4];
        size_t const total = m_.mOptBatchSize * m_.mOptSeqLen;
        result = run_fmha_v2_api((void*) inputs[0], (void*) mCuSeqLen.get(), (void*) outputs[0], total, mSM, mKernels,
            (size_t) m_.mOptBatchSize, (size_t) head_num, (size_t) size_per_head,
            (size_t) m_.mOptSeqLen, stream);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return result;
}

REGISTER_TENSORRT_PLUGIN(fmhaPluginCreator);

} // namespace plugin
} // namespace nvinfer1
