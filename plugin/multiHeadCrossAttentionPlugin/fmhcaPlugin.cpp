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

#include "fmhcaPlugin.h"
#include "fmhca.h"

using namespace nvinfer1;
using namespace plugin;

PluginFieldCollection fmhcaPluginCreator::mFC{};
std::vector<PluginField> fmhcaPluginCreator::mPluginAttributes;

int32_t fmhcaPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int32_t result{-1};
    try
    {
        PLUGIN_ASSERT(mKernels);
        PLUGIN_ASSERT(mSM);
        PLUGIN_ASSERT(mCuSeqLensQ);
        PLUGIN_ASSERT(mCuSeqLensKV);

        constexpr int32_t seqLenKvPadded = 128;
        int32_t const batchSize = inputDesc[0].dims.d[0];
        int32_t const seqLenQ = inputDesc[0].dims.d[1];
        int32_t const seqLenKV = inputDesc[1].dims.d[1];
        int32_t const headNum = inputDesc[0].dims.d[2];
        int32_t const sizePerHead = inputDesc[0].dims.d[3];

        // Check for seq len to support dynamic input shape
        if (sizePerHead <= 64)
        {
            PLUGIN_VALIDATE(seqLenQ % 64 == 0, "Not support q buffer sequence length not multiple of 64 when head size < 64 for plugin fMHCA");
        }
        else if (sizePerHead <= 128)
        {
            PLUGIN_VALIDATE(seqLenQ % 32 == 0, "Not support q buffer sequence length not multiple of 32 when head size between 64 and 128 for plugin fMHCA");
        }
        else
        {
            PLUGIN_VALIDATE(seqLenQ % 16 == 0, "Not support q buffer sequence length not multiple of 16 when head size > 128 for plugin fMHCA");
        }

        if (batchSize != m_.mOptBatchSize || m_.mOptSeqLenQ != seqLenQ || m_.mOptSeqLenKV != seqLenKV)
        {
            m_.mOptSeqLenQ = initializeSeqlens(batchSize, seqLenQ, mCuSeqLensQ.get(), stream);
            m_.mOptSeqLenKV = initializeSeqlens(batchSize, seqLenKV, mCuSeqLensKV.get(), stream);
        }

        result = run_fmhca_api((void*) inputs[0], (void*) inputs[1], mCuSeqLensQ.get(), mCuSeqLensKV.get(), (void*) outputs[0],
            mSM, mKernels, static_cast<size_t>(batchSize), static_cast<size_t>(headNum),
            static_cast<size_t>(sizePerHead), static_cast<size_t>(seqLenQ), static_cast<size_t>(seqLenKvPadded),
            stream);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return result;
}

REGISTER_TENSORRT_PLUGIN(fmhcaPluginCreator);
