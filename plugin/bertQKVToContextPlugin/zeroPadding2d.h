/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_ZERO_PADDING_2D_H
#define TRT_ZERO_PADDING_2D_H
#include "NvInferPlugin.h"
#include <cuda_runtime.h>

namespace nvinfer1
{
namespace plugin
{
namespace bert
{

struct MhaRunParameter
{
    nvinfer1::PluginTensorDesc inputDesc[4];
    nvinfer1::PluginTensorDesc outputDesc[1];
    void* inputs[4];
    void* outputs[1];
};

class QkvPaddingRunner
{
public:
    QkvPaddingRunner(nvinfer1::DataType dtype, int32_t maxPadSize = 64);
    ~QkvPaddingRunner() = default;
    int32_t getMaxPaddingHeadSize();
    size_t getWorkspaceSize(int32_t sumSeqLen, int32_t numHeads);
    void* get16BytesAlignedPointer(void* workspace, size_t offset);
    cudaError_t pad(void const* src, void* workspace, int32_t sumSeqLen, int32_t numHeads, int32_t headSize,
        int32_t padHeadSize, cudaStream_t stream);
    cudaError_t unpad(void const* workspace, void* dst, int32_t sumSeqLen, int32_t numHeads, int32_t headSize,
        int32_t padHeadSize, cudaStream_t stream);
    MhaRunParameter patchMhaArgs(nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
        void* paddingWorkspace, int32_t sumSeqLen, int32_t numHeads, int32_t padHeadSize);

private:
    size_t getInputSize(int32_t sumSeqLen, int32_t numHeads);
    size_t getOutputSize(int32_t sumSeqLen, int32_t numHeads);
    int32_t mMaxPaddingHeadSize{};
    int32_t mDtypeSize{};
};

} // namespace bert
} // namespace plugin
} // namespace nvinfer1
#endif // TRT_ZERO_PADDING_2D_H
