/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TENSORRT_DEBUG_TENSOR_WRITER_H
#define TENSORRT_DEBUG_TENSOR_WRITER_H

#include "NvInferRuntime.h"
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
namespace sample
{

class DebugTensorWriter : public nvinfer1::IDebugListener
{
public:
    DebugTensorWriter(std::unordered_map<std::string, std::string> const& debugTensorFileNames,
        std::vector<std::string> const& debugTensorFormats, std::string const& engineName = "",
        std::string const& cmdline = "");
    ~DebugTensorWriter() override;

    bool processDebugTensor(void const* addr, nvinfer1::TensorLocation location, nvinfer1::DataType type,
        nvinfer1::Dims const& shape, char const* name, cudaStream_t stream) override;

private:
    void writeSummaryHeader();
    void writeSummaryFooter();
    void writeSummary(std::string_view name, nvinfer1::Dims const& shape, nvinfer1::DataType type, int64_t volume,
        void const* addr_host, std::string_view assignedFileName, std::string_view numpyFileName,
        std::string_view stringFileName, std::string_view rawFileName);

    std::unordered_map<std::string, std::string> mDebugTensorFileNames;
    std::vector<std::string> mDebugTensorFormats;
    std::string mSummaryFileName;
    std::ofstream mSummaryFile;
    bool mFirstTensor{true};
    std::string mEngineName;
    std::string mCmdline;
    int32_t mTensorIndex{0};
};

} // namespace sample

#endif // TENSORRT_DEBUG_TENSOR_WRITER_H
