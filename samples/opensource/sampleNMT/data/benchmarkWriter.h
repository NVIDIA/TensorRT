/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SAMPLE_NMT_BENCHMARK_WRITER_
#define SAMPLE_NMT_BENCHMARK_WRITER_

#include <chrono>
#include <memory>

#include "dataWriter.h"

namespace nmtSample
{
/** \class BenchmarkWriter
 *
 * \brief all it does is to measure the performance of sequence generation
 *
 */
class BenchmarkWriter : public DataWriter
{
public:
    BenchmarkWriter();

    void write(const int* hOutputData, int actualOutputSequenceLength, int actualInputSequenceLength) override;

    void initialize() override;

    void finalize() override;

    std::string getInfo() override;

    ~BenchmarkWriter() override = default;

private:
    int mSampleCount;
    int mInputTokenCount;
    int mOutputTokenCount;
    std::chrono::high_resolution_clock::time_point mStartTS;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_BENCHMARK_WRITER_
