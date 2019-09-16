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

#include "benchmarkWriter.h"
#include "logger.h"

#include <iostream>

namespace nmtSample
{
BenchmarkWriter::BenchmarkWriter()
    : mSampleCount(0)
    , mInputTokenCount(0)
    , mOutputTokenCount(0)
    , mStartTS(std::chrono::high_resolution_clock::now())
{
}

void BenchmarkWriter::write(const int* hOutputData, int actualOutputSequenceLength, int actualInputSequenceLength)
{
    ++mSampleCount;
    mInputTokenCount += actualInputSequenceLength;
    mOutputTokenCount += actualOutputSequenceLength;
}

void BenchmarkWriter::initialize()
{
    mStartTS = std::chrono::high_resolution_clock::now();
}

void BenchmarkWriter::finalize()
{
    std::chrono::duration<float> sec = std::chrono::high_resolution_clock::now() - mStartTS;
    int totalTokenCount = mInputTokenCount + mOutputTokenCount;
    gLogInfo << mSampleCount << " sequences generated in " << sec.count() << " seconds, "
             << (mSampleCount / sec.count()) << " samples/sec" << std::endl;
    gLogInfo << totalTokenCount << " tokens processed (source and destination), " << (totalTokenCount / sec.count())
             << " tokens/sec" << std::endl;
}

std::string BenchmarkWriter::getInfo()
{
    return "Benchmark Writer";
}
} // namespace nmtSample
