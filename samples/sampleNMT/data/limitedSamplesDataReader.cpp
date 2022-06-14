/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "limitedSamplesDataReader.h"

#include <algorithm>
#include <sstream>

namespace nmtSample
{
LimitedSamplesDataReader::LimitedSamplesDataReader(int32_t maxSamplesToRead, DataReader::ptr originalDataReader)
    : gMaxSamplesToRead(maxSamplesToRead)
    , gOriginalDataReader(originalDataReader)
    , gCurrentPosition(0)
{
}

int32_t LimitedSamplesDataReader::read(
    int32_t samplesToRead, int32_t maxInputSequenceLength, int32_t* hInputData, int32_t* hActualInputSequenceLengths)
{
    int32_t limitedSmplesToRead = std::min(samplesToRead, std::max(gMaxSamplesToRead - gCurrentPosition, 0));
    int32_t samplesRead = gOriginalDataReader->read(
        limitedSmplesToRead, maxInputSequenceLength, hInputData, hActualInputSequenceLengths);
    gCurrentPosition += samplesRead;
    return samplesRead;
}

void LimitedSamplesDataReader::reset()
{
    gOriginalDataReader->reset();
    gCurrentPosition = 0;
}

std::string LimitedSamplesDataReader::getInfo()
{
    std::stringstream ss;
    ss << "Limited Samples Reader, max samples = " << gMaxSamplesToRead
       << ", original reader info: " << gOriginalDataReader->getInfo();
    return ss.str();
}
} // namespace nmtSample
