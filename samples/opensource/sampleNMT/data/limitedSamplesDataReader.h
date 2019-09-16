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
#ifndef SAMPLE_NMT_LIMITED_SAMPLES_DATA_READER_
#define SAMPLE_NMT_LIMITED_SAMPLES_DATA_READER_

#include "dataReader.h"

namespace nmtSample
{
/** \class LimitedSamplesDataReader
 *
 * \brief wraps another data reader and limits the number of samples to read
 *
 */
class LimitedSamplesDataReader : public DataReader
{
public:
    LimitedSamplesDataReader(int maxSamplesToRead, DataReader::ptr originalDataReader);

    int read(int samplesToRead, int maxInputSequenceLength, int* hInputData, int* hActualInputSequenceLengths) override;

    void reset() override;

    std::string getInfo() override;

private:
    int gMaxSamplesToRead;
    DataReader::ptr gOriginalDataReader;
    int gCurrentPosition;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_LIMITED_SAMPLES_DATA_READER_
