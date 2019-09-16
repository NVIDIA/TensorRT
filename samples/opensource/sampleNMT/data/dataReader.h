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

#ifndef SAMPLE_NMT_DATA_READER_
#define SAMPLE_NMT_DATA_READER_

#include <memory>

#include "../component.h"

namespace nmtSample
{
/** \class DataReader
 *
 * \brief reader of sequences of data
 *
 */
class DataReader : public Component
{
public:
    typedef std::shared_ptr<DataReader> ptr;

    DataReader() = default;

    /**
     * \brief reads the batch of smaples/sequences
     *
     * \return the actual number of samples read
     */
    virtual int read(int samplesToRead, int maxInputSequenceLength, int* hInputData, int* hActualInputSequenceLengths)
        = 0;

    /**
     * \brief Reset the reader position, the data reader is ready to read the data from th ebeginning again after the
     * function returns
     */
    virtual void reset() = 0;

    ~DataReader() override = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_DATA_READER_
