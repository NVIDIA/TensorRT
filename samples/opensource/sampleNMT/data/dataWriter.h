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
#ifndef SAMPLE_NMT_DATA_WRITER_
#define SAMPLE_NMT_DATA_WRITER_

#include <memory>
#include <string>

#include "../component.h"
#include "vocabulary.h"

namespace nmtSample
{
/** \class DataWriter
 *
 * \brief writer of sequences of data
 *
 */
class DataWriter : public Component
{
public:
    typedef std::shared_ptr<DataWriter> ptr;

    DataWriter() = default;

    /**
     * \brief write the generated sequence
     */
    virtual void write(const int* hOutputData, int actualOutputSequenceLength, int actualInputSequenceLength) = 0;

    /**
     * \brief it is called right before inference starts
     */
    virtual void initialize() = 0;

    /**
     * \brief it is called right after inference ends
     */
    virtual void finalize() = 0;

    ~DataWriter() override = default;

protected:
    static std::string generateText(int sequenceLength, const int* currentOutputData, Vocabulary::ptr vocabulary);
};
} // namespace nmtSample

#endif // SAMPLE_NMT_DATA_WRITER_
