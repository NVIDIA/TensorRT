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

#ifndef SAMPLE_NMT_TEXT_WRITER_
#define SAMPLE_NMT_TEXT_WRITER_

#include <memory>
#include <ostream>

#include "dataWriter.h"
#include "vocabulary.h"

namespace nmtSample
{
/** \class TextReader
 *
 * \brief writes sequences of data into output stream
 *
 */
class TextWriter : public DataWriter
{
public:
    TextWriter(std::shared_ptr<std::ostream> textOnput, Vocabulary::ptr vocabulary);

    void write(const int* hOutputData, int actualOutputSequenceLength, int actualInputSequenceLength) override;

    void initialize() override;

    void finalize() override;

    std::string getInfo() override;

    ~TextWriter() override = default;

private:
    std::shared_ptr<std::ostream> mOutput;
    Vocabulary::ptr mVocabulary;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_TEXT_WRITER_
