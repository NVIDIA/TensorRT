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

#include "textWriter.h"

#include <iostream>
#include <sstream>

namespace nmtSample
{
TextWriter::TextWriter(std::shared_ptr<std::ostream> textOnput, Vocabulary::ptr vocabulary)
    : mOutput(textOnput)
    , mVocabulary(vocabulary)
{
}

void TextWriter::write(const int* hOutputData, int actualOutputSequenceLength, int actualInputSequenceLength)
{
    // if clean and handle BPE outputs is required
    *mOutput << DataWriter::generateText(actualOutputSequenceLength, hOutputData, mVocabulary) << "\n";
}

void TextWriter::initialize() {}

void TextWriter::finalize() {}

std::string TextWriter::getInfo()
{
    std::stringstream ss;
    ss << "Text Writer, vocabulary size = " << mVocabulary->getSize();
    return ss.str();
}
} // namespace nmtSample
