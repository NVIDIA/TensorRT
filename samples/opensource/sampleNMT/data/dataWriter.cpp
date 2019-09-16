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

#include <sstream>

#include "dataWriter.h"

namespace nmtSample
{
std::string DataWriter::generateText(int sequenceLength, const int* currentOutputData, Vocabulary::ptr vocabulary)
{
    // if clean and handle BPE outputs is required
    std::string delimiter = "@@";
    size_t delimiterSize = delimiter.size();
    std::stringstream sentence;
    std::string word("");
    const char* wordDelimiter = "";
    for (int i = 0; i < sequenceLength; ++i)
    {
        int id = currentOutputData[i];
        if (id != vocabulary->getEndSequenceId())
        {
            std::string token = vocabulary->getToken(id);
            if ((token.size() >= delimiterSize)
                && (token.compare(token.size() - delimiterSize, delimiterSize, delimiter) == 0))
            {
                word = word + token.erase(token.size() - delimiterSize, delimiterSize);
            }
            else
            {
                word = word + token;
                sentence << wordDelimiter;
                sentence << word;
                word = "";
                wordDelimiter = " ";
            }
        }
    }
    return sentence.str();
}
} // namespace nmtSample