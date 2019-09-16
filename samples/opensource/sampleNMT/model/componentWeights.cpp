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
#include "componentWeights.h"
#include <cassert>
#include <string>

namespace nmtSample
{
std::istream& operator>>(std::istream& input, ComponentWeights& value)
{
    std::string footerString("trtsamplenmt");
    size_t footerSize = sizeof(int32_t) + footerString.size();
    char* footer = (char*) malloc(footerSize);

    input.seekg(0, std::ios::end);
    size_t fileSize = input.tellg();

    input.seekg(-footerSize, std::ios::end);
    input.read(footer, footerSize);

    size_t metaDataCount = ((int32_t*) footer)[0];
    std::string str(footer + sizeof(int32_t), footer + footerSize);
    assert(footerString.compare(str) == 0);
    free(footer);

    input.seekg(-(footerSize + metaDataCount * sizeof(int32_t)), std::ios::end);
    value.mMetaData.resize(metaDataCount);
    size_t metaSize = metaDataCount * sizeof(int32_t);
    input.read((char*) (&value.mMetaData[0]), metaSize);

    size_t dataSize = fileSize - footerSize - metaSize;
    input.seekg(0, input.beg);
    value.mWeights.resize(dataSize);
    input.read(&value.mWeights[0], dataSize);

    return input;
}
} // namespace nmtSample
