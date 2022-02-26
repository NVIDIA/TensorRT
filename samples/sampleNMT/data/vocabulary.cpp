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

#include "common.h"
#include "vocabulary.h"
#include <clocale>
#include <iostream>
#include <istream>

namespace nmtSample
{
const std::string Vocabulary::mSosStr = "<s>";
const std::string Vocabulary::mEosStr = "</s>";
const std::string Vocabulary::mUnkStr = "<unk>";

Vocabulary::Vocabulary()
    : mNumTokens(0)
    , mSosId(0)
    , mEosId(0)
    , mUnkId(0)
{
}

void Vocabulary::add(const std::string& token)
{
    ASSERT(mTokenToId.find(token) == mTokenToId.end());
    mTokenToId[token] = mNumTokens;
    mIdToToken.push_back(token);
    mNumTokens++;
}

int32_t Vocabulary::getId(const std::string& token) const
{
    auto it = mTokenToId.find(token);
    if (it != mTokenToId.end())
        return it->second;
    return mUnkId;
}

std::string Vocabulary::getToken(int32_t id) const
{
    ASSERT(id < mNumTokens);
    return mIdToToken[id];
}

int32_t Vocabulary::getSize() const
{
    return mNumTokens;
}

// cppcheck-suppress unusedFunction
std::istream& operator>>(std::istream& input, Vocabulary& value)
{
    // stream should contain "<s>", "</s>" and "<unk>" tokens
    std::setlocale(LC_ALL, "en_US.UTF-8");
    std::string word;
    while (input >> word)
    {
        value.add(word);
    }

    {
        auto it = value.mTokenToId.find(Vocabulary::mSosStr);
        ASSERT(it != value.mTokenToId.end());
        value.mSosId = it->second;
    }

    {
        auto it = value.mTokenToId.find(Vocabulary::mEosStr);
        ASSERT(it != value.mTokenToId.end());
        value.mEosId = it->second;
    }

    {
        auto it = value.mTokenToId.find(Vocabulary::mUnkStr);
        ASSERT(it != value.mTokenToId.end());
        value.mUnkId = it->second;
    }

    return input;
}

int32_t Vocabulary::getStartSequenceId()
{
    return mSosId;
}

int32_t Vocabulary::getEndSequenceId()
{
    return mEosId;
}
} // namespace nmtSample
