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

#include "vocabulary.h"
#include <assert.h>
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
{
}

void Vocabulary::add(const std::string& token)
{
    assert(mTokenToId.find(token) == mTokenToId.end());
    mTokenToId[token] = mNumTokens;
    mIdToToken.push_back(token);
    mNumTokens++;
}

int Vocabulary::getId(const std::string& token) const
{
    auto it = mTokenToId.find(token);
    if (it != mTokenToId.end())
        return it->second;
    return mUnkId;
}

std::string Vocabulary::getToken(int id) const
{
    assert(id < mNumTokens);
    return mIdToToken[id];
}

int Vocabulary::getSize() const
{
    return mNumTokens;
}

std::istream& operator>>(std::istream& input, Vocabulary& value)
{
    // stream should contain "<s>", "</s>" and "<unk>" tokens
    std::setlocale(LC_ALL, "en_US.UTF-8");
    std::string line;
    std::string word;
    while (input >> word)
    {
        value.add(word);
    }

    {
        auto it = value.mTokenToId.find(Vocabulary::mSosStr);
        assert(it != value.mTokenToId.end());
        value.mSosId = it->second;
    }

    {
        auto it = value.mTokenToId.find(Vocabulary::mEosStr);
        assert(it != value.mTokenToId.end());
        value.mEosId = it->second;
    }

    {
        auto it = value.mTokenToId.find(Vocabulary::mUnkStr);
        assert(it != value.mTokenToId.end());
        value.mUnkId = it->second;
    }

    return input;
}

int Vocabulary::getStartSequenceId()
{
    return mSosId;
}

int Vocabulary::getEndSequenceId()
{
    return mEosId;
}
} // namespace nmtSample