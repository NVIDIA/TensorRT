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

#ifndef SAMPLE_NMT_VOCABULARY_
#define SAMPLE_NMT_VOCABULARY_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "sequenceProperties.h"

namespace nmtSample
{
/** \class Vocabulary
 *
 * \brief String<->Id bijection storage
 *
 */
class Vocabulary : public SequenceProperties
{
public:
    typedef std::shared_ptr<Vocabulary> ptr;

    Vocabulary();

    friend std::istream& operator>>(std::istream& input, Vocabulary& value);

    /**
     * \brief add new token to vocabulary, ID is auto-generated
     */
    void add(const std::string& token);

    /**
     * \brief get the ID of the token
     */
    int32_t getId(const std::string& token) const;

    /**
     * \brief get token by ID
     */
    std::string getToken(int32_t id) const;

    /**
     * \brief get the number of elements in the vocabulary
     */
    int32_t getSize() const;

    int32_t getStartSequenceId() override;

    int32_t getEndSequenceId() override;

private:
    static const std::string mSosStr;
    static const std::string mUnkStr;
    static const std::string mEosStr;

    std::map<std::string, int32_t> mTokenToId;
    std::vector<std::string> mIdToToken;
    int32_t mNumTokens;

    int32_t mSosId;
    int32_t mEosId;
    int32_t mUnkId;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_VOCABULARY_
