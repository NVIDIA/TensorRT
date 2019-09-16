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

#ifndef SAMPLE_NMT_SEQUENCE_PROPERTIES_
#define SAMPLE_NMT_SEQUENCE_PROPERTIES_

#include <memory>

namespace nmtSample
{
/** \class SequenceProperties
 *
 * \brief provides encoder/decoder relevant properties of sequences
 *
 */
class SequenceProperties
{
public:
    typedef std::shared_ptr<SequenceProperties> ptr;

    SequenceProperties() = default;

    virtual int getStartSequenceId() = 0;

    virtual int getEndSequenceId() = 0;

    virtual ~SequenceProperties() = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_SEQUENCE_PROPERTIES_
