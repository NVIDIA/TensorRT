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

#ifndef SAMPLE_NMT_COMPONENT_
#define SAMPLE_NMT_COMPONENT_

#include <memory>
#include <string>

namespace nmtSample
{
/** \class Component
 *
 * \brief a functional part of the sample
 *
 */
class Component
{
public:
    typedef std::shared_ptr<Component> ptr;

    /**
     * \brief get the textual description of the component
     */
    virtual std::string getInfo() = 0;

protected:
    Component() = default;

    virtual ~Component() = default;
};
} // namespace nmtSample

#endif // SAMPLE_NMT_COMPONENT_
