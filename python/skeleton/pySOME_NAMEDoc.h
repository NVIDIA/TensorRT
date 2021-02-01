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

// This file contains all docstrings for <SOME_NAME>.
#pragma once

namespace tensorrt
{
    namespace <SOME_CLASS>Doc
    {
        constexpr const char* descr = R"trtdoc(
            Description of the class/enum

            :ivar property1: :class:`TYPE` DESCRIPTION
        )trtdoc";

        constexpr const char* func1 = R"trtdoc(
            Description of some function (attr for enum) of the class/enum

            :param param1: DESCRIPTION

            :returns: DESCRIPTION
        )trtdoc";
    } /* DimsDoc */

} /* tensorrt */
