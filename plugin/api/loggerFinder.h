/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_PLUGIN_API_LOGGER_FINDER_H
#define TRT_PLUGIN_API_LOGGER_FINDER_H

#include "plugin/common/vfcCommon.h"

namespace nvinfer1
{

namespace plugin
{
class VCPluginLoggerFinder : public ILoggerFinder
{
public:
    ILogger* findLogger() override
    {
        return getLogger();
    }
};

VCPluginLoggerFinder gVCPluginLoggerFinder;

//!
//! \brief Set a Logger finder for Version Compatibility (VC) plugin library so that all VC plugins can
//! use getLogger without dependency on nvinfer. This function shall be called once for the loaded vc plugin
//! library.
//!
//! \param setLoggerFinderFunc function in VC plugin library for setting logger finder.
//!
void setVCPluginLoggerFinder(std::function<void(ILoggerFinder*)> setLoggerFinderFunc)
{
    setLoggerFinderFunc(static_cast<ILoggerFinder*>(&gVCPluginLoggerFinder));
}

} // namespace plugin

} // namespace nvinfer1

#endif // TRT_RUNTIME_RT_LOGGER_FINDER_H
