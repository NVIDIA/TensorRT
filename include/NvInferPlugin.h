/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NV_INFER_PLUGIN_H
#define NV_INFER_PLUGIN_H

#include "NvInfer.h"
#include "NvInferPluginUtils.h"
//!
//! \file NvInferPlugin.h
//!
//! This is the API for the Nvidia provided TensorRT plugins.
//!

extern "C"
{
    //!
    //! \brief Initialize and register all the existing TensorRT plugins to the Plugin Registry with an optional
    //! namespace. The plugin library author should ensure that this function name is unique to the library. This
    //! function should be called once before accessing the Plugin Registry.
    //! \param logger Logger object to print plugin registration information
    //! \param libNamespace Namespace used to register all the plugins in this library
    //!
    TENSORRTAPI bool initLibNvInferPlugins(void* logger, char const* libNamespace);

} // extern "C"

#endif // NV_INFER_PLUGIN_H
