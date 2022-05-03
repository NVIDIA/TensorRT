/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
