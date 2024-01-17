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

//!
//! \file NvInferVersion.h
//!
//! Defines the TensorRT version
//!
#ifndef NV_INFER_VERSION_H
#define NV_INFER_VERSION_H

#define NV_TENSORRT_MAJOR 8 //!< TensorRT major version.
#define NV_TENSORRT_MINOR 6 //!< TensorRT minor version.
#define NV_TENSORRT_PATCH 1 //!< TensorRT patch version.
#define NV_TENSORRT_BUILD 5 //!< TensorRT build number.

#define NV_TENSORRT_LWS_MAJOR 0 //!< TensorRT LWS major version.
#define NV_TENSORRT_LWS_MINOR 0 //!< TensorRT LWS minor version.
#define NV_TENSORRT_LWS_PATCH 0 //!< TensorRT LWS patch version.

// This #define is deprecated in TensorRT 8.6 and will be removed in 10.0. Use NV_TENSORRT_MAJOR.
#define NV_TENSORRT_SONAME_MAJOR 8 //!< Shared object library major version number.
// This #define is deprecated in TensorRT 8.6 and will be removed in 10.0. Use NV_TENSORRT_MINOR.
#define NV_TENSORRT_SONAME_MINOR 6 //!< Shared object library minor version number.
// This #define is deprecated in TensorRT 8.6 and will be removed in 10.0. Use NV_TENSORRT_PATCH.
#define NV_TENSORRT_SONAME_PATCH 1 //!< Shared object library patch version number.

#define NV_TENSORRT_RELEASE_TYPE_EARLY_ACCESS 0         //!< An early access release
#define NV_TENSORRT_RELEASE_TYPE_RELEASE_CANDIDATE 1    //!< A release candidate
#define NV_TENSORRT_RELEASE_TYPE_GENERAL_AVAILABILITY 2 //!< A final release

#define NV_TENSORRT_RELEASE_TYPE NV_TENSORRT_RELEASE_TYPE_GENERAL_AVAILABILITY //!< TensorRT release type

#endif // NV_INFER_VERSION_H
