/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

//!
//! \file NvInferVersion.h
//!
//! Defines the TensorRT version
//!
#ifndef NV_INFER_VERSION_H
#define NV_INFER_VERSION_H

#define NV_TENSORRT_MAJOR 9 //!< TensorRT major version.
#define NV_TENSORRT_MINOR 2 //!< TensorRT minor version.
#define NV_TENSORRT_PATCH 0 //!< TensorRT patch version.
#define NV_TENSORRT_BUILD 5 //!< TensorRT build number.

#define NV_TENSORRT_LWS_MAJOR 0 //!< TensorRT LWS major version.
#define NV_TENSORRT_LWS_MINOR 0 //!< TensorRT LWS minor version.
#define NV_TENSORRT_LWS_PATCH 0 //!< TensorRT LWS patch version.

// This #define is deprecated in TensorRT 8.6. Use NV_TENSORRT_MAJOR.
#define NV_TENSORRT_SONAME_MAJOR 9 //!< Shared object library major version number.
// This #define is deprecated in TensorRT 8.6. Use NV_TENSORRT_MINOR.
#define NV_TENSORRT_SONAME_MINOR 2 //!< Shared object library minor version number.
// This #define is deprecated in TensorRT 8.6. Use NV_TENSORRT_PATCH.
#define NV_TENSORRT_SONAME_PATCH 0 //!< Shared object library patch version number.

#define NV_TENSORRT_RELEASE_TYPE_EARLY_ACCESS 0         //!< An early access release
#define NV_TENSORRT_RELEASE_TYPE_RELEASE_CANDIDATE 1    //!< A release candidate
#define NV_TENSORRT_RELEASE_TYPE_GENERAL_AVAILABILITY 2 //!< A final release

#define NV_TENSORRT_RELEASE_TYPE NV_TENSORRT_RELEASE_TYPE_GENERAL_AVAILABILITY //!< TensorRT release type

#endif // NV_INFER_VERSION_H
