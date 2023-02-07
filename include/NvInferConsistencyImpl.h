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

#ifndef NV_INFER_CONSISTENCY_IMPL_H
#define NV_INFER_CONSISTENCY_IMPL_H

namespace nvinfer1
{

//!
//! \file NvInferConsistencyImpl.h
//!
//! This file contains definitions for API methods that cross the shared library boundary. These
//! methods must not be called directly by applications; they should only be called through the
//! API classes.
//!

namespace apiv
{

class VConsistencyChecker
{
public:
    virtual ~VConsistencyChecker() noexcept = default;
    virtual bool validate() const noexcept = 0;
};

} // namespace apiv
} // namespace nvinfer1

#endif // NV_INFER_CONSISTENCY_IMPL_H
