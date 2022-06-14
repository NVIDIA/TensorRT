/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
