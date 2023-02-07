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

#ifndef NV_INFER_LEGACY_DIMS_H
#define NV_INFER_LEGACY_DIMS_H

#include "NvInferRuntimeCommon.h"

//!
//! \file NvInferLegacyDims.h
//!
//! This file contains declarations of legacy dimensions types which use channel
//! semantics in their names, and declarations on which those types rely.
//!

//!
//! \namespace nvinfer1
//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{
//!
//! \class Dims2
//! \brief Descriptor for two-dimensional data.
//!
class Dims2 : public Dims
{
public:
    //!
    //! \brief Construct an empty Dims2 object.
    //!
    Dims2()
        : Dims2(0, 0)
    {
    }

    //!
    //! \brief Construct a Dims2 from 2 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //!
    Dims2(int32_t d0, int32_t d1)
    {
        nbDims = 2;
        d[0] = d0;
        d[1] = d1;
        for (int32_t i{nbDims}; i < Dims::MAX_DIMS; ++i)
        {
            d[i] = 0;
        }
    }
};

//!
//! \class DimsHW
//! \brief Descriptor for two-dimensional spatial data.
//!
class DimsHW : public Dims2
{
public:
    //!
    //! \brief Construct an empty DimsHW object.
    //!
    DimsHW()
        : Dims2()
    {
    }

    //!
    //! \brief Construct a DimsHW given height and width.
    //!
    //! \param height the height of the data
    //! \param width the width of the data
    //!
    DimsHW(int32_t height, int32_t width)
        : Dims2(height, width)
    {
    }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int32_t& h()
    {
        return d[0];
    }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int32_t h() const
    {
        return d[0];
    }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int32_t& w()
    {
        return d[1];
    }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int32_t w() const
    {
        return d[1];
    }
};

//!
//! \class Dims3
//! \brief Descriptor for three-dimensional data.
//!
class Dims3 : public Dims2
{
public:
    //!
    //! \brief Construct an empty Dims3 object.
    //!
    Dims3()
        : Dims3(0, 0, 0)
    {
    }

    //!
    //! \brief Construct a Dims3 from 3 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //! \param d2 The third element.
    //!
    Dims3(int32_t d0, int32_t d1, int32_t d2)
        : Dims2(d0, d1)
    {
        nbDims = 3;
        d[2] = d2;
    }
};

//!
//! \class Dims4
//! \brief Descriptor for four-dimensional data.
//!
class Dims4 : public Dims3
{
public:
    //!
    //! \brief Construct an empty Dims4 object.
    //!
    Dims4()
        : Dims4(0, 0, 0, 0)
    {
    }

    //!
    //! \brief Construct a Dims4 from 4 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //! \param d2 The third element.
    //! \param d3 The fourth element.
    //!
    Dims4(int32_t d0, int32_t d1, int32_t d2, int32_t d3)
        : Dims3(d0, d1, d2)
    {
        nbDims = 4;
        d[3] = d3;
    }
};

} // namespace nvinfer1

#endif // NV_INFER_LEGCY_DIMS_H
