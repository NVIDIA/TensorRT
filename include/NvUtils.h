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

#ifndef NV_UTILS_H
#define NV_UTILS_H

#include "NvInfer.h"

//!
//! \file NvUtils.h
//!
//! This file includes various utility functions
//!

namespace nvinfer1
{
namespace utils
{

//!
//! \param input The input weights to reshape.
//! \param shape The shape of the weights.
//! \param shapeOrder The order of the dimensions to process for the output.
//! \param data The location where the output data is placed.
//! \param nbDims The number of dimensions to process.
//!
//! \brief Reformat the input weights of the given shape based on the new
//! order of dimensions.
//!
//! Take the weights specified by \p input with the dimensions specified by
//! \p shape and re-order the weights based on the new dimensions specified
//! by \p shapeOrder. The size of each dimension and the input data is not
//! modified. The output volume pointed to by \p data must be the same as
//! he \p input volume.
//!
//! Example usage:
//! float *out = new float[N*C*H*W];
//! Weights input{DataType::kFLOAT, {0 ... N*C*H*W-1}, N*C*H*W size};
//! int32_t order[4]{1, 0, 3, 2};
//! int32_t shape[4]{C, N, W, H};
//! reshapeWeights(input, shape, order, out, 4);
//! Weights reshaped{input.type, out, input.count};
//!
//! Input Matrix{3, 2, 3, 2}:
//! { 0  1}, { 2  3}, { 4  5} <-- {0, 0, *, *}
//! { 6  7}, { 8  9}, {10 11} <-- {0, 1, *, *}
//! {12 13}, {14 15}, {16 17} <-- {1, 0, *, *}
//! {18 19}, {20 21}, {22 23} <-- {1, 1, *, *}
//! {24 25}, {26 27}, {28 29} <-- {2, 0, *, *}
//! {30 31}, {32 33}, {34 35} <-- {2, 1, *, *}
//!
//! Output Matrix{2, 3, 2, 3}:
//! { 0  2  4}, { 1  3  5} <-- {0, 0, *, *}
//! {12 14 16}, {13 15 17} <-- {0, 1, *, *}
//! {24 26 28}, {25 27 29} <-- {0, 2, *, *}
//! { 6  8 10}, { 7  9 11} <-- {1, 0, *, *}
//! {18 20 22}, {19 21 23} <-- {1, 1, *, *}
//! {30 32 34}, {31 33 35} <-- {1, 2, *, *}
//!
//! \return True on success, false on failure.
//!
//! \deprecated Deprecated in TensorRT 8.0.
//!
//! \warning This file will be removed in TensorRT 10.0.
//!
TRT_DEPRECATED TENSORRTAPI bool reshapeWeights(
    Weights const& input, int32_t const* shape, int32_t const* shapeOrder, void* data, int32_t nbDims) noexcept;

//!
//! \param input The input data to re-order.
//! \param order The new order of the data sub-buffers.
//! \param num The number of data sub-buffers to re-order.
//! \param size The size of each data sub-buffer in bytes.
//!
//! \brief Takes an input stream and re-orders \p num chunks of the data
//! given the \p size and \p order.
//!
//! In some frameworks, the ordering of the sub-buffers within a dimension
//! is different than the way that TensorRT expects them.
//! TensorRT expects the gate/bias sub-buffers for LSTM's to be in fico order.
//! TensorFlow however formats the sub-buffers in icfo order.
//! This helper function solves this in a generic fashion.
//!
//! Example usage output of reshapeWeights above:
//! int32_t indir[1]{1, 0}
//! int32_t stride = W*H;
//! for (int32_t x = 0, y = N*C; x < y; ++x)
//! reorderSubBuffers(out + x * stride, indir, H, W);
//!
//! Input Matrix{2, 3, 2, 3}:
//! { 0  2  4}, { 1  3  5} <-- {0, 0, *, *}
//! {12 14 16}, {13 15 17} <-- {0, 1, *, *}
//! {24 26 28}, {25 27 29} <-- {0, 2, *, *}
//! { 6  8 10}, { 7  9 11} <-- {1, 0, *, *}
//! {18 20 22}, {19 21 23} <-- {1, 1, *, *}
//! {30 32 34}, {31 33 35} <-- {1, 2, *, *}
//!
//! Output Matrix{2, 3, 2, 3}:
//! { 1  3  5}, { 0  2  4} <-- {0, 0, *, *}
//! {13 15 17}, {12 14 16} <-- {0, 1, *, *}
//! {25 27 29}, {24 26 28} <-- {0, 2, *, *}
//! { 7  9 11}, { 6  8 10} <-- {1, 0, *, *}
//! {19 21 23}, {18 20 22} <-- {1, 1, *, *}
//! {31 33 35}, {30 32 34} <-- {1, 2, *, *}
//!
//! \return True on success, false on failure.
//!
//! \see reshapeWeights()
//!
//! \deprecated Deprecated in TensorRT 8.0.
//!
//! \warning This file will be removed in TensorRT 10.0.
//!
TRT_DEPRECATED TENSORRTAPI bool reorderSubBuffers(
    void* input, int32_t const* order, int32_t num, int32_t size) noexcept;

//!
//! \param input The input data to transpose.
//! \param type The type of the data to transpose.
//! \param num The number of data sub-buffers to transpose.
//! \param height The size of the height dimension to transpose.
//! \param width The size of the width dimension to transpose.
//!
//! \brief Transpose \p num sub-buffers of \p height * \p width.
//!
//! \return True on success, false on failure.
//!
//! \deprecated Deprecated in TensorRT 8.0.
//!
//! \warning This file will be removed in TensorRT 10.0.
//!
TRT_DEPRECATED TENSORRTAPI bool transposeSubBuffers(
    void* input, DataType type, int32_t num, int32_t height, int32_t width) noexcept;

} // namespace utils
} // namespace nvinfer1
#endif // NV_UTILS_H
