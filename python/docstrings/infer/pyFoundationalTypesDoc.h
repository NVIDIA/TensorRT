/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This file contains all Dims docstrings, since these are typically too long to keep in the binding code.
#pragma once

namespace tensorrt
{
namespace DataTypeDoc
{
constexpr const char* descr = R"trtdoc(
    Represents data types.

    :ivar itemsize: :class:`int` The size in bytes of this :class:`DataType` .
)trtdoc";

constexpr char const* float32 = R"trtdoc(32-bit floating point format.)trtdoc";
constexpr char const* float16 = R"trtdoc(IEEE 16-bit floating-point format.)trtdoc";
constexpr char const* bfloat16 = R"trtdoc(Brain float -- has an 8 bit exponent and 8 bit significand)trtdoc";
constexpr char const* int8 = R"trtdoc(Signed 8-bit integer representing a quantized floating-point value.)trtdoc";
constexpr char const* int32 = R"trtdoc(Signed 32-bit integer format.)trtdoc";
constexpr char const* int64 = R"trtdoc(Signed 64-bit integer format.)trtdoc";
constexpr char const* boolean = R"trtdoc(8-bit boolean. 0 = false, 1 = true, other values undefined.)trtdoc";
constexpr char const* uint8 = R"trtdoc(
    Unsigned 8-bit integer format.
    Cannot be used to represent quantized floating-point values.
    Use the IdentityLayer to convert ``uint8`` network-level inputs to {``float32``, ``float16``} prior
    to use with other TensorRT layers, or to convert intermediate output
    before ``uint8`` network-level outputs from {``float32``, ``float16``} to ``uint8``.
    ``uint8`` conversions are only supported for {``float32``, ``float16``}.
    ``uint8`` to {``float32``, ``float16``} conversion will convert the integer values
    to equivalent floating point values.
    {``float32``, ``float16``} to ``uint8`` conversion will convert the floating point values
    to integer values by truncating towards zero. This conversion has undefined behavior for
    floating point values outside the range [0.0f, 256.0) after truncation.
    ``uint8`` conversions are not supported for {``int8``, ``int32``, ``bool``}.
)trtdoc";
constexpr char const* fp8 = R"trtdoc(
    Signed 8-bit floating point with 1 sign bit, 4 exponent bits, 3 mantissa
    bits, and exponent-bias 7.
)trtdoc";
constexpr char const* int4 = R"trtdoc(Signed 4-bit integer representing a quantized floating-point value.)trtdoc";
constexpr char const* fp4
    = R"trtdoc(Signed 4-bit floating point with 1 sign bit, 2 exponent bits and 1 mantissa bits.)trtdoc";
constexpr char const* e8m0 = R"trtdoc(Unsigned 8-bit exponent-only floating point.)trtdoc";

} // namespace DataTypeDoc

namespace WeightsRoleDoc
{
constexpr const char* descr
    = R"trtdoc(How a layer uses particular Weights. The power weights of an IScaleLayer are omitted.  Refitting those is not supported.)trtdoc";
constexpr const char* KERNEL = R"trtdoc(Kernel for :class:`IConvolutionLayer` or :class:`IDeconvolutionLayer` .)trtdoc";
constexpr const char* BIAS = R"trtdoc(Bias for :class:`IConvolutionLayer` or :class:`IDeconvolutionLayer` .)trtdoc";
constexpr const char* SHIFT = R"trtdoc(Shift part of :class:`IScaleLayer` .)trtdoc";
constexpr const char* SCALE = R"trtdoc(Scale part of :class:`IScaleLayer` .)trtdoc";
constexpr const char* CONSTANT = R"trtdoc(Weights for :class:`IConstantLayer` .)trtdoc";
constexpr const char* ANY = R"trtdoc(Any other weights role.)trtdoc";

} // namespace WeightsRoleDoc

namespace WeightsDoc
{
constexpr const char* descr = R"trtdoc(
    An array of weights used as a layer parameter.
    The weights are held by reference until the engine has been built - deep copies are not made automatically.

    :ivar dtype: :class:`DataType` The type of the weights.
    :ivar size: :class:`int` The number of weights in the array.
    :ivar nbytes: :class:`int` Total bytes consumed by the elements of the weights buffer.
)trtdoc";

// FIXME: Weird bug occurring here. Cannot provide :arg:
constexpr const char* init_type = R"trtdoc(
    Initializes an empty (0-length) Weights object with the specified type.

    :type: A type to initialize the weights with. Default: :class:`tensorrt.float32`
)trtdoc";

constexpr const char* init_ptr = R"trtdoc(
    Initializes a Weights object with the specified data.

    :type: A type to initialize the weights with.
    :ptr: A pointer to the data.
    :count: The number of weights.
)trtdoc";

// FIXME: Weird bug occurring here. Cannot provide :arg:
constexpr const char* init_numpy = R"trtdoc(
    :a: A numpy array whose values to use. No deep copies are made.
)trtdoc";

constexpr const char* numpy = R"trtdoc(
    Create a numpy array using the underlying buffer of this weights object.
    The resulting array is just a view over the existing data, i.e. no deep copy is made.

    If the weights cannot be converted to NumPy (e.g. due to unsupported data type), the original weights are returned.

    :returns: The NumPy array or the original weights.
)trtdoc";
} // namespace WeightsDoc

namespace DimsDoc
{
constexpr const char* descr = R"trtdoc(
    Structure to define the dimensions of a tensor. :class:`Dims` and all derived classes behave like Python :class:`tuple` s. Furthermore, the TensorRT API can implicitly convert Python iterables to :class:`Dims` objects, so :class:`tuple` or :class:`list` can be used in place of this class.
)trtdoc";

constexpr const char* volume = R"trtdoc(
    Computes the total volume of the dimensions

    :returns: Total volume. `0` for empty dimensions.
)trtdoc";

constexpr const char* get_type = R"trtdoc(
    Queries the type of a dimension.

    :returns: The type of the specified dimension.
)trtdoc";

constexpr const char* MAX_DIMS = R"trtdoc(
    The maximum number of dimensions supported by :class:`Dims`.
)trtdoc";

} // namespace DimsDoc

namespace Dims2Doc
{
constexpr const char* descr = R"trtdoc(
    Structure to define 2D shape.
)trtdoc";
} // namespace Dims2Doc

namespace DimsHWDoc
{
constexpr const char* descr = R"trtdoc(
    Structure to define 2D shape with height and width.

    :ivar h: :class:`int` The first dimension (height).
    :ivar w: :class:`int` The second dimension (width).
)trtdoc";
} // namespace DimsHWDoc

namespace Dims3Doc
{
constexpr const char* descr = R"trtdoc(
    Structure to define 3D shape.
)trtdoc";
} // namespace Dims3Doc

namespace DimsCHWDoc
{
constexpr const char* descr = R"trtdoc(
    Structure to define 3D tensor with a channel dimension, height, and width.

    :ivar c: :class:`int` The first dimension (channel).
    :ivar h: :class:`int` The second dimension (height).
    :ivar w: :class:`int` The third dimension (width).
)trtdoc";
} // namespace DimsCHWDoc

namespace Dims4Doc
{
constexpr const char* descr = R"trtdoc(
    Structure to define 4D tensor.
)trtdoc";
} // namespace Dims4Doc

namespace IVersionedInterfaceDoc
{
constexpr const char* descr = R"trtdoc(
    Base class for all versioned interfaces.
)trtdoc";
} // namespace IVersionedInterfaceDoc

namespace APILanguageDoc
{
constexpr const char* descr = R"trtdoc(
    The language used in the implementation of a TensorRT interface.
)trtdoc";
} // namespace APILanguageDoc

namespace InterfaceInfoDoc
{
constexpr const char* descr = R"trtdoc(
    Version information for a TensorRT interface.
)trtdoc";
} // namespace InterfaceInfoDoc

namespace DimsNCHWDoc
{
constexpr const char* descr = R"trtdoc(
    Structure to define 4D tensor with a batch dimension, a channel dimension, height and width.

    :ivar n: :class:`int` The first dimension (batch).
    :ivar c: :class:`int` The second dimension (channel).
    :ivar h: :class:`int` The third dimension (height).
    :ivar w: :class:`int` The fourth dimension (width).
)trtdoc";
} // namespace DimsNCHWDoc

namespace IHostMemoryDoc
{
constexpr const char* descr = R"trtdoc(
    Handles library allocated memory that is accessible to the user.

    The memory allocated via the host memory object is owned by the library and will be de-allocated when object is destroyed.

    This class exposes a buffer interface using Python's buffer protocol.

    :ivar dtype: :class:`DataType` The data type of this buffer.
    :ivar nbytes: :class:`int` Total bytes consumed by the elements of the buffer.
)trtdoc";
} // namespace IHostMemoryDoc

} // namespace tensorrt
