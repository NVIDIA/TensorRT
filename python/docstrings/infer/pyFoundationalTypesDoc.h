/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

// This file contains all Dims docstrings, since these are typically too long to keep in the binding code.
#pragma once

namespace tensorrt
{
namespace DataTypeDoc
{
constexpr const char* descr = R"trtdoc(
    Represents data types.

    :itemsize: :class:`int` The size in bytes of this :class:`DataType` .
)trtdoc";

constexpr const char* float32 = R"trtdoc(Represents a 32-bit floating point number.)trtdoc";
constexpr const char* float16 = R"trtdoc(Represents a 16-bit floating point number.)trtdoc";
constexpr const char* int8 = R"trtdoc(Represents an 8-bit integer.)trtdoc";
constexpr const char* int32 = R"trtdoc(Represents a 32-bit integer.)trtdoc";
constexpr const char* boolean = R"trtdoc(Represents a boolean.)trtdoc";

} // namespace DataTypeDoc

namespace WeightsRoleDoc
{
constexpr const char* descr
    = R"trtdoc(How a layer uses particular Weights. The power weights of an IScaleLayer are omitted.  Refitting those is not supported.)trtdoc";
constexpr const char* KERNEL
    = R"trtdoc(Kernel for :class:`IConvolutionLayer` , :class:`IDeconvolutionLayer` , or :class:`IFullyConnectedLayer` .)trtdoc";
constexpr const char* BIAS
    = R"trtdoc(Bias for :class:`IConvolutionLayer` , :class:`IDeconvolutionLayer` , or :class:`IFullyConnectedLayer` .)trtdoc";
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

// FIXME: Weird bug occurring here. Cannot provide :arg:
constexpr const char* init_numpy = R"trtdoc(
    :a: A numpy array whose values to use. No deep copies are made.
)trtdoc";

constexpr const char* numpy = R"trtdoc(
    Create a numpy array using the underlying buffer of this weights object.

    :returns: A new numpy array that holds a reference to this weight object's buffer - no deep copy is made.
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
