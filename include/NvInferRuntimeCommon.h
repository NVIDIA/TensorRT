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

#ifndef NV_INFER_RUNTIME_COMMON_H
#define NV_INFER_RUNTIME_COMMON_H

#include "NvInferVersion.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

// Items that are marked as deprecated will be removed in a future release.
#if __cplusplus >= 201402L
#define TRT_DEPRECATED [[deprecated]]
#if __GNUC__ < 6
#define TRT_DEPRECATED_ENUM
#else
#define TRT_DEPRECATED_ENUM TRT_DEPRECATED
#endif
#ifdef _MSC_VER
#define TRT_DEPRECATED_API __declspec(dllexport)
#else
#define TRT_DEPRECATED_API [[deprecated]] __attribute__((visibility("default")))
#endif
#else
#ifdef _MSC_VER
#define TRT_DEPRECATED
#define TRT_DEPRECATED_ENUM
#define TRT_DEPRECATED_API __declspec(dllexport)
#else
#define TRT_DEPRECATED __attribute__((deprecated))
#define TRT_DEPRECATED_ENUM
#define TRT_DEPRECATED_API __attribute__((deprecated, visibility("default")))
#endif
#endif

// Defines which symbols are exported
#ifdef TENSORRT_BUILD_LIB
#ifdef _MSC_VER
#define TENSORRTAPI __declspec(dllexport)
#else
#define TENSORRTAPI __attribute__((visibility("default")))
#endif
#else
#define TENSORRTAPI
#endif
#define TRTNOEXCEPT
//!
//! \file NvInferRuntimeCommon.h
//!
//! This is the top-level API file for TensorRT core runtime library.
//!

// forward declare some CUDA types to avoid an include dependency

extern "C"
{
    //! Forward declaration of cublasContext to use in other interfaces
    struct cublasContext;
    //! Forward declaration of cudnnContext to use in other interfaces
    struct cudnnContext;
}

#define NV_TENSORRT_VERSION nvinfer1::kNV_TENSORRT_VERSION_IMPL
//!
//! \namespace nvinfer1
//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{

static constexpr int32_t kNV_TENSORRT_VERSION_IMPL
    = (NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + NV_TENSORRT_PATCH; // major, minor, patch

//! char_t is the type used by TensorRT to represent all valid characters.
using char_t = char;
//! AsciiChar is the type used by TensorRT to represent valid ASCII characters.
using AsciiChar = char_t;

//! Forward declare IErrorRecorder for use in other interfaces.
class IErrorRecorder;
//! Forward declare IGpuAllocator for use in other interfaces.
class IGpuAllocator;

namespace impl
{
//! Declaration of EnumMaxImpl struct to store maximum number of elements in an enumeration type.
template <typename T>
struct EnumMaxImpl;
} // namespace impl

//! Maximum number of elements in an enumeration type.
template <typename T>
constexpr int32_t EnumMax() noexcept
{
    return impl::EnumMaxImpl<T>::kVALUE;
}

//!
//! \enum DataType
//! \brief The type of weights and tensors.
//!
enum class DataType : int32_t
{
    //! 32-bit floating point format.
    kFLOAT = 0,

    //! IEEE 16-bit floating-point format.
    kHALF = 1,

    //! 8-bit integer representing a quantized floating-point value.
    kINT8 = 2,

    //! Signed 32-bit integer format.
    kINT32 = 3,

    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    kBOOL = 4
};

namespace impl
{
//! Maximum number of elements in DataType enum. \see DataType
template <>
struct EnumMaxImpl<DataType>
{
    // Declaration of kVALUE that represents maximum number of elements in DataType enum
    static constexpr int32_t kVALUE = 5;
};
} // namespace impl

//!
//! \class Dims
//! \brief Structure to define the dimensions of a tensor.
//!
//! TensorRT can also return an invalid dims structure. This structure is represented by nbDims == -1
//! and d[i] == 0 for all d.
//!
//! TensorRT can also return an "unknown rank" dims structure. This structure is represented by nbDims == -1
//! and d[i] == -1 for all d.
//!
class Dims32
{
public:
    //! The maximum rank (number of dimensions) supported for a tensor.
    static constexpr int32_t MAX_DIMS{8};
    //! The rank (number of dimensions).
    int32_t nbDims;
    //! The extent of each dimension.
    int32_t d[MAX_DIMS];
};

//!
//! Alias for Dims32.
//!
//! \warning: This alias might change in the future.
//!
using Dims = Dims32;

//!
//! \enum TensorFormat
//!
//! \brief Format of the input/output tensors.
//!
//! This enum is used by both plugins and network I/O tensors.
//!
//! \see IPluginV2::supportsFormat(), safe::ICudaEngine::getBindingFormat()
//!
//! For more information about data formats, see the topic "Data Format Description" located in the
//! TensorRT Developer Guide.
//!
enum class TensorFormat : int32_t
{
    //! Row major linear format.
    //! For a tensor with dimensions {N, C, H, W} or {numbers, channels,
    //! columns, rows}, the dimensional index corresponds to {3, 2, 1, 0}
    //! and thus the order is W minor.
    //!
    //! For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
    //!
    kLINEAR = 0,

    //! Two wide channel vectorized row major format. This format is bound to
    //! FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+1)/2][H][W][2], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/2][h][w][c%2].
    kCHW2 = 1,

    //! Eight channel format where C is padded to a multiple of 8. This format
    //! is bound to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to the array with dimensions
    //! [N][H][W][(C+7)/8*8], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][h][w][c].
    kHWC8 = 2,

    //! Four wide channel vectorized row major format. This format is bound to
    //! INT8 or FP16. It is only available for dimensions >= 3.
    //! For INT8, the C dimension must be a build-time constant.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+3)/4][H][W][4], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/4][h][w][c%4].
    //!
    //! Deprecated usage:
    //!
    //! If running on the DLA, this format can be used for acceleration
    //! with the caveat that C must be equal or lesser than 4.
    //! If used as DLA input and the build option kGPU_FALLBACK is not specified,
    //! it needs to meet line stride requirement of DLA format. Column stride in bytes should
    //! be a multiple of 32 on Xavier and 64 on Orin.
    kCHW4 = 3,

    //! Sixteen wide channel vectorized row major format. This format is bound
    //! to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+15)/16][H][W][16], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/16][h][w][c%16].
    //!
    //! For DLA usage, this format maps to the native image format for FP16,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    //!
    kCHW16 = 4,

    //! Thirty-two wide channel vectorized row major format. This format is
    //! only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+31)/32][H][W][32], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/32][h][w][c%32].
    //!
    //! For DLA usage, this format maps to the native image format for INT8,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    kCHW32 = 5,

    //! Eight channel format where C is padded to a multiple of 8. This format
    //! is bound to FP16, and it is only available for dimensions >= 4.
    //! For a tensor with dimensions {N, C, D, H, W},
    //! the memory layout is equivalent to an array with dimensions
    //! [N][D][H][W][(C+7)/8*8], with the tensor coordinates (n, c, d, h, w)
    //! mapping to array subscript [n][d][h][w][c].
    kDHWC8 = 6,

    //! Thirty-two wide channel vectorized row major format. This format is
    //! bound to FP16 and INT8 and is only available for dimensions >= 4.
    //! For a tensor with dimensions {N, C, D, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+31)/32][D][H][W][32], with the tensor coordinates (n, c, d, h, w)
    //! mapping to array subscript [n][c/32][d][h][w][c%32].
    kCDHW32 = 7,

    //! Non-vectorized channel-last format. This format is bound to FP32
    //! and is only available for dimensions >= 3.
    kHWC = 8,

    //! DLA planar format. For a tensor with dimension {N, C, H, W}, the W axis
    //! always has unit stride. The stride for stepping along the H axis is
    //! rounded up to 64 bytes.
    //!
    //! The memory layout is equivalent to a C array with dimensions
    //! [N][C][H][roundUp(W, 64/elementSize)] where elementSize is
    //! 2 for FP16 and 1 for Int8, with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c][h][w].
    kDLA_LINEAR = 9,

    //! DLA image format. For a tensor with dimension {N, C, H, W} the C axis
    //! always has unit stride. The stride for stepping along the H axis is rounded up
    //! to 32 bytes on Xavier and 64 bytes on Orin. C can only be 1, 3 or 4.
    //! If C == 1, it will map to grayscale format.
    //! If C == 3 or C == 4, it will map to color image format. And if C == 3,
    //! the stride for stepping along the W axis needs to be padded to 4 in elements.
    //!
    //! When C is {1, 3, 4}, then C' is {1, 4, 4} respectively,
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][H][roundUp(W, 32/C'/elementSize)][C'] on Xavier and [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin
    //! where elementSize is 2 for FP16
    //! and 1 for Int8. The tensor coordinates (n, c, h, w) mapping to array
    //! subscript [n][h][w][c].
    kDLA_HWC4 = 10,

    //! Sixteen channel format where C is padded to a multiple of 16. This format
    //! is bound to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to the array with dimensions
    //! [N][H][W][(C+15)/16*16], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][h][w][c].
    kHWC16 = 11
};

//!
//! \brief PluginFormat is reserved for backward compatibility.
//!
//! \see IPluginV2::supportsFormat()
//!
using PluginFormat = TensorFormat;

namespace impl
{
//! Maximum number of elements in TensorFormat enum. \see TensorFormat
template <>
struct EnumMaxImpl<TensorFormat>
{
    //! Declaration of kVALUE that represents maximum number of elements in TensorFormat enum
    static constexpr int32_t kVALUE = 12;
};
} // namespace impl

//! \struct PluginTensorDesc
//!
//! \brief Fields that a plugin might see for an input or output.
//!
//! Scale is only valid when data type is DataType::kINT8. TensorRT will set
//! the value to -1.0f if it is invalid.
//!
//! \see IPluginV2IOExt::supportsFormatCombination
//! \see IPluginV2IOExt::configurePlugin
//!
struct PluginTensorDesc
{
    //! Dimensions.
    Dims dims;
    //! \warning DataType:kBOOL not supported.
    DataType type;
    //! Tensor format.
    TensorFormat format;
    //! Scale for INT8 data type.
    float scale;
};

//! \struct PluginVersion
//!
//! \brief Definition of plugin versions.
//!
//! Tag for plug-in versions.  Used in upper byte of getTensorRTVersion().
//!
enum class PluginVersion : uint8_t
{
    //! IPluginV2
    kV2 = 0,
    //! IPluginV2Ext
    kV2_EXT = 1,
    //! IPluginV2IOExt
    kV2_IOEXT = 2,
    //! IPluginV2DynamicExt
    kV2_DYNAMICEXT = 3,
};

//! \class IPluginV2
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. When
//! combined with IPluginCreator it provides a mechanism to register plugins and
//! look up the Plugin Registry during de-serialization.
//!
//! \see IPluginCreator
//! \see IPluginRegistry
//!
class IPluginV2
{
public:
    //!
    //! \brief Return the API version with which this plugin was built.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with
    //! plugins.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, the implementation provided here is safe to call from any thread.
    //!
    virtual int32_t getTensorRTVersion() const noexcept
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief Return the plugin type. Should match the plugin name returned by the corresponding plugin creator
    //! \see IPluginCreator::getPluginName()
    //!
    //! \warning The string returned must be 1024 bytes or less including the NULL terminator and must be NULL
    //! terminated.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual AsciiChar const* getPluginType() const noexcept = 0;

    //!
    //! \brief Return the plugin version. Should match the plugin version returned by the corresponding plugin creator
    //! \see IPluginCreator::getPluginVersion()
    //!
    //! \warning The string returned must be 1024 bytes or less including the NULL terminator and must be NULL
    //! terminated.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    //!
    //! \brief Get the number of outputs from the layer.
    //!
    //! \return The number of outputs.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called
    //! prior to any call to initialize().
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual int32_t getNbOutputs() const noexcept = 0;

    //!
    //! \brief Get the dimension of an output tensor.
    //!
    //! \param index The index of the output tensor.
    //! \param inputs The input tensors.
    //! \param nbInputDims The number of input tensors.
    //!
    //! This function is called by the implementations of INetworkDefinition and IBuilder. In particular, it is called
    //! prior to any call to initialize().
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    //! \note In any non-IPluginV2DynamicExt plugin, batch size should not be included in the returned dimensions,
    //! even if the plugin is expected to be run in a network with explicit batch mode enabled.
    //! Please see the TensorRT Developer Guide for more details on how plugin inputs and outputs behave.
    //!
    virtual Dims getOutputDimensions(int32_t index, Dims const* inputs, int32_t nbInputDims) noexcept = 0;

    //!
    //! \brief Check format support.
    //!
    //! \param type DataType requested.
    //! \param format PluginFormat requested.
    //! \return true if the plugin supports the type-format combination.
    //!
    //! This function is called by the implementations of INetworkDefinition, IBuilder, and
    //! safe::ICudaEngine/ICudaEngine. In particular, it is called when creating an engine and when deserializing an
    //! engine.
    //!
    //! \warning for the format field, the values PluginFormat::kCHW4, PluginFormat::kCHW16, and PluginFormat::kCHW32
    //! will not be passed in, this is to keep backward compatibility with TensorRT 5.x series.  Use PluginV2IOExt
    //! or PluginV2DynamicExt for other PluginFormats.
    //!
    //! \warning DataType:kBOOL not supported.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual bool supportsFormat(DataType type, PluginFormat format) const noexcept = 0;

    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
    //! algorithm choices on the basis of its weights, dimensions, and maximum batch size.
    //!
    //! \param inputDims The input tensor dimensions.
    //! \param nbInputs The number of inputs.
    //! \param outputDims The output tensor dimensions.
    //! \param nbOutputs The number of outputs.
    //! \param type The data type selected for the engine.
    //! \param format The format selected for the engine.
    //! \param maxBatchSize The maximum batch size.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be
    //! 3-dimensional CHW dimensions).
    //!
    //! \warning for the format field, the values PluginFormat::kCHW4, PluginFormat::kCHW16, and PluginFormat::kCHW32
    //! will not be passed in, this is to keep backward compatibility with TensorRT 5.x series.  Use PluginV2IOExt
    //! or PluginV2DynamicExt for other PluginFormats.
    //!
    //! \warning DataType:kBOOL not supported.
    //!
    //! \see clone()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin. However, TensorRT
    //!                  will not call this method from two threads simultaneously on a given clone of a plugin.
    //!
    virtual void configureWithFormat(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType type, PluginFormat format, int32_t maxBatchSize) noexcept
        = 0;

    //!
    //! \brief Initialize the layer for execution. This is called when the engine is created.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when using multiple
    //!                  execution contexts using this plugin.
    //!
    virtual int32_t initialize() noexcept = 0;

    //!
    //! \brief Release resources acquired during plugin layer initialization. This is called when the engine is
    //! destroyed.
    //! \see initialize()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when using multiple
    //!                  execution contexts using this plugin. However, TensorRT will not call this method from
    //!                  two threads simultaneously on a given clone of a plugin.
    //!
    virtual void terminate() noexcept = 0;

    //!
    //! \brief Find the workspace size required by the layer.
    //!
    //! This function is called during engine startup, after initialize(). The workspace size returned should be
    //! sufficient for any batch size up to the maximum.
    //!
    //! \return The workspace size.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin. However, TensorRT
    //!                  will not call this method from two threads simultaneously on a given clone of a plugin.
    //!
    virtual size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept = 0;

    //!
    //! \brief Execute the layer.
    //!
    //! \param batchSize The number of inputs in the batch.
    //! \param inputs The memory for the input tensors.
    //! \param outputs The memory for the output tensors.
    //! \param workspace Workspace for execution.
    //! \param stream The stream in which to execute the kernels.
    //!
    //! \return 0 for success, else non-zero (which will cause engine termination).
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept
        = 0;

    //!
    //! \brief Find the size of the serialization buffer required.
    //!
    //! \return The size of the serialization buffer.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual size_t getSerializationSize() const noexcept = 0;

    //!
    //! \brief Serialize the layer.
    //!
    //! \param buffer A pointer to a buffer to serialize data. Size of buffer must be equal to value returned by
    //! getSerializationSize.
    //!
    //! \see getSerializationSize()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void serialize(void* buffer) const noexcept = 0;

    //!
    //! \brief Destroy the plugin object. This will be called when the network, builder or engine is destroyed.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void destroy() noexcept = 0;

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters and returns a new plugin object with
    //! these parameters.
    //!
    //! The TensorRT runtime calls clone() to clone the plugin when an execution context is created for an engine,
    //! after the engine has been created.  The runtime does not call initialize() on the cloned plugin,
    //! so the cloned plugin should be created in an initialized state.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when creating multiple
    //!                  execution contexts.
    //!
    virtual IPluginV2* clone() const noexcept = 0;

    //!
    //! \brief Set the namespace that this plugin object belongs to. Ideally, all plugin
    //! objects from the same plugin library should have the same namespace.
    //!
    //! \param pluginNamespace The namespace for the plugin object.
    //!
    //! \warning The string pluginNamespace must be 1024 bytes or less including the NULL terminator and must be NULL
    //! terminated.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept = 0;

    //!
    //! \brief Return the namespace of the plugin object.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

    // @cond SuppressDoxyWarnings
    IPluginV2() = default;
    virtual ~IPluginV2() noexcept = default;
// @endcond

protected:
// @cond SuppressDoxyWarnings
    IPluginV2(IPluginV2 const&) = default;
    IPluginV2(IPluginV2&&) = default;
    IPluginV2& operator=(IPluginV2 const&) & = default;
    IPluginV2& operator=(IPluginV2&&) & = default;
// @endcond
};

//! \class IPluginV2Ext
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. This
//! interface provides additional capabilities to the IPluginV2 interface by
//! supporting different output data types and broadcast across batch.
//!
//! \see IPluginV2
//!
class IPluginV2Ext : public IPluginV2
{
public:
    //!
    //! \brief Return the DataType of the plugin output at the requested index.
    //!
    //! The default behavior should be to return the type of the first input, or DataType::kFLOAT if the layer has no
    //! inputs. The returned data type must have a format that is supported by the plugin.
    //!
    //! \see supportsFormat()
    //!
    //! \warning DataType:kBOOL not supported.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
        = 0;

    //! \brief Return true if output tensor is broadcast across a batch.
    //!
    //! \param outputIndex The index of the output
    //! \param inputIsBroadcasted The ith element is true if the tensor for the ith input is broadcast across a batch.
    //! \param nbInputs The number of inputs
    //!
    //! The values in inputIsBroadcasted refer to broadcasting at the semantic level,
    //! i.e. are unaffected by whether method canBroadcastInputAcrossBatch requests
    //! physical replication of the values.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual bool isOutputBroadcastAcrossBatch(
        int32_t outputIndex, bool const* inputIsBroadcasted, int32_t nbInputs) const noexcept
        = 0;

    //! \brief Return true if plugin can use input that is broadcast across batch without replication.
    //!
    //! \param inputIndex Index of input that could be broadcast.
    //!
    //! For each input whose tensor is semantically broadcast across a batch,
    //! TensorRT calls this method before calling configurePlugin.
    //! If canBroadcastInputAcrossBatch returns true, TensorRT will not replicate the input tensor;
    //! i.e., there will be a single copy that the plugin should share across the batch.
    //! If it returns false, TensorRT will replicate the input tensor
    //! so that it appears like a non-broadcasted tensor.
    //!
    //! This method is called only for inputs that can be broadcast.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept = 0;

    //!
    //! \brief Configure the layer with input and output data types.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
    //! algorithm choices on the basis of its weights, dimensions, data types and maximum batch size.
    //!
    //! \param inputDims The input tensor dimensions.
    //! \param nbInputs The number of inputs.
    //! \param outputDims The output tensor dimensions.
    //! \param nbOutputs The number of outputs.
    //! \param inputTypes The data types selected for the plugin inputs.
    //! \param outputTypes The data types selected for the plugin outputs.
    //! \param inputIsBroadcast True for each input that the plugin must broadcast across the batch.
    //! \param outputIsBroadcast True for each output that TensorRT will broadcast across the batch.
    //! \param floatFormat The format selected for the engine for the floating point inputs/outputs.
    //! \param maxBatchSize The maximum batch size.
    //!
    //! The dimensions passed here do not include the outermost batch size (i.e. for 2-D image networks, they will be
    //! 3-dimensional CHW dimensions). When inputIsBroadcast or outputIsBroadcast is true, the outermost batch size for
    //! that input or output should be treated as if it is one.
    //! \ref inputIsBroadcast[i] is true only if the input is semantically broadcast across the batch and
    //! \ref canBroadcastInputAcrossBatch(i) returned true.
    //! \ref outputIsBroadcast[i] is true only if \ref isOutputBroadcastAcrossBatch(i) returns true.
    //!
    //! \warning for the floatFormat field, the values PluginFormat::kCHW4, PluginFormat::kCHW16, and
    //! PluginFormat::kCHW32 will not be passed in, this is to keep backward compatibility with TensorRT 5.x series. Use
    //! PluginV2IOExt or PluginV2DynamicExt for other PluginFormats.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin. However, TensorRT
    //!                  will not call this method from two threads simultaneously on a given clone of a plugin.
    //!
    virtual void configurePlugin(Dims const* inputDims, int32_t nbInputs, Dims const* outputDims, int32_t nbOutputs,
        DataType const* inputTypes, DataType const* outputTypes, bool const* inputIsBroadcast,
        bool const* outputIsBroadcast, PluginFormat floatFormat, int32_t maxBatchSize) noexcept
        = 0;

    IPluginV2Ext() = default;
    ~IPluginV2Ext() override = default;

    //!
    //! \brief Attach the plugin object to an execution context and grant the plugin the access to some context
    //! resource.
    //!
    //! \param cudnn The CUDNN context handle of the execution context
    //! \param cublas The cublas context handle of the execution context
    //! \param allocator The allocator used by the execution context
    //!
    //! This function is called automatically for each plugin when a new execution context is created. If the context
    //! was created without resources, this method is not called until the resources are assigned. It is also called if
    //! new resources are assigned to the context.
    //!
    //! If the plugin needs per-context resource, it can be allocated here.
    //! The plugin can also get context-owned CUDNN and CUBLAS context here.
    //!
    //! \note In the automotive safety context, the CUDNN and CUBLAS parameters will be nullptr because CUDNN and CUBLAS
    //!       is not used by the safe runtime.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void attachToContext(
        cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, IGpuAllocator* /*allocator*/) noexcept
    {
    }

    //!
    //! \brief Detach the plugin object from its execution context.
    //!
    //! This function is called automatically for each plugin when a execution context is destroyed or the context
    //! resources are unassigned from the context.
    //!
    //! If the plugin owns per-context resource, it can be released here.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual void detachFromContext() noexcept {}

    //!
    //! \brief Clone the plugin object. This copies over internal plugin parameters as well and returns a new plugin
    //! object with these parameters. If the source plugin is pre-configured with configurePlugin(), the returned object
    //! should also be pre-configured. The returned object should allow attachToContext() with a new execution context.
    //! Cloned plugin objects can share the same per-engine immutable resource (e.g. weights) with the source object
    //! (e.g. via ref-counting) to avoid duplication.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    IPluginV2Ext* clone() const noexcept override = 0;

protected:
    // @cond SuppressDoxyWarnings
    IPluginV2Ext(IPluginV2Ext const&) = default;
    IPluginV2Ext(IPluginV2Ext&&) = default;
    IPluginV2Ext& operator=(IPluginV2Ext const&) & = default;
    IPluginV2Ext& operator=(IPluginV2Ext&&) & = default;
// @endcond

    //!
    //! \brief Return the API version with which this plugin was built. The
    //!  upper byte reserved by TensorRT and is used to differentiate this from IPluginV2.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with
    //! plugins.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, the implementation provided here is safe to call from any thread.
    //!
    int32_t getTensorRTVersion() const noexcept override
    {
        return static_cast<int32_t>((static_cast<uint32_t>(PluginVersion::kV2_EXT) << 24U)
            | (static_cast<uint32_t>(NV_TENSORRT_VERSION) & 0xFFFFFFU));
    }

    //!
    //! \brief Derived classes should not implement this. In a C++11 API it would be override final.
    //!
    void configureWithFormat(Dims const* /*inputDims*/, int32_t /*nbInputs*/, Dims const* /*outputDims*/,
        int32_t /*nbOutputs*/, DataType /*type*/, PluginFormat /*format*/, int32_t /*maxBatchSize*/) noexcept override
    {
    }
};

//! \class IPluginV2IOExt
//!
//! \brief Plugin class for user-implemented layers.
//!
//! Plugins are a mechanism for applications to implement custom layers. This interface provides additional
//! capabilities to the IPluginV2Ext interface by extending different I/O data types and tensor formats.
//!
//! \see IPluginV2Ext
//!
class IPluginV2IOExt : public IPluginV2Ext
{
public:
    //!
    //! \brief Configure the layer.
    //!
    //! This function is called by the builder prior to initialize(). It provides an opportunity for the layer to make
    //! algorithm choices on the basis of I/O PluginTensorDesc and the maximum batch size.
    //!
    //! \param in The input tensors attributes that are used for configuration.
    //! \param nbInput Number of input tensors.
    //! \param out The output tensors attributes that are used for configuration.
    //! \param nbOutput Number of output tensors.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin. However, TensorRT
    //!                  will not call this method from two threads simultaneously on a given clone of a plugin.
    //!
    virtual void configurePlugin(
        PluginTensorDesc const* in, int32_t nbInput, PluginTensorDesc const* out, int32_t nbOutput) noexcept
        = 0;

    //!
    //! \brief Return true if plugin supports the format and datatype for the input/output indexed by pos.
    //!
    //! For this method inputs are numbered 0..(nbInputs-1) and outputs are numbered nbInputs..(nbInputs+nbOutputs-1).
    //! Using this numbering, pos is an index into InOut, where 0 <= pos < nbInputs+nbOutputs-1.
    //!
    //! TensorRT invokes this method to ask if the input/output indexed by pos supports the format/datatype specified
    //! by inOut[pos].format and inOut[pos].type. The override should return true if that format/datatype at inOut[pos]
    //! are supported by the plugin. If support is conditional on other input/output formats/datatypes, the plugin can
    //! make its result conditional on the formats/datatypes in inOut[0..pos-1], which will be set to values
    //! that the plugin supports. The override should not inspect inOut[pos+1..nbInputs+nbOutputs-1],
    //! which will have invalid values.  In other words, the decision for pos must be based on inOut[0..pos] only.
    //!
    //! Some examples:
    //!
    //! * A definition for a plugin that supports only FP16 NCHW:
    //!
    //!         return inOut.format[pos] == TensorFormat::kLINEAR && inOut.type[pos] == DataType::kHALF;
    //!
    //! * A definition for a plugin that supports only FP16 NCHW for its two inputs,
    //!   and FP32 NCHW for its single output:
    //!
    //!         return inOut.format[pos] == TensorFormat::kLINEAR &&
    //!                (inOut.type[pos] == (pos < 2 ?  DataType::kHALF : DataType::kFLOAT));
    //!
    //! * A definition for a "polymorphic" plugin with two inputs and one output that supports
    //!   any format or type, but the inputs and output must have the same format and type:
    //!
    //!         return pos == 0 || (inOut.format[pos] == inOut.format[0] && inOut.type[pos] == inOut.type[0]);
    //!
    //! Warning: TensorRT will stop asking for formats once it finds kFORMAT_COMBINATION_LIMIT on combinations.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin.
    //!
    virtual bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept
        = 0;

    // @cond SuppressDoxyWarnings
    IPluginV2IOExt() = default;
    ~IPluginV2IOExt() override = default;
// @endcond

protected:
// @cond SuppressDoxyWarnings
    IPluginV2IOExt(IPluginV2IOExt const&) = default;
    IPluginV2IOExt(IPluginV2IOExt&&) = default;
    IPluginV2IOExt& operator=(IPluginV2IOExt const&) & = default;
    IPluginV2IOExt& operator=(IPluginV2IOExt&&) & = default;
// @endcond

    //!
    //! \brief Return the API version with which this plugin was built. The upper byte is reserved by TensorRT and is
    //! used to differentiate this from IPluginV2 and IPluginV2Ext.
    //!
    //! Do not override this method as it is used by the TensorRT library to maintain backwards-compatibility with
    //! plugins.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, the implementation provided here is safe to call from any thread.
    //!
    int32_t getTensorRTVersion() const noexcept override
    {
        return static_cast<int32_t>((static_cast<uint32_t>(PluginVersion::kV2_IOEXT) << 24U)
            | (static_cast<uint32_t>(NV_TENSORRT_VERSION) & 0xFFFFFFU));
    }

private:
    // Following are obsolete base class methods, and must not be implemented or used.

    void configurePlugin(Dims const*, int32_t, Dims const*, int32_t, DataType const*, DataType const*, bool const*,
        bool const*, PluginFormat, int32_t) noexcept final
    {
    }

    bool supportsFormat(DataType, PluginFormat) const noexcept final
    {
        return false;
    }
};

//!
//! \enum FieldType
//! \brief The possible field types for custom layer.
//!

enum class PluginFieldType : int32_t
{
    //! FP16 field type.
    kFLOAT16 = 0,
    //! FP32 field type.
    kFLOAT32 = 1,
    //! FP64 field type.
    kFLOAT64 = 2,
    //! INT8 field type.
    kINT8 = 3,
    //! INT16 field type.
    kINT16 = 4,
    //! INT32 field type.
    kINT32 = 5,
    //! char field type.
    kCHAR = 6,
    //! nvinfer1::Dims field type.
    kDIMS = 7,
    //! Unknown field type.
    kUNKNOWN = 8
};

//!
//! \class PluginField
//!
//! \brief Structure containing plugin attribute field names and associated data
//! This information can be parsed to decode necessary plugin metadata
//!
//!
class PluginField
{
public:
    //!
    //! \brief Plugin field attribute name
    //!
    AsciiChar const* name;
    //!
    //! \brief Plugin field attribute data
    //!
    void const* data;
    //!
    //! \brief Plugin field attribute type
    //! \see PluginFieldType
    //!
    PluginFieldType type;
    //!
    //! \brief Number of data entries in the Plugin attribute
    //!
    int32_t length;

    PluginField(AsciiChar const* const name_ = nullptr, void const* const data_ = nullptr,
        PluginFieldType const type_ = PluginFieldType::kUNKNOWN, int32_t const length_ = 0) noexcept
        : name(name_)
        , data(data_)
        , type(type_)
        , length(length_)
    {
    }
};

//! Plugin field collection struct.
struct PluginFieldCollection
{
    //! Number of PluginField entries.
    int32_t nbFields;
    //! Pointer to PluginField entries.
    PluginField const* fields;
};

//!
//! \class IPluginCreator
//!
//! \brief Plugin creator class for user implemented layers.
//!
//! \see IPlugin and IPluginFactory
//!

class IPluginCreator
{
public:
    //!
    //! \brief Return the version of the API the plugin creator was compiled with.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, the implementation provided here is safe to call from any thread.
    //!
    virtual int32_t getTensorRTVersion() const noexcept
    {
        return NV_TENSORRT_VERSION;
    }

    //!
    //! \brief Return the plugin name.
    //!
    //! \warning The string returned must be 1024 bytes or less including the NULL terminator and must be NULL
    //! terminated.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual AsciiChar const* getPluginName() const noexcept = 0;

    //!
    //! \brief Return the plugin version.
    //!
    //! \warning The string returned must be 1024 bytes or less including the NULL terminator and must be NULL
    //! terminated.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual AsciiChar const* getPluginVersion() const noexcept = 0;

    //!
    //! \brief Return a list of fields that needs to be passed to createPlugin.
    //! \see PluginFieldCollection
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual PluginFieldCollection const* getFieldNames() noexcept = 0;

    //!
    //! \brief Return a plugin object. Return nullptr in case of error.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual IPluginV2* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc) noexcept = 0;

    //!
    //! \brief Called during deserialization of plugin layer. Return a plugin object.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual IPluginV2* deserializePlugin(AsciiChar const* name, void const* serialData, size_t serialLength) noexcept
        = 0;

    //!
    //! \brief Set the namespace of the plugin creator based on the plugin
    //! library it belongs to. This can be set while registering the plugin creator.
    //!
    //! \see IPluginRegistry::registerCreator()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual void setPluginNamespace(AsciiChar const* pluginNamespace) noexcept = 0;

    //!
    //! \brief Return the namespace of the plugin creator object.
    //!
    //! \warning The string returned must be 1024 bytes or less including the NULL terminator and must be NULL
    //! terminated.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when building networks on multiple devices sharing the same plugin or when deserializing
    //!                  multiple engines concurrently sharing plugins.
    //!
    virtual AsciiChar const* getPluginNamespace() const noexcept = 0;

    IPluginCreator() = default;
    virtual ~IPluginCreator() = default;

protected:
// @cond SuppressDoxyWarnings
    IPluginCreator(IPluginCreator const&) = default;
    IPluginCreator(IPluginCreator&&) = default;
    IPluginCreator& operator=(IPluginCreator const&) & = default;
    IPluginCreator& operator=(IPluginCreator&&) & = default;
// @endcond
};

//!
//! \class IPluginRegistry
//!
//! \brief Single registration point for all plugins in an application. It is
//! used to find plugin implementations during engine deserialization.
//! Internally, the plugin registry is considered to be a singleton so all
//! plugins in an application are part of the same global registry.
//! Note that the plugin registry is only supported for plugins of type
//! IPluginV2 and should also have a corresponding IPluginCreator implementation.
//!
//! \see IPluginV2 and IPluginCreator
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \warning In the automotive safety context, be sure to call IPluginRegistry::setErrorRecorder() to register
//! an error recorder with the registry before using other methods in the registry.
//!

class IPluginRegistry
{
public:
    //!
    //! \brief Register a plugin creator. Returns false if one with same type
    //! is already registered.
    //!
    //! \warning The string pluginNamespace must be 1024 bytes or less including the NULL terminator and must be NULL
    //! terminated.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes; calls to this method will be synchronized by a mutex.
    //!
    virtual bool registerCreator(IPluginCreator& creator, AsciiChar const* const pluginNamespace) noexcept = 0;

    //!
    //! \brief Return all the registered plugin creators and the number of
    //! registered plugin creators. Returns nullptr if none found.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual IPluginCreator* const* getPluginCreatorList(int32_t* const numCreators) const noexcept = 0;

    //!
    //! \brief Return plugin creator based on plugin name, version, and
    //! namespace associated with plugin during network creation.
    //!
    //! \warning The strings pluginName, pluginVersion, and pluginNamespace must be 1024 bytes or less including the
    //! NULL terminator and must be NULL terminated.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual IPluginCreator* getPluginCreator(AsciiChar const* const pluginName, AsciiChar const* const pluginVersion,
        AsciiChar const* const pluginNamespace = "") noexcept
        = 0;

    // @cond SuppressDoxyWarnings
    IPluginRegistry() = default;
    IPluginRegistry(IPluginRegistry const&) = delete;
    IPluginRegistry(IPluginRegistry&&) = delete;
    IPluginRegistry& operator=(IPluginRegistry const&) & = delete;
    IPluginRegistry& operator=(IPluginRegistry&&) & = delete;
// @endcond

protected:
    virtual ~IPluginRegistry() noexcept = default;

public:
    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual void setErrorRecorder(IErrorRecorder* const recorder) noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called, or an ErrorRecorder has not been
    //! inherited.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

    //!
    //! \brief Deregister a previously registered plugin creator.
    //!
    //! Since there may be a desire to limit the number of plugins,
    //! this function provides a mechanism for removing plugin creators registered in TensorRT.
    //! The plugin creator that is specified by \p creator is removed from TensorRT and no longer tracked.
    //!
    //! \return True if the plugin creator was deregistered, false if it was not found in the registry or otherwise
    //! could
    //!     not be deregistered.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual bool deregisterCreator(IPluginCreator const& creator) noexcept = 0;
};

enum class AllocatorFlag : int32_t
{
    kRESIZABLE = 0, //!< TensorRT may call realloc() on this allocation
};

namespace impl
{
//! Maximum number of elements in AllocatorFlag enum. \see AllocatorFlag
template <>
struct EnumMaxImpl<AllocatorFlag>
{
    static constexpr int32_t kVALUE = 1;        //!< maximum number of elements in AllocatorFlag enum
};
} // namespace impl

using AllocatorFlags = uint32_t;

//!
//! \class IGpuAllocator
//!
//! \brief Application-implemented class for controlling allocation on the GPU.
//!
class IGpuAllocator
{
public:
    //!
    //! A thread-safe callback implemented by the application to handle acquisition of GPU memory.
    //!
    //! \param size The size of the memory required.
    //! \param alignment The required alignment of memory. Alignment will be zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //!
    //! If an allocation request of size 0 is made, nullptr should be returned.
    //!
    //! If an allocation request cannot be satisfied, nullptr should be returned.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/free/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual void* allocate(uint64_t const size, uint64_t const alignment, AllocatorFlags const flags) noexcept = 0;

    //!
    //! A thread-safe callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory The acquired memory.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/free/reallocate/deallocate
    //! requests.
    //!
    //! \see deallocate()
    //!
    //! \deprecated Superseded by deallocate. Deprecated in TensorRT 8.0.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    TRT_DEPRECATED virtual void free(void* const memory) noexcept = 0;

    //!
    //! Destructor declared virtual as general good practice for a class with virtual methods.
    //! TensorRT never calls the destructor for an IGpuAllocator defined by the application.
    //!
    virtual ~IGpuAllocator() = default;
    IGpuAllocator() = default;

    //!
    //! A thread-safe callback implemented by the application to resize an existing allocation.
    //!
    //! Only allocations which were allocated with AllocatorFlag::kRESIZABLE will be resized.
    //!
    //! Options are one of:
    //! * resize in place leaving min(oldSize, newSize) bytes unchanged and return the original address
    //! * move min(oldSize, newSize) bytes to a new location of sufficient size and return its address
    //! * return nullptr, to indicate that the request could not be fulfilled.
    //!
    //! If nullptr is returned, TensorRT will assume that resize() is not implemented, and that the
    //! allocation at baseAddr is still valid.
    //!
    //! This method is made available for use cases where delegating the resize
    //! strategy to the application provides an opportunity to improve memory management.
    //! One possible implementation is to allocate a large virtual device buffer and
    //! progressively commit physical memory with cuMemMap. CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    //! is suggested in this case.
    //!
    //! TensorRT may call realloc to increase the buffer by relatively small amounts.
    //!
    //! \param baseAddr the address of the original allocation.
    //! \param alignment The alignment used by the original allocation.
    //! \param newSize The new memory size required.
    //! \return the address of the reallocated memory
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/free/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual void* reallocate(void* /*baseAddr*/, uint64_t /*alignment*/, uint64_t /*newSize*/) noexcept
    {
        return nullptr;
    }

    //!
    //! A thread-safe callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory The acquired memory.
    //! \return True if the acquired memory is released successfully.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/free/reallocate/deallocate
    //! requests.
    //!
    //! \note If user-implemented free() might hit an error condition, the user should override deallocate() as the
    //! primary implementation and override free() to call deallocate() for backwards compatibility.
    //!
    //! \see free()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual bool deallocate(void* const memory) noexcept
    {
        this->free(memory);
        return true;
    }

protected:
// @cond SuppressDoxyWarnings
    IGpuAllocator(IGpuAllocator const&) = default;
    IGpuAllocator(IGpuAllocator&&) = default;
    IGpuAllocator& operator=(IGpuAllocator const&) & = default;
    IGpuAllocator& operator=(IGpuAllocator&&) & = default;
// @endcond
};

//!
//! \class ILogger
//!
//! \brief Application-implemented logging interface for the builder, refitter and runtime.
//!
//! The logger used to create an instance of IBuilder, IRuntime or IRefitter is used for all objects created through
//! that interface. The logger should be valid until all objects created are released.
//!
//! The Logger object implementation must be thread safe. All locking and synchronization is pushed to the
//! interface implementation and TensorRT does not hold any synchronization primitives when calling the interface
//! functions.
//!
class ILogger
{
public:
    //!
    //! \enum Severity
    //!
    //! The severity corresponding to a log message.
    //!
    enum class Severity : int32_t
    {
        //! An internal error has occurred. Execution is unrecoverable.
        kINTERNAL_ERROR = 0,
        //! An application error has occurred.
        kERROR = 1,
        //! An application error has been discovered, but TensorRT has recovered or fallen back to a default.
        kWARNING = 2,
        //!  Informational messages with instructional information.
        kINFO = 3,
        //!  Verbose messages with debugging information.
        kVERBOSE = 4,
    };

    //!
    //! A callback implemented by the application to handle logging messages;
    //!
    //! \param severity The severity of the message.
    //! \param msg A null-terminated log message.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime, or if the same logger is used
    //!                  for multiple runtimes, builders, or refitters.
    //!
    virtual void log(Severity severity, AsciiChar const* msg) noexcept = 0;

    ILogger() = default;
    virtual ~ILogger() = default;

protected:
// @cond SuppressDoxyWarnings
    ILogger(ILogger const&) = default;
    ILogger(ILogger&&) = default;
    ILogger& operator=(ILogger const&) & = default;
    ILogger& operator=(ILogger&&) & = default;
// @endcond
};

namespace impl
{
//! Maximum number of elements in ILogger::Severity enum. \see ILogger::Severity
template <>
struct EnumMaxImpl<ILogger::Severity>
{
    //! Declaration of kVALUE that represents maximum number of elements in ILogger::Severity enum
    static constexpr int32_t kVALUE = 5;
};
} // namespace impl

//!
//! \enum ErrorCode
//!
//! \brief Error codes that can be returned by TensorRT during execution.
//!
enum class ErrorCode : int32_t
{
    //!
    //! Execution completed successfully.
    //!
    kSUCCESS = 0,

    //!
    //! An error that does not fall into any other category. This error is included for forward compatibility.
    //!
    kUNSPECIFIED_ERROR = 1,

    //!
    //! A non-recoverable TensorRT error occurred. TensorRT is in an invalid internal state when this error is
    //! emitted and any further calls to TensorRT will result in undefined behavior.
    //!
    kINTERNAL_ERROR = 2,

    //!
    //! An argument passed to the function is invalid in isolation.
    //! This is a violation of the API contract.
    //!
    kINVALID_ARGUMENT = 3,

    //!
    //! An error occurred when comparing the state of an argument relative to other arguments. For example, the
    //! dimensions for concat differ between two tensors outside of the channel dimension. This error is triggered
    //! when an argument is correct in isolation, but not relative to other arguments. This is to help to distinguish
    //! from the simple errors from the more complex errors.
    //! This is a violation of the API contract.
    //!
    kINVALID_CONFIG = 4,

    //!
    //! An error occurred when performing an allocation of memory on the host or the device.
    //! A memory allocation error is normally fatal, but in the case where the application provided its own memory
    //! allocation routine, it is possible to increase the pool of available memory and resume execution.
    //!
    kFAILED_ALLOCATION = 5,

    //!
    //! One, or more, of the components that TensorRT relies on did not initialize correctly.
    //! This is a system setup issue.
    //!
    kFAILED_INITIALIZATION = 6,

    //!
    //! An error occurred during execution that caused TensorRT to end prematurely, either an asynchronous error or
    //! other execution errors reported by CUDA/DLA. In a dynamic system, the
    //! data can be thrown away and the next frame can be processed or execution can be retried.
    //! This is either an execution error or a memory error.
    //!
    kFAILED_EXECUTION = 7,

    //!
    //! An error occurred during execution that caused the data to become corrupted, but execution finished. Examples
    //! of this error are NaN squashing or integer overflow. In a dynamic system, the data can be thrown away and the
    //! next frame can be processed or execution can be retried.
    //! This is either a data corruption error, an input error, or a range error.
    //! This is not used in safety but may be used in standard.
    //!
    kFAILED_COMPUTATION = 8,

    //!
    //! TensorRT was put into a bad state by incorrect sequence of function calls. An example of an invalid state is
    //! specifying a layer to be DLA only without GPU fallback, and that layer is not supported by DLA. This can occur
    //! in situations where a service is optimistically executing networks for multiple different configurations
    //! without checking proper error configurations, and instead throwing away bad configurations caught by TensorRT.
    //! This is a violation of the API contract, but can be recoverable.
    //!
    //! Example of a recovery:
    //! GPU fallback is disabled and conv layer with large filter(63x63) is specified to run on DLA. This will fail due
    //! to DLA not supporting the large kernel size. This can be recovered by either turning on GPU fallback
    //! or setting the layer to run on the GPU.
    //!
    kINVALID_STATE = 9,

    //!
    //! An error occurred due to the network not being supported on the device due to constraints of the hardware or
    //! system. An example is running a unsafe layer in a safety certified context, or a resource requirement for the
    //! current network is greater than the capabilities of the target device. The network is otherwise correct, but
    //! the network and hardware combination is problematic. This can be recoverable.
    //! Examples:
    //!  * Scratch space requests larger than available device memory and can be recovered by increasing allowed
    //!    workspace size.
    //!  * Tensor size exceeds the maximum element count and can be recovered by reducing the maximum batch size.
    //!
    kUNSUPPORTED_STATE = 10,

};

namespace impl
{
//! Maximum number of elements in ErrorCode enum. \see ErrorCode
template <>
struct EnumMaxImpl<ErrorCode>
{
    //! Declaration of kVALUE
    static constexpr int32_t kVALUE = 11;
};
} // namespace impl

//!
//! \class IErrorRecorder
//!
//! \brief Reference counted application-implemented error reporting interface for TensorRT objects.
//!
//! The error reporting mechanism is a user defined object that interacts with the internal state of the object
//! that it is assigned to in order to determine information about abnormalities in execution. The error recorder
//! gets both an error enum that is more descriptive than pass/fail and also a string description that gives more
//! detail on the exact failure modes. In the safety context, the error strings are all limited to 1024 characters
//! in length.
//!
//! The ErrorRecorder gets passed along to any class that is created from another class that has an ErrorRecorder
//! assigned to it. For example, assigning an ErrorRecorder to an IBuilder allows all INetwork's, ILayer's, and
//! ITensor's to use the same error recorder. For functions that have their own ErrorRecorder accessor functions.
//! This allows registering a different error recorder or de-registering of the error recorder for that specific
//! object.
//!
//! The ErrorRecorder object implementation must be thread safe. All locking and synchronization is pushed to the
//! interface implementation and TensorRT does not hold any synchronization primitives when calling the interface
//! functions.
//!
//! The lifetime of the ErrorRecorder object must exceed the lifetime of all TensorRT objects that use it.
//!
class IErrorRecorder
{
public:
    //!
    //! A typedef of a C-style string for reporting error descriptions.
    //!
    using ErrorDesc = char const*;

    //!
    //! The length limit for an error description, excluding the '\0' string terminator.
    //!
    static constexpr size_t kMAX_DESC_LENGTH{127U};

    //!
    //! A typedef of a 32bit integer for reference counting.
    //!
    using RefCount = int32_t;

    IErrorRecorder() = default;
    virtual ~IErrorRecorder() noexcept = default;

    // Public API used to retrieve information from the error recorder.

    //!
    //! \brief Return the number of errors
    //!
    //! Determines the number of errors that occurred between the current point in execution
    //! and the last time that the clear() was executed. Due to the possibility of asynchronous
    //! errors occuring, a TensorRT API can return correct results, but still register errors
    //! with the Error Recorder. The value of getNbErrors must monotonically increases until clear()
    //! is called.
    //!
    //! \return Returns the number of errors detected, or 0 if there are no errors.
    //!
    //! \see clear
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual int32_t getNbErrors() const noexcept = 0;

    //!
    //! \brief Returns the ErrorCode enumeration.
    //!
    //! \param errorIdx A 32-bit integer that indexes into the error array.
    //!
    //! The errorIdx specifies what error code from 0 to getNbErrors()-1 that the application
    //! wants to analyze and return the error code enum.
    //!
    //! \return Returns the enum corresponding to errorIdx.
    //!
    //! \see getErrorDesc, ErrorCode
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual ErrorCode getErrorCode(int32_t errorIdx) const noexcept = 0;

    //!
    //! \brief Returns a null-terminated C-style string description of the error.
    //!
    //! \param errorIdx A 32-bit integer that indexes into the error array.
    //!
    //! For the error specified by the idx value, return the string description of the error. The
    //! error string is a null-terminated C-style string. In the safety context there is a
    //! constant length requirement to remove any dynamic memory allocations and the error message
    //! may be truncated. The format of the string is "<EnumAsStr> - <Description>".
    //!
    //! \return Returns a string representation of the error along with a description of the error.
    //!
    //! \see getErrorCode
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept = 0;

    //!
    //! \brief Determine if the error stack has overflowed.
    //!
    //! In the case when the number of errors is large, this function is used to query if one or more
    //! errors have been dropped due to lack of storage capacity. This is especially important in the
    //! automotive safety case where the internal error handling mechanisms cannot allocate memory.
    //!
    //! \return true if errors have been dropped due to overflowing the error stack.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual bool hasOverflowed() const noexcept = 0;

    //!
    //! \brief Clear the error stack on the error recorder.
    //!
    //! Removes all the tracked errors by the error recorder.  This function must guarantee that after
    //! this function is called, and as long as no error occurs, the next call to getNbErrors will return
    //! zero.
    //!
    //! \see getNbErrors
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual void clear() noexcept = 0;

    // API used by TensorRT to report Error information to the application.

    //!
    //! \brief Report an error to the error recorder with the corresponding enum and description.
    //!
    //! \param val The error code enum that is being reported.
    //! \param desc The string description of the error.
    //!
    //! Report an error to the user that has a given value and human readable description. The function returns false
    //! if processing can continue, which implies that the reported error is not fatal. This does not guarantee that
    //! processing continues, but provides a hint to TensorRT.
    //! The desc C-string data is only valid during the call to reportError and may be immediately deallocated by the
    //! caller when reportError returns. The implementation must not store the desc pointer in the ErrorRecorder object
    //! or otherwise access the data from desc after reportError returns.
    //!
    //! \return True if the error is determined to be fatal and processing of the current function must end.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual bool reportError(ErrorCode val, ErrorDesc desc) noexcept = 0;

    //!
    //! \brief Increments the refcount for the current ErrorRecorder.
    //!
    //! Increments the reference count for the object by one and returns the current value.  This reference count allows
    //! the application to know that an object inside of TensorRT has taken a reference to the ErrorRecorder.  TensorRT
    //! guarantees that every call to IErrorRecorder::incRefCount will be paired with a call to
    //! IErrorRecorder::decRefCount when the reference is released.  It is undefined behavior to destruct the
    //! ErrorRecorder when incRefCount has been called without a corresponding decRefCount.
    //!
    //! \return The reference counted value after the increment completes.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual RefCount incRefCount() noexcept = 0;

    //!
    //! \brief Decrements the refcount for the current ErrorRecorder.
    //!
    //! Decrements the reference count for the object by one and returns the current value.  This reference count allows
    //! the application to know that an object inside of TensorRT has taken a reference to the ErrorRecorder.  TensorRT
    //! guarantees that every call to IErrorRecorder::decRefCount will be preceded by a call to
    //! IErrorRecorder::incRefCount.  It is undefined behavior to destruct the ErrorRecorder when incRefCount has been
    //! called without a corresponding decRefCount.
    //!
    //! \return The reference counted value after the decrement completes.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads
    //!                  when multiple execution contexts are used during runtime.
    //!
    virtual RefCount decRefCount() noexcept = 0;

protected:
    // @cond SuppressDoxyWarnings
    IErrorRecorder(IErrorRecorder const&) = default;
    IErrorRecorder(IErrorRecorder&&) = default;
    IErrorRecorder& operator=(IErrorRecorder const&) & = default;
    IErrorRecorder& operator=(IErrorRecorder&&) & = default;
    // @endcond
}; // class IErrorRecorder
} // namespace nvinfer1

//!
//! \brief Return the library version number.
//!
//! The format is as for TENSORRT_VERSION: (TENSORRT_MAJOR * 1000) + (TENSORRT_MINOR * 100) + TENSOR_PATCH.
//!
extern "C" TENSORRTAPI int32_t getInferLibVersion() noexcept;

#endif // NV_INFER_RUNTIME_COMMON_H
