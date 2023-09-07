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

#ifndef NV_INFER_RUNTIME_BASE_H
#define NV_INFER_RUNTIME_BASE_H

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
//! \file NvInferRuntimeBase.h
//!
//! This file contains common definitions, data structures and interfaces shared between the standard and safe runtime.
//!
//! \warning Do not directly include this file. Instead include either NvInferRuntime.h (for the standard runtime) or
//! NvInferSafeRuntime.h (for the safety runtime).
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
//! This type is used by IPluginV2, PluginField, IPluginCreator, IPluginRegistry, and
//! ILogger due to their use in automotive safety context.
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

    //! IEEE 16-bit floating-point format -- has a 5 bit exponent and 11 bit significand.
    kHALF = 1,

    //! Signed 8-bit integer representing a quantized floating-point value.
    kINT8 = 2,

    //! Signed 32-bit integer format.
    kINT32 = 3,

    //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
    kBOOL = 4,

    //! Unsigned 8-bit integer format.
    //! Cannot be used to represent quantized floating-point values.
    //! Use the IdentityLayer to convert kUINT8 network-level inputs to {kFLOAT, kHALF} prior
    //! to use with other TensorRT layers, or to convert intermediate output
    //! before kUINT8 network-level outputs from {kFLOAT, kHALF} to kUINT8.
    //! kUINT8 conversions are only supported for {kFLOAT, kHALF}.
    //! kUINT8 to {kFLOAT, kHALF} conversion will convert the integer values
    //! to equivalent floating point values.
    //! {kFLOAT, kHALF} to kUINT8 conversion will convert the floating point values
    //! to integer values by truncating towards zero. This conversion has undefined behavior for
    //! floating point values outside the range [0.0f, 256.0f) after truncation.
    //! kUINT8 conversions are not supported for {kINT8, kINT32, kBOOL}.
    kUINT8 = 5,

    //! Signed 8-bit floating point with
    //! 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7.
    kFP8 = 6,

    //! Brain float -- has an 8 bit exponent and 8 bit significand.
    kBF16 = 7,

    //! Signed 64-bit integer type.
    kINT64 = 8,
};

namespace impl
{
//! Maximum number of elements in DataType enum. \see DataType
template <>
struct EnumMaxImpl<DataType>
{
    // Declaration of kVALUE that represents maximum number of elements in DataType enum
    static constexpr int32_t kVALUE = 9;
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
    //! it needs to meet line stride requirement of DLA format. Column stride in bytes must
    //! be a multiple of 32 on Xavier and 64 on Orin.
    kCHW4 = 3,

    //! Sixteen wide channel vectorized row major format. This format is bound
    //! to FP16. It is only available for dimensions >= 3.
    //! For a tensor with dimensions {N, C, H, W},
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][(C+15)/16][H][W][16], with the tensor coordinates (n, c, h, w)
    //! mapping to array subscript [n][c/16][h][w][c%16].
    //!
    //! For DLA usage, this format maps to the native feature format for FP16,
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
    //! For DLA usage, this format maps to the native feature format for INT8,
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

    //! Non-vectorized channel-last format. This format is bound to either FP32 or UINT8,
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
    kHWC16 = 11,

    //! Non-vectorized channel-last format. This format is bound to FP32.
    //! It is only available for dimensions >= 4.
    kDHWC = 12
};

namespace impl
{
//! Maximum number of elements in TensorFormat enum. \see TensorFormat
template <>
struct EnumMaxImpl<TensorFormat>
{
    //! Declaration of kVALUE that represents maximum number of elements in TensorFormat enum
    static constexpr int32_t kVALUE = 13;
};
} // namespace impl

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
//! \warning The lifetime of an IGpuAllocator object must exceed that of all objects that use it.
//!
class IGpuAllocator
{
public:
    //!
    //! A thread-safe callback implemented by the application to handle acquisition of GPU memory.
    //!
    //! \param size The size of the memory block required (in bytes).
    //! \param alignment The required alignment of memory. Alignment will be zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //!
    //! \return If the allocation was successful, the start address of a device memory block of the requested size.
    //! If an allocation request of size 0 is made, nullptr must be returned.
    //! If an allocation request cannot be satisfied, nullptr must be returned.
    //! If a non-null address is returned, it is guaranteed to have the specified alignment.
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
    //! \param memory A memory address that was previously returned by calling allocate() or reallocate() on the same
    //!        allocator object, or nullptr.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/free/reallocate/deallocate
    //! requests.
    //!
    //! \see deallocate()
    //!
    //! \deprecated Deprecated in TensorRT 8.0. Superseded by deallocate.
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
    //! \param baseAddr the address of the original allocation, which will have been returned by previously calling
    //!        allocate() or reallocate() on the same object.
    //! \param alignment The alignment used by the original allocation. This will be the same value that was previously
    //!        passed to the allocate() or reallocate() call that returned baseAddr.
    //! \param newSize The new memory size required (in bytes).
    //! \return The address of the reallocated memory, or nullptr. If a non-null address is returned, it is
    //!         guaranteed to have the specified alignment.
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
    //! \param memory A memory address that was previously returned by an allocate() or reallocate() call of the same allocator
    //!        object.
    //! \return True if the acquired memory is released successfully.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/free/reallocate/deallocate
    //! requests.
    //!
    //! \note If user-implemented free() might hit an error condition, the user must override deallocate() as the
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
//! that interface. The logger must be valid until all objects created are released.
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
    //! \warning Loggers used in the safety certified runtime must set a maximum message length and truncate
    //!          messages exceeding this length. It is up to the implementer of the derived class to define
    //!          a suitable limit that will prevent buffer overruns, resource exhaustion, and other security
    //!          vulnerabilities in their implementation. The TensorRT safety certified runtime will never
    //!          emit messages longer than 1024 bytes.
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
    //! An error occurred during execution that caused TensorRT to end prematurely, either an asynchronous error,
    //! user cancellation, or other execution errors reported by CUDA/DLA. In a dynamic system, the
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
//! detail on the exact failure modes. In the safety context, the error strings are all limited to 128 bytes
//! or less in length, including the NULL terminator.
//!
//! The ErrorRecorder gets passed along to any class that is created from another class that has an ErrorRecorder
//! assigned to it. For example, assigning an ErrorRecorder to an IBuilder allows all INetwork's, ILayer's, and
//! ITensor's to use the same error recorder. For functions that have their own ErrorRecorder accessor functions.
//! This allows registering a different error recorder or de-registering of the error recorder for that specific
//! object.
//!
//! ErrorRecorder objects that are used in the safety runtime must define an implementation-dependent upper limit
//! of errors whose information can be stored, and drop errors above this upper limit. The limit must fit in int32_t.
//! The IErrorRecorder::hasOverflowed() method is used to signal that one or more errors have been dropped.
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
    //! The length limit for an error description in bytes, excluding the '\0' string terminator.
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
    //! with the Error Recorder. The value of getNbErrors() must increment by 1 after each reportError()
    //! call until clear() is called, or the maximum number of errors that can be stored is exceeded.
    //!
    //! \return Returns the number of errors detected, or 0 if there are no errors.
    //!         If the upper bound of errors that can be stored is exceeded, the upper bound value must
    //!         be returned.
    //!
    //! For example, if the error recorder can store up to 16 error descriptions but recordError() has
    //! been called 20 times, getNbErrors() must return 16.
    //!
    //! \see clear(), hasOverflowed()
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
    //! \return Returns the enum corresponding to errorIdx if errorIdx is in range (between 0 and getNbErrors()-1).
    //!         ErrorCode::kUNSPECIFIED_ERROR must be returned if errorIdx is not in range.
    //!
    //! \see getErrorDesc(), ErrorCode
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
    //! will be truncated if it exceeds kMAX_DESC_LENGTH bytes.
    //! The format of the string is "<EnumAsStr> - <Description>".
    //!
    //! \return Returns a string representation of the error along with a description of the error if errorIdx is in
    //!         range (between 0 and getNbErrors()-1). An empty string will be returned if errorIdx is not in range.
    //!
    //! \see getErrorCode()
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
    //! Removes all the tracked errors by the error recorder.  The implementation must guarantee that after
    //! this function is called, and as long as no error occurs, the next call to getNbErrors will return
    //! zero and hasOverflowed will return false.
    //!
    //! \see getNbErrors(), hasOverflowed()
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
    //! \param desc The string description of the error, which will be a NULL-terminated string of kMAX_DESC_LENGTH
    //!        bytes or less (excluding the NULL terminator). Descriptions that exceed this limit will be silently
    //!        truncated.
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
    //! \warning If the error recorder's maximum number of storable errors is exceeded, the error description will be
    //!          silently dropped and the value returned by getNbErrors() will not be incremented. However, the return
    //!          value will still signal whether the error must be considered fatal.
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
    //! guarantees that every call to IErrorRecorder::incRefCount() will be paired with a call to
    //! IErrorRecorder::decRefCount() when the reference is released.  It is undefined behavior to destruct the
    //! ErrorRecorder when incRefCount() has been called without a corresponding decRefCount().
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
    //! guarantees that every call to IErrorRecorder::decRefCount() will be preceded by a call to
    //! IErrorRecorder::incRefCount().  It is undefined behavior to destruct the ErrorRecorder when incRefCount() has been
    //! called without a corresponding decRefCount().
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

//!
//! \enum TensorIOMode
//!
//! \brief Definition of tensor IO Mode.
//!
enum class TensorIOMode : int32_t
{
    //! Tensor is not an input or output.
    kNONE = 0,

    //! Tensor is input to the engine.
    kINPUT = 1,

    //! Tensor is output by the engine.
    kOUTPUT = 2
};

namespace impl
{
//! Maximum number of elements in TensorIOMode enum. \see TensorIOMode
template <>
struct EnumMaxImpl<TensorIOMode>
{
    // Declaration of kVALUE that represents maximum number of elements in TensorIOMode enum
    static constexpr int32_t kVALUE = 3;
};
} // namespace impl
} // namespace nvinfer1

//!
//! \brief Return the library version number.
//!
//! The format is as for TENSORRT_VERSION: (TENSORRT_MAJOR * 1000) + (TENSORRT_MINOR * 100) + TENSOR_PATCH.
//!
extern "C" TENSORRTAPI int32_t getInferLibVersion() noexcept;

#endif // NV_INFER_RUNTIME_BASE_H
