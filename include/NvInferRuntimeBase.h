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
//! \warning Do not directly include this file. Instead include one of:
//! * NvInferRuntime.h (for the standard runtime)
//! * NvInferSafeRuntime.h (for the safety runtime)
//! * NvInferConsistency.h (for consistency checker)
//! * NvInferPluginUtils.h (for plugin utilities)
//!
#if !defined(NV_INFER_INTERNAL_INCLUDE_RUNTIME_BASE)
static_assert(false, "Do not directly include this file. Include NvInferRuntime.h or NvInferSafeRuntime.h or NvInferConsistency.h or NvInferPluginUtils.h");
#endif

//! Forward declare some CUDA types to avoid an include dependency.

extern "C"
{
    //! Forward declaration of cublasContext to use in other interfaces.
    struct cublasContext;
    //! Forward declaration of cudnnContext to use in other interfaces.
    struct cudnnContext;
}

//! Construct a single integer denoting TensorRT version.
//! Usable in preprocessor expressions.
#define NV_TENSORRT_VERSION_INT(major, minor, patch) ((major) *10000L + (minor) *100L + (patch) *1L)

//! TensorRT version as a single integer.
//! Usable in preprocessor expressions.
#define NV_TENSORRT_VERSION NV_TENSORRT_VERSION_INT(NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH)

//!
//! \namespace nvinfer1
//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{
//! char_t is the type used by TensorRT to represent all valid characters.
using char_t = char;

//! AsciiChar is the type used by TensorRT to represent valid ASCII characters.
//! This type is widely used in automotive safety context.
using AsciiChar = char_t;

//! Forward declare IErrorRecorder for use in other interfaces.
namespace v_1_0
{
class IErrorRecorder;
}
using IErrorRecorder = v_1_0::IErrorRecorder;

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
    //! floating point values outside the range [0.0F, 256.0F) after truncation.
    //! kUINT8 conversions are not supported for {kINT8, kINT32, kBOOL}.
    kUINT8 = 5,

    //! Signed 8-bit floating point with
    //! 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7.
    kFP8 = 6,

    //! Brain float -- has an 8 bit exponent and 8 bit significand.
    kBF16 = 7,

    //! Signed 64-bit integer type.
    kINT64 = 8,

    //! Signed 4-bit integer type.
    kINT4 = 9,

};

namespace impl
{
//! Maximum number of elements in DataType enum. \see DataType
template <>
struct EnumMaxImpl<DataType>
{
//! Declaration of kVALUE that represents the maximum number of elements in the DataType enum.
    static constexpr int32_t kVALUE = 10;
};
} // namespace impl

//!
//! \class Dims
//! \brief Structure to define the dimensions of a tensor.
//!
//! TensorRT can also return an "invalid dims" structure. This structure is
//! represented by nbDims == -1 and d[i] == 0 for all i.
//!
//! TensorRT can also return an "unknown rank" dims structure. This structure is
//! represented by nbDims == -1 and d[i] == -1 for all i.
//!
class Dims64
{
public:
    //! The maximum rank (number of dimensions) supported for a tensor.
    static constexpr int32_t MAX_DIMS{8};

    //! The rank (number of dimensions).
    int32_t nbDims;

    //! The extent of each dimension.
    int64_t d[MAX_DIMS];
};

//!
//! Alias for Dims64.
//!
using Dims = Dims64;

//!
//! \enum TensorFormat
//!
//! \brief Format of the input/output tensors.
//!
//! This enum is used by both plugins and network I/O tensors.
//!
//! \see IPluginV2::supportsFormat(), safe::ICudaEngine::getBindingFormat()
//!
//! Many of the formats are **vector-major** or **vector-minor**. These formats specify
//! a <em>vector dimension</em> and <em>scalars per vector</em>.
//! For example, suppose that the tensor has has dimensions [M,N,C,H,W],
//! the vector dimension is C and there are V scalars per vector.
//!
//! * A **vector-major** format splits the vectorized dimension into two axes in the
//!   memory layout. The vectorized dimension is replaced by an axis of length ceil(C/V)
//!   and a new dimension of length V is appended. For the example tensor, the memory layout
//!   is equivalent to an array with dimensions [M][N][ceil(C/V)][H][W][V].
//!   Tensor coordinate (m,n,c,h,w) maps to array location [m][n][c/V][h][w][c\%V].
//!
//! * A **vector-minor** format moves the vectorized dimension to become the last axis
//!   in the memory layout. For the example tensor, the memory layout is equivalent to an
//!   array with dimensions [M][N][H][W][ceil(C/V)*V]. Tensor coordinate (m,n,c,h,w) maps
//!   array location subscript [m][n][h][w][c].
//!
//! In interfaces that refer to "components per element", that's the value of V above.
//!
//! For more information about data formats, see the topic "Data Format Description" located in the
//! TensorRT Developer Guide. https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#data-format-desc
//!
enum class TensorFormat : int32_t
{
    //! Memory layout is similar to an array in C or C++.
    //! The stride of each dimension is the product of the dimensions after it.
    //! The last dimension has unit stride.
    //!
    //! For DLA usage, the tensor sizes are limited to C,H,W in the range [1,8192].
    kLINEAR = 0,

    //! Vector-major format with two scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires FP16 or BF16 and at least three dimensions.
    kCHW2 = 1,

    //! Vector-minor format with eight scalars per vector.
    //! Vector dimension is third to last.
    //! This format requires FP16 or BF16 and at least three dimensions.
    kHWC8 = 2,

    //! Vector-major format with four scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires INT8 or FP16 and at least three dimensions.
    //! For INT8, the length of the vector dimension must be a build-time constant.
    //!
    //! Deprecated usage:
    //!
    //! If running on the DLA, this format can be used for acceleration
    //! with the caveat that C must be less than or equal to 4.
    //! If used as DLA input and the build option kGPU_FALLBACK is not specified,
    //! it needs to meet line stride requirement of DLA format. Column stride in
    //! bytes must be a multiple of 64 on Orin.
    kCHW4 = 3,

    //! Vector-major format with 16 scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires INT8 or FP16 and at least three dimensions.
    //!
    //! For DLA usage, this format maps to the native feature format for FP16,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    kCHW16 = 4,

    //! Vector-major format with 32 scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This format requires at least three dimensions.
    //!
    //! For DLA usage, this format maps to the native feature format for INT8,
    //! and the tensor sizes are limited to C,H,W in the range [1,8192].
    kCHW32 = 5,

    //! Vector-minor format with eight scalars per vector.
    //! Vector dimension is fourth to last.
    //!
    //! This format requires FP16 or BF16 and at least four dimensions.
    kDHWC8 = 6,

    //! Vector-major format with 32 scalars per vector.
    //! Vector dimension is fourth to last.
    //!
    //! This format requires FP16 or INT8 and at least four dimensions.
    kCDHW32 = 7,

    //! Vector-minor format where channel dimension is third to last and unpadded.
    //!
    //! This format requires either FP32 or UINT8 and at least three dimensions.
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
    //! to 64 bytes on Orin. C can only be 1, 3 or 4.
    //! If C == 1, it will map to grayscale format.
    //! If C == 3 or C == 4, it will map to color image format. And if C == 3,
    //! the stride for stepping along the W axis needs to be padded to 4 in elements.
    //!
    //! When C is {1, 3, 4}, then C' is {1, 4, 4} respectively,
    //! the memory layout is equivalent to a C array with dimensions
    //! [N][H][roundUp(W, 64/C'/elementSize)][C'] on Orin
    //! where elementSize is 2 for FP16
    //! and 1 for Int8. The tensor coordinates (n, c, h, w) mapping to array
    //! subscript [n][h][w][c].
    kDLA_HWC4 = 10,

    //! Vector-minor format with 16 scalars per vector.
    //! Vector dimension is third to last.
    //!
    //! This requires FP16 and at least three dimensions.
    kHWC16 = 11,

    //! Vector-minor format with one scalar per vector.
    //! Vector dimension is fourth to last.
    //!
    //! This format requires FP32 and at least four dimensions.
    kDHWC = 12
};

using InterfaceKind = char const*;

//!
//! \class InterfaceInfo
//!
//! \brief Version information associated with a TRT interface
//!
class InterfaceInfo
{
public:
    InterfaceKind kind;
    int32_t major;
    int32_t minor;
};

//!
//! \enum APILanguage
//!
//! \brief Programming language used in the implementation of a TRT interface
//!
enum class APILanguage : int32_t
{
    kCPP = 0,
    kPYTHON = 1
};

namespace impl
{
//! Maximum number of elements in APILanguage enum. \see APILanguage
template <>
struct EnumMaxImpl<APILanguage>
{
    //! Declaration of kVALUE that represents the maximum number of elements in the APILanguage enum.
    static constexpr int32_t kVALUE = 2;
};
} // namespace impl

//!
//! \class IVersionedInterface
//!
//! \brief An Interface class for version control.
//!
class IVersionedInterface
{
public:
    //!
    //! \brief The language used to build the implementation of this Interface.
    //!
    //! Applications must not override this method.
    //!
    virtual APILanguage getAPILanguage() const noexcept
    {
        return APILanguage::kCPP;
    }

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    virtual InterfaceInfo getInterfaceInfo() const noexcept = 0;

    virtual ~IVersionedInterface() noexcept = default;

protected:
    IVersionedInterface() = default;
    IVersionedInterface(IVersionedInterface const&) = default;
    IVersionedInterface(IVersionedInterface&&) = default;
    IVersionedInterface& operator=(IVersionedInterface const&) & = default;
    IVersionedInterface& operator=(IVersionedInterface&&) & = default;
};

namespace impl
{
//! Maximum number of elements in TensorFormat enum. \see TensorFormat
template <>
struct EnumMaxImpl<TensorFormat>
{
    //! Declaration of kVALUE that represents the maximum number of elements in the TensorFormat enum.
    static constexpr int32_t kVALUE = 13;
};
} // namespace impl


//!
//! \enum AllocatorFlag
//!
//! \brief Allowed type of memory allocation.
//!
enum class AllocatorFlag : int32_t
{
    //! TensorRT may call realloc() on this allocation.
    kRESIZABLE = 0,
};

namespace impl
{
//! Maximum number of elements in AllocatorFlag enum. \see AllocatorFlag
template <>
struct EnumMaxImpl<AllocatorFlag>
{
    //! Declaration of kVALUE that represents the maximum number of elements in the AllocatorFlag enum.
    static constexpr int32_t kVALUE = 1;
};
} // namespace impl

using AllocatorFlags = uint32_t;

//! DO NOT REFER TO namespace v_1_0 IN CODE. ALWAYS USE nvinfer1 INSTEAD.
//! The name v_1_0 may change in future versions of TensoRT.
namespace v_1_0
{

class IGpuAllocator : public IVersionedInterface
{
public:
    //!
    //! \brief A thread-safe callback implemented by the application to handle acquisition of GPU memory.
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
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    //! \deprecated Deprecated in TensorRT 10.0. Superseded by allocateAsync
    //!
    TRT_DEPRECATED virtual void* allocate(
        uint64_t const size, uint64_t const alignment, AllocatorFlags const flags) noexcept = 0;

    ~IGpuAllocator() override = default;
    IGpuAllocator() = default;

    //!
    //! \brief A thread-safe callback implemented by the application to resize an existing allocation.
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
    //!
    //! \return The address of the reallocated memory, or nullptr. If a non-null address is returned, it is
    //!         guaranteed to have the specified alignment.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual void* reallocate(void* const /*baseAddr*/, uint64_t /*alignment*/, uint64_t /*newSize*/) noexcept
    {
        return nullptr;
    }

    //!
    //! \brief A thread-safe callback implemented by the application to handle release of GPU memory.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory A memory address that was previously returned by an allocate() or reallocate() call of the same
    //! allocator object.
    //!
    //! \return True if the acquired memory is released successfully.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //! \deprecated Deprecated in TensorRT 10.0. Superseded by deallocateAsync
    //!
    TRT_DEPRECATED virtual bool deallocate(void* const memory) noexcept = 0;

    //!
    //! \brief A thread-safe callback implemented by the application to handle stream-ordered acquisition of GPU memory.
    //!
    //! The default behavior is to call method allocate(), which is synchronous and thus loses
    //! any performance benefits of asynchronous allocation. If you want the benefits of asynchronous
    //! allocation, see discussion of IGpuAsyncAllocator vs. IGpuAllocator in the documentation
    //! for nvinfer1::IGpuAllocator.
    //!
    //! \param size The size of the memory block required (in bytes).
    //! \param alignment The required alignment of memory. Alignment will be zero
    //!        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
    //!        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
    //!        An alignment value of zero indicates any alignment is acceptable.
    //! \param flags Reserved for future use. In the current release, 0 will be passed.
    //! \param stream specifies the cudaStream for asynchronous usage.
    //!
    //! \return If the allocation was successful, the start address of a device memory block of the requested size.
    //! If an allocation request of size 0 is made, nullptr must be returned.
    //! If an allocation request cannot be satisfied, nullptr must be returned.
    //! If a non-null address is returned, it is guaranteed to have the specified alignment.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual void* allocateAsync(
        uint64_t const size, uint64_t const alignment, AllocatorFlags const flags, cudaStream_t /*stream*/) noexcept
    {
        return allocate(size, alignment, flags);
    }
    //!
    //! \brief A thread-safe callback implemented by the application to handle stream-ordered release of GPU memory.
    //!
    //! The default behavior is to call method deallocate(), which is synchronous and thus loses
    //! any performance benefits of asynchronous deallocation. If you want the benefits of asynchronous
    //! deallocation, see discussion of IGpuAsyncAllocator vs. IGpuAllocator in the documentation
    //! for nvinfer1::IGpuAllocator.
    //!
    //! TensorRT may pass a nullptr to this function if it was previously returned by allocate().
    //!
    //! \param memory A memory address that was previously returned by an allocate() or reallocate() call of the same
    //! allocator object.
    //! \param stream specifies the cudaStream for asynchronous usage.
    //!
    //! \return True if the acquired memory is released successfully.
    //!
    //! \note The implementation must guarantee thread safety for concurrent allocate/reallocate/deallocate
    //! requests.
    //!
    //! \note The implementation is not required to be asynchronous. It is permitted to synchronize,
    //! albeit doing so will lose the performance advantage of asynchronous deallocation.
    //! Either way, it is critical that it not actually free the memory until the current
    //! stream position is reached.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual bool deallocateAsync(void* const memory, cudaStream_t /*stream*/) noexcept
    {
        return deallocate(memory);
    }

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return {"IGpuAllocator", 1, 0};
    }

protected:
    // @cond SuppressDoxyWarnings
    IGpuAllocator(IGpuAllocator const&) = default;
    IGpuAllocator(IGpuAllocator&&) = default;
    IGpuAllocator& operator=(IGpuAllocator const&) & = default;
    IGpuAllocator& operator=(IGpuAllocator&&) & = default;
    // @endcond
};

} // namespace v_1_0

//!
//! \class IGpuAllocator
//!
//! \brief Application-implemented class for controlling allocation on the GPU.
//!
//! \warning The lifetime of an IGpuAllocator object must exceed that of all objects that use it.
//!
//! This class is intended as a base class for allocators that implement synchronous allocation.
//! If you want the benefits of asynchronous allocation, you can do either of:
//!
//! * Derive your class from IGpuAllocator and override all four of its virtual methods
//!   for allocation/deallocation, including the two deprecated methods.
//!
//! * Derive your class from IGpuAsyncAllocator and override its two pure virtual
//!   methods for allocation/deallocation.
//!
//! The latter style is preferred because it does not tie code to deprecated methods.
//!
//! \see IGpuAsyncAllocator.
//!
using IGpuAllocator = v_1_0::IGpuAllocator;

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
    //! \brief The severity corresponding to a log message.
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
    //! \brief A callback implemented by the application to handle logging messages;
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
    //! Declaration of kVALUE that represents the maximum number of elements in the ILogger::Severity enum.
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
    //! system. An example is running an unsafe layer in a safety certified context, or a resource requirement for the
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

namespace v_1_0
{
class IErrorRecorder : public IVersionedInterface
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IErrorRecorder", 1, 0};
    }

    //!
    //! \brief A typedef of a C-style string for reporting error descriptions.
    //!
    using ErrorDesc = char const*;

    //!
    //! \brief The length limit for an error description in bytes, excluding the '\0' string terminator.
    //!        Only applicable to safe runtime.
    //!        General error recorder implementation can use any size appropriate for the use case.
    //!
    static constexpr size_t kMAX_DESC_LENGTH{127U};

    //!
    //! \brief A typedef of a 32-bit integer for reference counting.
    //!
    using RefCount = int32_t;

    IErrorRecorder() = default;
    ~IErrorRecorder() noexcept override = default;

    // Public API used to retrieve information from the error recorder.

    //!
    //! \brief Return the number of errors
    //!
    //! Determines the number of errors that occurred between the current point in execution
    //! and the last time that the clear() was executed. Due to the possibility of asynchronous
    //! errors occurring, a TensorRT API can return correct results, but still register errors
    //! with the Error Recorder. The value of getNbErrors() must increment by 1 after each reportError()
    //! call until clear() is called, or the maximum number of errors that can be stored is exceeded.
    //!
    //! \return Returns the number of errors detected, or 0 if there are no errors.
    //!         If the upper bound of errors that can be stored is exceeded, the upper bound value must
    //!         be returned.
    //!
    //! For example, if the error recorder can store up to 16 error descriptions but reportError() has
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
    //! \param val  The error code enum that is being reported.
    //! \param desc The string description of the error, which will be a NULL-terminated string.
    //!             For safety use cases its length is limited to kMAX_DESC_LENGTH bytes
    //!             (excluding the NULL terminator) and descriptions that exceed this limit will be silently truncated.
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
} // namespace v_1_0

//!
//! \class IErrorRecorder
//!
//! \brief Reference counted application-implemented error reporting interface for TensorRT objects.
//!
//! The error reporting mechanism is a user-defined object that interacts with the internal state of the object
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
using IErrorRecorder = v_1_0::IErrorRecorder;

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

namespace v_1_0
{
class IStreamReader : public IVersionedInterface
{
public:
    //!
    //! TensorRT never calls the destructor for an IStreamReader defined by the
    //! application.
    //!
    ~IStreamReader() override = default;
    IStreamReader() = default;

    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IStreamReader", 1, 0};
    }

    //!
    //! \brief Read the next number of bytes in the stream.
    //!
    //! \param destination The memory to write to
    //! \param nbBytes The number of bytes to read
    //!
    //! \returns The number of bytes read. Negative values will be considered an automatic error.
    //!
    virtual int64_t read(void* destination, int64_t nbBytes) = 0;

protected:
    IStreamReader(IStreamReader const&) = default;
    IStreamReader(IStreamReader&&) = default;
    IStreamReader& operator=(IStreamReader const&) & = default;
    IStreamReader& operator=(IStreamReader&&) & = default;
};
} // namespace v_1_0

//!
//! \class IStreamReader
//!
//! \brief Application-implemented class for reading data in a stream-based manner.
//!
//! \note To ensure compatibility of source code with future versions of TensorRT, use IStreamReader, not
//!       v_1_0::IStreamReader
//!
using IStreamReader = v_1_0::IStreamReader;

namespace v_1_0
{

class IPluginResource : public IVersionedInterface
{
public:
    //!
    //! \brief Return version information associated with this interface. Applications must not override this method.
    //!
    InterfaceInfo getInterfaceInfo() const noexcept override
    {
        return InterfaceInfo{"IPluginResource", 1, 0};
    }
    //!
    //! \brief Free the underlying resource
    //!
    //! This will only be called for IPluginResource objects that were produced from IPluginResource::clone()
    //!
    //! The IPluginResource object on which release() is called must still be in a clone-able state
    //! after release() returns
    //!
    //! \return 0 for success, else non-zero
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No; this method is not required to be thread-safe
    //!
    virtual int32_t release() noexcept = 0;

    //!
    //! \brief Clone the resource object
    //!
    //! \note Resource initialization (if any) may be skipped for non-cloned objects since only clones will be
    //! registered by TensorRT
    //!
    //! \return Pointer to cloned object. nullptr if there was an issue.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes; this method is required to be thread-safe and may be called from multiple threads.
    //!
    virtual IPluginResource* clone() noexcept = 0;

    ~IPluginResource() noexcept override = default;

    IPluginResource() = default;
    IPluginResource(IPluginResource const&) = default;
    IPluginResource(IPluginResource&&) = default;
    IPluginResource& operator=(IPluginResource const&) & = default;
    IPluginResource& operator=(IPluginResource&&) & = default;
}; // class IPluginResource
} // namespace v_1_0

//!
//! \class IPluginResource
//!
//! \brief Interface for plugins to define custom resources that could be shared through the plugin registry
//!
//! \see IPluginRegistry::acquirePluginResource
//! \see IPluginRegistry::releasePluginResource
//!
using IPluginResource = v_1_0::IPluginResource;

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
//! The format is as for TENSORRT_VERSION: (MAJOR * 100 + MINOR) * 100 + PATCH
//!
extern "C" TENSORRTAPI int32_t getInferLibVersion() noexcept;

#endif // NV_INFER_RUNTIME_BASE_H
