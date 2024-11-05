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
//! * NvInferPluginUtils.h (for plugin utilities)
//!
#if !defined(NV_INFER_INTERNAL_INCLUDE)
static_assert(false, "Do not directly include this file. Include NvInferRuntime.h or NvInferPluginUtils.h");
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
