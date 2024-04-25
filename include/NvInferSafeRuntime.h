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

#ifndef NV_INFER_SAFE_RUNTIME_H
#define NV_INFER_SAFE_RUNTIME_H

#define NV_INFER_INTERNAL_INCLUDE_RUNTIME_BASE 1
#include "NvInferRuntimeBase.h"
#undef NV_INFER_INTERNAL_INCLUDE_RUNTIME_BASE
#include "NvInferRuntimePlugin.h"
#include <cstddef>
#include <cstdint>

//!
//! \file NvInferSafeRuntime.h
//!
//! The functionality in this file is only supported in NVIDIA Drive(R) products.

//!
//! \namespace nvinfer1
//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{
//!
//! \namespace nvinfer1::safe
//!
//! \brief The safety subset of TensorRT's API version 1 namespace.
//!
namespace safe
{
//! Forward declare safe::ICudaEngine for use in other interfaces.
class ICudaEngine;
//! Forward declare safe::IExecutionContext for use in other interfaces.
class IExecutionContext;

//!
//! \class IRuntime
//!
//! \brief Allows a serialized functionally safe engine to be deserialized.
//!
//! \warning In the safety runtime the application is required to set the error reporter for correct error handling.
//! \see setErrorRecorder()
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRuntime
{
public:
    //!
    //! \brief Deserialize an engine from a byte array.
    //!
    //! If the serialized engine requires plugins the plugin creator must be registered by calling
    //! IPluginRegistry::registerCreator() before calling deserializeCudaEngine().
    //!
    //! \param blob The memory that holds the serialized engine. The content must be a copy of
    //! the result of calling IHostMemory::data() on a serialized plan that was created via calling
    //! IBuilder::buildSerializedNetwork() on a network within the supported safety scope.
    //! Additionally, it must have been validated via IConsistencyChecker::validate().
    //!
    //! \param size The size of the memory in bytes. This must be the result of calling IHostMemory::size()
    //! on the same IHostMemory object that is associated with the blob parameter.
    //!
    //! \return The engine, or nullptr if it could not be deserialized.
    //!
    //! \see IPluginRegistry::registerCreator()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes, if called from different instances of safe::IRuntime. Calling deserializeCudaEngine
    //!                  of the same safety runtime from multiple threads is not guaranteed to be thread safe.
    //!
    virtual ICudaEngine* deserializeCudaEngine(void const* const blob, std::size_t const size) noexcept = 0;

    //!
    //! \brief Set the GPU allocator.
    //!
    //! \param allocator The GPU allocator to be used by the runtime. All GPU memory acquired will use this
    //! allocator. If nullptr is passed, the default allocator will be used, which calls cudaMalloc and cudaFree.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual void setGpuAllocator(IGpuAllocator* const allocator) noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface.
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. If the recorder is set to
    //! nullptr, an error code of ErrorCode::kINVALID_ARGUMENT will be emitted if the recorder has already been
    //! registered, or ILogger::Severity::kERROR will be logged if the recorder has not yet been registered.
    //!
    //! \param recorder The error recorder to register with this interface, or nullptr to deregister the current
    //! error recorder.
    //!
    //! \see getErrorRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual void setErrorRecorder(IErrorRecorder* const recorder) noexcept = 0;

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called or a previously assigned error recorder
    //! has been deregistered.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered, or nullptr if no error recorder is set.
    //!
    //! \see setErrorRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

    IRuntime() = default;
    virtual ~IRuntime() noexcept = default;
    IRuntime(IRuntime const&) = delete;
    IRuntime(IRuntime&&) = delete;
    IRuntime& operator=(IRuntime const&) & = delete;
    IRuntime& operator=(IRuntime&&) & = delete;
};

//!
//! \class ICudaEngine
//!
//! \brief A functionally safe engine for executing inference on a built network.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ICudaEngine
{
public:
    //!
    //! \brief Create an execution context.
    //!
    //! \see safe::IExecutionContext.
    //!
    //! \return An execution context object if it can be constructed, or nullptr if the construction fails.
    //!
    //! \details Reasons for failure may include but not be limited to:
    //! - Heap memory exhaustion
    //! - Device memory exhaustion
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes; if createExecutionContext fails, users must treat this as a critical
    //!                  error and not perform any subsequent TensorRT operations apart from outputting
    //!                  the error logs.
    //!
    virtual IExecutionContext* createExecutionContext() noexcept = 0;

    //!
    //! \brief Create an execution context without any device memory allocated.
    //!
    //! The memory for execution of this device context must be supplied by the application by calling
    //! safe::IExecutionContext::setDeviceMemory().
    //!
    //! \see getDeviceMemorySize() safe::IExecutionContext::setDeviceMemory()
    //!
    //! \return An execution context object if it can be constructed, or nullptr if the construction fails.
    //!
    //! \details Reasons for failure may include but not be limited to heap memory exhaustion.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes; if createExecutionContext fails, users must treat this as a critical
    //!                  error and not perform any subsequent TensorRT operations apart from outputting
    //!                  the error logs.
    //!
    virtual IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept = 0;

    //!
    //! \brief Return the amount of device memory required by an execution context.
    //!
    //! \see safe::IExecutionContext::setDeviceMemory()
    //!
    //! \return Size of a contiguous memory buffer (in bytes) that users need to provide to
    //! safe::IExecutionContext::setDeviceMemory() if the execution context has been created by calling
    //! createExecutionContextWithoutDeviceMemory().
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual size_t getDeviceMemorySize() const noexcept = 0;

    //!
    //! \brief Returns the name of the network associated with the engine.
    //!
    //! The name is set during network creation and is retrieved after
    //! building or deserialization.
    //!
    //! \see INetworkDefinition::setName(), INetworkDefinition::getName()
    //!
    //! \return A NULL-terminated C-style string representing the name of the network, which will have a length of
    //! 1024 bytes or less including the NULL terminator.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual AsciiChar const* getName() const noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface.
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. If the recorder is set to
    //! nullptr, the error code ErrorCode::kINVALID_ARGUMENT will be emitted if the recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface, or nullptr to deregister the current.
    //! error recorder.
    //!
    //! \see getErrorRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual void setErrorRecorder(IErrorRecorder* const recorder) noexcept = 0;

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A
    //! nullptr will be returned if an error reporter has not been inherited
    //! from the IRuntime, and setErrorReporter() has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered, or nullptr if none has been
    //! registered.
    //!
    //! \see setErrorRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

    ICudaEngine() = default;
    virtual ~ICudaEngine() noexcept = default;
    ICudaEngine(ICudaEngine const&) = delete;
    ICudaEngine(ICudaEngine&&) = delete;
    ICudaEngine& operator=(ICudaEngine const&) & = delete;
    ICudaEngine& operator=(ICudaEngine&&) & = delete;

    //!
    //! \brief Get the extent of an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \return Extent of the tensor. The invalid value Dims{-1, {}} will be returned if
    //! - name is not the name of an input or output tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual Dims getTensorShape(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Determine the required data type for a buffer from its tensor name.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \return The type of the data in the buffer. The default value DataType::kFLOAT will be returned if
    //! - name is not the name of an input or output tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual DataType getTensorDataType(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Determine whether a tensor is an input or output tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \return kINPUT if tensorName is the name of an input tensor, kOUTPUT if tensorName is the name of an output
    //! tensor. The invalid value kNONE is returned if
    //! - tensorName exceeds the string length limit, or
    //! - tensorName is nullptr, or
    //! - tensorName does not correspond to any input or output tensor.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual TensorIOMode getTensorIOMode(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Return the size of the tensor data type in bytes for a vectorized tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \return The size of the tensor data type in bytes if the tensor is vectorized (4 for float and int32,
    //! 2 for half, 1 for int8). 0 will be returned if
    //! - name is not the name of an input or output tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit, or
    //! - the tensor of the given name is not vectorized.
    //!
    //! \see safe::ICudaEngine::getTensorVectorizedDim()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual std::int32_t getTensorBytesPerComponent(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Return the number of components included in one element for a vectorized tensor.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \return The vector length (in scalars) for a vectorized tensor, or 1 for a scalar tensor.
    //! The invalid value -1 will be returned if
    //! - name is not the name of an input or output tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit.
    //!
    //! \see safe::ICudaEngine::getTensorVectorizedDim()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual std::int32_t getTensorComponentsPerElement(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Return the tensor format.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \return The tensor format. TensorFormat::kLINEAR will be returned if
    //! - name is not the name of an input or output tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual TensorFormat getTensorFormat(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Return the dimension index along which the buffer is vectorized.
    //!
    //! Specifically, -1 is returned if the tensor is scalar.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less including the
    //! NULL terminator.
    //!
    //! \return The dimension index along which the buffer is vectorized. -1 will be returned if
    //! - name is not the name of an input or output tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit (1024 bytes or less including the NULL terminator), or
    //! - the tensor of given name is not vectorized.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual std::int32_t getTensorVectorizedDim(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Return the number of input and output tensors for the network from which the engine was built.
    //!
    //! \return The number of IO tensors.
    //!
    //! \see getIOTensorName()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual std::int32_t getNbIOTensors() const noexcept = 0;

    //!
    //! \brief Return the name of an IO tensor.
    //!
    //! If the index does not fall between 0 and getNbIOTensors()-1, the function will fail with an error code
    //! of ErrorCode::kINVALID_ARGUMENT(3) that is emitted to the registered IErrorRecorder.
    //!
    //! \param index The IO tensor index.
    //!
    //! \return The name of an IO tensor, which will be a NULL-terminated string of 1024 bytes or less (including the
    //! NULL terminator) if the index is in the range (between 0 and getNbIOTensors()-1). nullptr will be returned if
    //! the index is not in range.
    //!
    //! \see getNbIOTensors()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual AsciiChar const* getIOTensorName(std::int32_t const index) const noexcept = 0;
};

//!
//! \brief Space to record information about runtime errors.
//!
//! kNAN_CONSUMED errors occur when NAN values are stored in an INT8 quantized datatype.
//! kINF_CONSUMED errors occur when +-INF values are stored in an INT8 quantized datatype.
//! kGATHER_OOB errors occur when a gather index tensor contains a value that is outside of the data tensor.
//! kSCATTER_OOB and kSCATTER_RACE are reserved for future use.
//!
//! Mark the RuntimeErrorType that occurs during asynchronous kernel execution.
struct RuntimeErrorInformation
{
    //! Each bit represents a RuntimeErrorType that has occurred during kernel execution.
    uint64_t bitMask;
};

//!
//! \brief Enum to represent runtime error types.
//!
enum class RuntimeErrorType : uint64_t
{
    //! NaN floating-point value was silently consumed
    kNAN_CONSUMED = 1ULL << 0,
    //! Inf floating-point value was silently consumed
    kINF_CONSUMED = 1ULL << 1,
    //! Out-of-bounds access in gather operation
    kGATHER_OOB = 1ULL << 2,
    //! Out-of-bounds access in scatter operation
    kSCATTER_OOB = 1ULL << 3,
    //! Race condition in scatter operation
    kSCATTER_RACE = 1ULL << 4,
};

//!
//! \class IExecutionContext
//!
//! \brief Functionally safe context for executing inference using an engine.
//!
//! Multiple safe execution contexts may exist for one safe::ICudaEngine instance, allowing the same
//! engine to be used for the execution of multiple inputs simultaneously.
//!
//! \warning Do not call the APIs of the same IExecutionContext from multiple threads at any given time.
//! Each concurrent execution must have its own instance of an IExecutionContext.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IExecutionContext
{
public:
    //!
    //! \brief Get the associated engine.
    //!
    //! \see safe::ICudaEngine
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual ICudaEngine const& getEngine() const noexcept = 0;

    //!
    //! \brief Set the name of the execution context.
    //!
    //! This method copies the name string.
    //!
    //! \warning Strings passed to the runtime must be NULL terminated and have a length of 1024 bytes or less
    //! including the NULL terminator. Otherwise, the operation will not change the execution context name, and
    //! an error message will be recorded via the error recorder.
    //!
    //! \see getName()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual void setName(AsciiChar const* const name) noexcept = 0;

    //!
    //! \brief Return the name of the execution context.
    //!
    //! \return The name that was passed to setName(), as a NULL-terminated string of 1024 bytes or less including
    //! the NULL terminator. An empty string will be returned as the default value.
    //!
    //! \see setName()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual AsciiChar const* getName() const noexcept = 0;

    //!
    //! \brief Set the device memory for use by this execution context.
    //!
    //! \param memory The start address of a device memory buffer whose size in bytes must be at least the value
    //! returned by getEngine().getDeviceMemorySize().
    //!
    //! If using enqueueV2() to run the network, The memory is in use
    //! from the invocation of enqueueV2() until network execution is complete.
    //! Releasing or otherwise using the memory for other purposes during this time will result in undefined behavior.
    //!
    //! \warning Do not release or use for other purposes the memory set here during network execution.
    //!
    //! \warning If the execution context has been created by calling createExecutionContext(), this
    //! function must not be used and will fail with an error message if called.
    //!
    //! \see safe::ICudaEngine::getDeviceMemorySize() safe::ICudaEngine::createExecutionContextWithoutDeviceMemory()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual void setDeviceMemory(void* const memory) noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface.
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. If the recorder is set to
    //! nullptr, the error code ErrorCode::kINVALID_ARGUMENT will be emitted if the recorder has been registered. The
    //! lifetime of the error recorder object must exceed the lifetime of the execution context.
    //!
    //! \param recorder Either a pointer to a valid error recorder object to register with this interface,
    //!                 or nullptr to deregister the current recorder.
    //!
    //! \see getErrorRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual void setErrorRecorder(IErrorRecorder* const recorder) noexcept = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered, or nullptr if the error recorder
    //! has been deregistered or not set.
    //!
    //! \see setErrorRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;

    IExecutionContext() = default;
    virtual ~IExecutionContext() noexcept = default;
    IExecutionContext(IExecutionContext const&) = delete;
    IExecutionContext(IExecutionContext&&) = delete;
    IExecutionContext& operator=(IExecutionContext const&) & = delete;
    IExecutionContext& operator=(IExecutionContext&&) & = delete;

    //!
    //! \brief Set error buffer output for runtime errors.
    //!
    //! The error buffer output must be allocated in device memory and will be used for subsequent
    //! calls to enqueueV2() or enqueueV3(). Checking the contents of the error buffer after inference is the
    //! responsibility of the application. The pointer passed here must have alignment adequate for the
    //! RuntimeErrorInformation struct.
    //!
    //! \warning The buffer is written if reportable errors are encountered during network execution. Releasing the
    //! buffer before network execution is complete will result in undefined behavior. Accessing the memory before
    //! network execution is complete may not correctly capture the error state.
    //!
    //! \param buffer The device memory address of the runtime error information buffer.
    //!
    //! \see getErrorBuffer()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual void setErrorBuffer(RuntimeErrorInformation* const buffer) noexcept = 0;

    //!
    //! \brief Get error buffer output for runtime errors.
    //!
    //! \return Pointer to device memory to use as runtime error buffer or nullptr if not set.
    //!
    //! \see setErrorBuffer()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual RuntimeErrorInformation* getErrorBuffer() const noexcept = 0;

    //!
    //! \brief Return the strides of the buffer for the given tensor name.
    //!
    //! The strides are in units of elements, not components or bytes.
    //! Elements are vectors (for a vectorized format) or scalars (for a scalar format).
    //! For example, for TensorFormat::kHWC8, a stride of one spans 8 scalars.
    //!
    //! \param tensorName The name of an input or output tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less
    //! including the NULL terminator.
    //!
    //! \return The strides of the buffer for the given tensor name. Dims{-1, {}} will be returned if
    //! - name is not the name of an input or output tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual Dims getTensorStrides(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Set memory address for given input tensor.
    //!
    //! An address defaults to nullptr.
    //!
    //! Before calling enqueueV3(), each input must have a non-null address.
    //!
    //! \param tensorName The name of an input tensor.
    //! \param data The pointer (void const*) to the input tensor data, which is device memory owned by the user.
    //! Users are responsible for ensuring that the buffer size has at least the expected length, which is
    //! the product of the tensor dimensions (with the vectorized dimension padded to a multiple of the vector length)
    //! times the data type size.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less
    //! including the NULL terminator.
    //!
    //! \warning The data pointer must have 256-byte alignment.
    //!
    //! \return True on success, false if
    //! - name is not the name of an input tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit, or
    //! - pointer to the const data is nullptr or not correctly aligned.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual bool setInputTensorAddress(AsciiChar const* const tensorName, void const* const data) noexcept = 0;

    //!
    //! \brief Set memory address for given output tensor.
    //!
    //! An address defaults to nullptr.
    //!
    //! Before calling enqueueV3(), each output must have a non-null address.
    //!
    //! \param tensorName The name of an output tensor.
    //! \param data The pointer (void*) to the output tensor data, which is device memory owned by the user.
    //! Users are responsible for ensuring that the buffer size has at least the expected length, which is
    //! the product of the tensor dimensions (with the vectorized dimension padded to a multiple of the vector length)
    //! times the data type size.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less
    //! including the NULL terminator.
    //! \warning The data pointer must have 256-byte alignment.
    //!
    //! \return True on success. Return false if
    //! - name is not the name of an output tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit, or
    //! - pointer to data is nullptr or not aligned.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual bool setOutputTensorAddress(AsciiChar const* const tensorName, void* const data) noexcept = 0;

    //!
    //! \brief Set the event to mark inputs as consumed.
    //!
    //! Passing event==nullptr removes whatever event was set, if any.
    //!
    //! \param event The CUDA event that is signaled after all input tensors have been consumed, or nullptr to remove
    //!        an event that was previously set.
    //!
    //! \return True on success, false if an error occurred.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual bool setInputConsumedEvent(cudaEvent_t const event) noexcept = 0;

    //!
    //! \brief Return the event associated with consuming the input.
    //!
    //! \return The CUDA event that was passed to setInputConsumedEvent(). nullptr will be returned if the event is
    //! not set.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual cudaEvent_t getInputConsumedEvent() const noexcept = 0;

    //!
    //! \brief Get memory address for given input tensor.
    //!
    //! \param tensorName The name of an input tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less
    //! including the NULL terminator.
    //!
    //! \return The device memory address for the given input tensor. nullptr will be returned if
    //! - name is not the name of an input tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit, or
    //! - the memory address for the given input tensor is not set.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual void const* getInputTensorAddress(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Get memory address for given output tensor.
    //!
    //! \param tensorName The name of an output tensor.
    //!
    //! \warning The string tensorName must be NULL terminated and have a length of 1024 bytes or less
    //! including the NULL terminator.
    //!
    //! \return The device memory address for the given output tensor. Return nullptr if
    //! - name is not the name of an output tensor, or
    //! - name is nullptr, or
    //! - name exceeds the string length limit, or
    //! - the memory address for the given output tensor is not set.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual void* getOutputTensorAddress(AsciiChar const* const tensorName) const noexcept = 0;

    //!
    //! \brief Enqueue inference on a stream.
    //!
    //! Modifying or releasing memory that has been registered for the tensors before stream
    //! synchronization or the event passed to setInputConsumedEvent has been signaled results in undefined
    //! behavior.
    //!
    //! \param stream A CUDA stream on which the inference kernels will be enqueued.
    //!
    //! \return True on success, false if any execution error occurred.
    //! Errors may include but not be limited to:
    //! - Internal errors during executing one engine layer
    //! - CUDA errors
    //! - Some input or output tensor addresses have not been set.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual bool enqueueV3(cudaStream_t const stream) noexcept = 0;
};

//!
//! \class IPluginRegistry
//!
//! \brief Single registration point for all plugins in an application. It is
//! used to find plugin implementations during engine deserialization.
//! Internally, the plugin registry is considered to be a singleton so all
//! plugins in an application are part of the same global registry.
//! Note that the plugin registry is only supported for plugins of type
//! IPluginV2 and must also have a corresponding IPluginCreator implementation.
//!
//! \see IPluginV2 and IPluginCreator
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
//! \warning IPluginRegistry::setErrorRecorder() must be called to register
//! an error recorder with the registry before using other methods in the registry.
//!

class IPluginRegistry
{
public:
    //!
    //! \brief Register a plugin creator.
    //!
    //! \param creator The plugin creator to be registered.
    //!
    //! \param pluginNamespace A NULL-terminated namespace string, which must be 1024 bytes or less including the NULL
    //! terminator. It must be identical with the result of calling
    //! IPluginCreator::getPluginNamespace() on the creator object.
    //!
    //! \return True if the registration succeeded, else false.
    //!
    //! \details Registration may fail for any of the following reasons:
    //! - The pluginNamespace string is a nullptr.
    //! - The pluginNamespace string exceeds the maximum length.
    //! - The pluginNamespace string does not match the result of creator.getPluginNamespace().
    //! - There have already been 100 plugin creators registered (maximum number of plugins exceeded).
    //! - Another plugin creator with the same combination of plugin name, version and namespace has already been
    //!   registered.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes; calls to this method will be synchronized by a mutex.
    //!
    virtual bool registerCreator(IPluginCreator& creator, AsciiChar const* const pluginNamespace) noexcept = 0;

    //!
    //! \brief Return all the registered plugin creators and the number of
    //! registered plugin creators. Returns nullptr if none is found.
    //!
    //! \param[out] numCreators If the call completes successfully, the number of registered plugin creators (which
    //!                         will be an integer between 0 and 100 inclusive)
    //! \return The start address of an IPluginCreator* array of length numCreators if at least one plugin creator
    //!         has been registered, or nullptr if there are no registered plugin creators.
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
    //! \warning The strings pluginName, pluginVersion, and pluginNamespace must be NULL terminated and have a length
    //! of 1024 bytes or less including the NULL terminator.
    //!
    //! \param pluginName The plugin name string
    //! \param pluginVersion The plugin version string
    //! \param pluginNamespace The plugin namespace (by default empty string)
    //!
    //! \return If a plugin creator corresponding to the passed name, version and namespace can be found in the
    //!         registry, it is returned. nullptr is returned in the following situations:
    //!         - Any of the input arguments is nullptr.
    //!         - Any of the input arguments exceeds the string length limit.
    //!         - No plugin creator corresponding to the input arguments can be found in the registry.
    //!         - A plugin creator can be found, but its stored namespace attribute does not match the pluginNamespace.
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: Yes
    //!
    virtual IPluginCreator* getPluginCreator(AsciiChar const* const pluginName, AsciiChar const* const pluginVersion,
        AsciiChar const* const pluginNamespace = "") noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. If the recorder is set to
    //! nullptr, the error code ErrorCode::kINVALID_ARGUMENT will be emitted if the recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface, or nullptr to deregister the current
    //!                 recorder.
    //!
    //! \see getErrorRecorder()
    //!
    //! \usage
    //! - Allowed context for the API call
    //!   - Thread-safe: No
    //!
    virtual void setErrorRecorder(IErrorRecorder* const recorder) noexcept = 0;

    //!
    //! \brief Get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called, or an ErrorRecorder has not been
    //! inherited.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered, or nullptr if:
    //!         - no error recorder has been set, or
    //!         - the last error recorder has been deregistered via setErrorRecorder(nullptr).
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
    //! \param creator The plugin creator to deregister.
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

    // @cond SuppressDoxyWarnings
    IPluginRegistry() = default;
    IPluginRegistry(IPluginRegistry const&) = delete;
    IPluginRegistry(IPluginRegistry&&) = delete;
    IPluginRegistry& operator=(IPluginRegistry const&) & = delete;
    IPluginRegistry& operator=(IPluginRegistry&&) & = delete;
    // @endcond

protected:
    virtual ~IPluginRegistry() noexcept = default;
};

//!
//! \brief Create an instance of a safe::IRuntime class.
//!
//! \param logger A logger object whose lifetime must exceed that of the returned runtime.
//! Loggers must be thread-safe.
//!
//! \return A safe runtime object that can be used for safe plan file deserialization.
//!
//! This class is the logging class for the runtime.
//!
//! \usage
//! - Allowed context for the API call
//!   - Thread-safe: Yes
//!
IRuntime* createInferRuntime(ILogger& logger) noexcept;

//!
//! \brief Return the safe plugin registry
//!
//! \usage
//! - Allowed context for the API call
//!   - Thread-safe: Yes
//!
extern "C" TENSORRTAPI IPluginRegistry* getSafePluginRegistry() noexcept;

//!
//! \brief Register the plugin creator to the registry
//! The static registry object will be instantiated when the plugin library is
//! loaded. This static object will register all creators available in the
//! library to the registry.
//!
//! \warning Statically registering plugins must be avoided in the automotive
//!  safety context as the application developer must first register an error recorder
//!  with the plugin registry via IPluginRegistry::setErrorRecorder() before using
//!  IPluginRegistry::registerCreator() or other methods.
//!
template <typename T>
class PluginRegistrar
{
public:
    PluginRegistrar()
    {
        getSafePluginRegistry()->registerCreator(instance, "");
    }

private:
    //! Plugin instance.
    T instance{};
};

} // namespace safe

} // namespace nvinfer1

#define REGISTER_SAFE_TENSORRT_PLUGIN(name)                                                                            \
    static nvinfer1::safe::PluginRegistrar<name> pluginRegistrar##name {}
#endif // NV_INFER_SAFE_RUNTIME_H
