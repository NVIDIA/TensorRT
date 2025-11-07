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

// This file contains all Core docstrings, since these are typically too long to keep in the binding code.
#pragma once

namespace tensorrt
{

namespace ILoggerDoc
{
constexpr char const* descr = R"trtdoc(
Abstract base Logger class for the :class:`Builder`, :class:`ICudaEngine` and :class:`Runtime` .

To implement a custom logger, ensure that you explicitly instantiate the base class in :func:`__init__` :
::

    class MyLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)

        def log(self, severity, msg):
            ... # Your implementation here


:arg min_severity: The initial minimum severity of this Logger.

:ivar min_severity: :class:`Logger.Severity` This minimum required severity of messages for the logger to log them.

The logger used to create an instance of IBuilder, IRuntime or IRefitter is used for all objects created through that interface.
The logger should be valid until all objects created are released.
)trtdoc";

constexpr char const* log = R"trtdoc(
Logs a message to `stderr` . This function must be overriden by a derived class.

:arg severity: The severity of the message.
:arg msg: The log message.

)trtdoc";
} // namespace ILoggerDoc

namespace LoggerDoc
{
constexpr char const* descr = R"trtdoc(
Logger for the :class:`Builder`, :class:`ICudaEngine` and :class:`Runtime` .

:arg min_severity: The initial minimum severity of this Logger.

:ivar min_severity: :class:`Logger.Severity` This minimum required severity of messages for the logger to log them.

)trtdoc";

constexpr char const* log = R"trtdoc(
Logs a message.

:arg severity: The severity of the message.
:arg msg: The log message.
)trtdoc";
} // namespace LoggerDoc

namespace DefaultLoggerDoc
{
constexpr char const* log = R"trtdoc(
Logs a message to `stdout`.

:arg severity: The severity of the message.
:arg msg: The log message.
)trtdoc";

}

namespace SeverityDoc
{
constexpr char const* descr = R"trtdoc(
    Indicates the severity of a message. The values in this enum are also accessible in the :class:`ILogger` directly.
    For example, ``tensorrt.ILogger.INFO`` corresponds to ``tensorrt.ILogger.Severity.INFO`` .
)trtdoc";

constexpr char const* internal_error = R"trtdoc(
    Represents an internal error. Execution is unrecoverable.
)trtdoc";

constexpr char const* error = R"trtdoc(
    Represents an application error.
)trtdoc";

constexpr char const* warning = R"trtdoc(
    Represents an application error that TensorRT has recovered from or fallen back to a default.
)trtdoc";

constexpr char const* info = R"trtdoc(
    Represents informational messages.
)trtdoc";

constexpr char const* verbose = R"trtdoc(
    Verbose messages with debugging information.
)trtdoc";
} // namespace SeverityDoc

namespace IProfilerDoc
{
constexpr char const* descr = R"trtdoc(
    Abstract base Profiler class.

    To implement a custom profiler, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::

        class MyProfiler(trt.IProfiler):
            def __init__(self):
                trt.IProfiler.__init__(self)

            def report_layer_time(self, layer_name, ms):
                ... # Your implementation here

    When this class is added to an :class:`IExecutionContext`, the profiler will be called once per layer for each invocation of :func:`IExecutionContext.execute_v2()`.

    It is not recommended to run inference with profiler enabled when the inference execution time is critical since the profiler may affect execution time negatively.
)trtdoc";

constexpr char const* report_layer_time = R"trtdoc(
    Reports time in milliseconds for each layer. This function must be overriden a derived class.

    :arg layer_name: The name of the layer, set when constructing the :class:`INetworkDefinition` . If the engine is built with profiling verbosity set to NONE, the layerName is the decimal index of the layer.
    :arg ms: The time in milliseconds to execute the layer.
)trtdoc";
} // namespace IProfilerDoc

namespace ProfilerDoc
{
constexpr char const* descr = R"trtdoc(
    When this class is added to an :class:`IExecutionContext`, the profiler will be called once per layer for each invocation of :func:`IExecutionContext.execute_v2()`.

    It is not recommended to run inference with profiler enabled when the inference execution time is critical since the profiler may affect execution time negatively.
)trtdoc";

constexpr char const* report_layer_time = R"trtdoc(
    Prints time in milliseconds for each layer to stdout.

    :arg layer_name: The name of the layer, set when constructing the :class:`INetworkDefinition` .
    :arg ms: The time in milliseconds to execute the layer.
)trtdoc";
} // namespace ProfilerDoc

namespace IOptimizationProfileDoc
{
constexpr char const* descr = R"trtdoc(
    Optimization profile for dynamic input dimensions and shape tensors.

    When building an :class:`ICudaEngine` from an :class:`INetworkDefinition` that has dynamically resizable inputs (at least
    one input tensor has one or more of its dimensions specified as -1) or shape input tensors, users need to specify
    at least one optimization profile. Optimization profiles are numbered 0, 1, ...

    The first optimization profile that has been defined (with index 0) will be used by the :class:`ICudaEngine` whenever no
    optimization profile has been selected explicitly. If none of the inputs are dynamic, the default optimization
    profile will be generated automatically unless it is explicitly provided by the user (this is possible but not
    required in this case). If more than a single optimization profile is defined, users may set a target how
    much additional weight space should be maximally allocated to each additional profile (as a fraction of the
    maximum, unconstrained memory).

    Users set optimum input tensor dimensions, as well as minimum and maximum input tensor dimensions. The builder
    selects the kernels that result in the lowest runtime for the optimum input tensor dimensions, and are valid for
    all input tensor sizes in the valid range between minimum and maximum dimensions. A runtime error will be raised
    if the input tensor dimensions fall outside the valid range for this profile. Likewise, users provide minimum,
    optimum, and maximum values for all shape tensor input values.

    :class:`IOptimizationProfile` implements :func:`__nonzero__` and :func:`__bool__` such that evaluating a profile as a :class:`bool` (e.g. ``if profile:``) will check whether the optimization profile can be passed to an IBuilderConfig object. This will perform partial validation, by e.g. checking that the maximum dimensions are at least as large as the optimum dimensions, and that the optimum dimensions are always as least as large as the minimum dimensions. Some validation steps require knowledge of the network definition and are deferred to engine build time.

    :ivar extra_memory_target: Additional memory that the builder should aim to maximally allocate for this profile, as a fraction of the memory it would use if the user did not impose any constraints on memory. This unconstrained case is the default; it corresponds to ``extra_memory_target`` == 1.0. If ``extra_memory_target`` == 0.0, the builder aims to create the new optimization profile without allocating any additional weight memory. Valid inputs lie between 0.0 and 1.0. This parameter is only a hint, and TensorRT does not guarantee that the ``extra_memory_target`` will be reached. This parameter is ignored for the first (default) optimization profile that is defined.
)trtdoc";

constexpr char const* set_shape = R"trtdoc(
    Set the minimum/optimum/maximum dimensions for a dynamic input tensor.

    This function must be called for any network input tensor that has dynamic dimensions. If ``min``, ``opt``, and ``max`` are the minimum, optimum, and maximum dimensions, and ``real_shape`` is the shape for this input tensor provided to the :class:`INetworkDefinition` ,then the following conditions must hold:

    (1) ``len(min)`` == ``len(opt)`` == ``len(max)`` == ``len(real_shape)``
    (2) 0 <= ``min[i]`` <= ``opt[i]`` <= ``max[i]`` for all ``i``
    (3) if ``real_shape[i]`` != -1, then ``min[i]`` == ``opt[i]`` == ``max[i]`` == ``real_shape[i]``

    This function may (but need not be) called for an input tensor that does not have dynamic dimensions. In this
    case, all shapes must equal ``real_shape``.

    :arg input: The name of the input tensor.
    :arg min: The minimum dimensions for this input tensor.
    :arg opt: The optimum dimensions for this input tensor.
    :arg max: The maximum dimensions for this input tensor.

    :raises: :class:`ValueError` if an inconsistency was detected. Note that inputs can be validated only partially; a full validation is performed at engine build time.
)trtdoc";

constexpr char const* get_shape = R"trtdoc(
    Get the minimum/optimum/maximum dimensions for a dynamic input tensor.
    If the dimensions have not been previously set via :func:`set_shape`, return an invalid :class:`Dims` with a length of -1.

    :returns: A ``List[Dims]`` of length 3, containing the minimum, optimum, and maximum shapes, in that order. If the shapes have not been set yet, an empty list is returned.
)trtdoc";

constexpr char const* set_shape_input = R"trtdoc(
    Set the minimum/optimum/maximum values for a shape input tensor.

    This function must be called for every input tensor ``t`` that is a shape tensor (``t.is_shape`` == ``True``).
    This implies that the datatype of ``t`` is ``int64`` or ``int32``, the rank is either 0 or 1, and the dimensions of ``t``
    are fixed at network definition time. This function must NOT be called for any input tensor that is not a
    shape tensor.

    If ``min``, ``opt``, and ``max`` are the minimum, optimum, and maximum values, it must be true that ``min[i]`` <= ``opt[i]`` <= ``max[i]`` for
    all ``i``.

    :arg input: The name of the input tensor.
    :arg min: The minimum values for this shape tensor.
    :arg opt: The optimum values for this shape tensor.
    :arg max: The maximum values for this shape tensor.

    :raises: :class:`ValueError` if an inconsistency was detected. Note that inputs can be validated only partially; a full validation is performed at engine build time.
)trtdoc";

constexpr char const* get_shape_input = R"trtdoc(
    Get the minimum/optimum/maximum values for a shape input tensor.

    :returns: A ``List[List[int]]`` of length 3, containing the minimum, optimum, and maximum values, in that order. If the values have not been set yet, an empty list is returned.
)trtdoc";
} // namespace IOptimizationProfileDoc

namespace ErrorCodeDoc
{
constexpr char const* descr = R"trtdoc(Error codes that can be returned by TensorRT during execution.)trtdoc";

constexpr char const* SUCCESS = R"trtdoc(Execution completed successfully.)trtdoc";

constexpr char const* UNSPECIFIED_ERROR = R"trtdoc(
    An error that does not fall into any other category. This error is included for forward compatibility.
)trtdoc";

constexpr char const* INTERNAL_ERROR = R"trtdoc(A non-recoverable TensorRT error occurred.)trtdoc";

constexpr char const* INVALID_ARGUMENT = R"trtdoc(
    An argument passed to the function is invalid in isolation. This is a violation of the API contract.
)trtdoc";

constexpr char const* INVALID_CONFIG = R"trtdoc(
    An error occurred when comparing the state of an argument relative to other arguments. For example, the
    dimensions for concat differ between two tensors outside of the channel dimension. This error is triggered
    when an argument is correct in isolation, but not relative to other arguments. This is to help to distinguish
    from the simple errors from the more complex errors.
    This is a violation of the API contract.
)trtdoc";

constexpr char const* FAILED_ALLOCATION = R"trtdoc(
    An error occurred when performing an allocation of memory on the host or the device.
    A memory allocation error is normally fatal, but in the case where the application provided its own memory
    allocation routine, it is possible to increase the pool of available memory and resume execution.
)trtdoc";

constexpr char const* FAILED_INITIALIZATION = R"trtdoc(
    One, or more, of the components that TensorRT relies on did not initialize correctly.
    This is a system setup issue.
)trtdoc";

constexpr char const* FAILED_EXECUTION = R"trtdoc(
    An error occurred during execution that caused TensorRT to end prematurely, either an asynchronous error,
    user cancellation, or other execution errors reported by CUDA/DLA. In a dynamic system, the
    data can be thrown away and the next frame can be processed or execution can be retried.
    This is either an execution error or a memory error.
)trtdoc";

constexpr char const* FAILED_COMPUTATION = R"trtdoc(
    An error occurred during execution that caused the data to become corrupted, but execution finished. Examples
    of this error are NaN squashing or integer overflow. In a dynamic system, the data can be thrown away and the
    next frame can be processed or execution can be retried.
    This is either a data corruption error, an input error, or a range error.
)trtdoc";

constexpr char const* INVALID_STATE = R"trtdoc(
    TensorRT was put into a bad state by incorrect sequence of function calls. An example of an invalid state is
    specifying a layer to be DLA only without GPU fallback, and that layer is not supported by DLA. This can occur
    in situations where a service is optimistically executing networks for multiple different configurations
    without checking proper error configurations, and instead throwing away bad configurations caught by TensorRT.
    This is a violation of the API contract, but can be recoverable.

    Example of a recovery:
    GPU fallback is disabled and conv layer with large filter(63x63) is specified to run on DLA. This will fail due
    to DLA not supporting the large kernel size. This can be recovered by either turning on GPU fallback
    or setting the layer to run on the GPU.
)trtdoc";

constexpr char const* UNSUPPORTED_STATE = R"trtdoc(
    An error occurred due to the network not being supported on the device due to constraints of the hardware or
    system. An example is running a unsafe layer in a safety certified context, or a resource requirement for the
    current network is greater than the capabilities of the target device. The network is otherwise correct, but
    the network and hardware combination is problematic. This can be recoverable.
    Examples:
    * Scratch space requests larger than available device memory and can be recovered by increasing allowed workspace size.
    * Tensor size exceeds the maximum element count and can be recovered by reducing the maximum batch size.
)trtdoc";
} // namespace ErrorCodeDoc

namespace IErrorRecorderDoc
{
constexpr char const* descr = R"trtdoc(
    Reference counted application-implemented error reporting interface for TensorRT objects.

    The error reporting mechanism is a user defined object that interacts with the internal state of the object
    that it is assigned to in order to determine information about abnormalities in execution. The error recorder
    gets both an error enum that is more descriptive than pass/fail and also a description that gives more
    detail on the exact failure modes. In the safety context, the error strings are all limited to 128 characters
    in length.
    The ErrorRecorder gets passed along to any class that is created from another class that has an ErrorRecorder
    assigned to it. For example, assigning an ErrorRecorder to an Builder allows all INetwork's, ILayer's, and
    ITensor's to use the same error recorder. For functions that have their own ErrorRecorder accessor functions.
    This allows registering a different error recorder or de-registering of the error recorder for that specific
    object.

    The ErrorRecorder object implementation must be thread safe if the same ErrorRecorder is passed to different
    interface objects being executed in parallel in different threads. All locking and synchronization is
    pushed to the interface implementation and TensorRT does not hold any synchronization primitives when accessing
    the interface functions.
)trtdoc";

constexpr char const* has_overflowed = R"trtdoc(
    Determine if the error stack has overflowed.

    In the case when the number of errors is large, this function is used to query if one or more
    errors have been dropped due to lack of storage capacity. This is especially important in the
    automotive safety case where the internal error handling mechanisms cannot allocate memory.

    :returns: True if errors have been dropped due to overflowing the error stack.
)trtdoc";

constexpr char const* get_num_errors = R"trtdoc(
    Return the number of errors

    Determines the number of errors that occurred between the current point in execution
    and the last time that the clear() was executed. Due to the possibility of asynchronous
    errors occuring, a TensorRT API can return correct results, but still register errors
    with the Error Recorder. The value of getNbErrors must monotonically increases until clear()
    is called.

    :returns: Returns the number of errors detected, or 0 if there are no errors.
)trtdoc";

constexpr char const* get_error_code = R"trtdoc(
    Returns the ErrorCode enumeration.

    The error_idx specifies what error code from 0 to :attr:`num_errors`-1 that the application
    wants to analyze and return the error code enum.

    :arg error_idx: A 32bit integer that indexes into the error array.

    :returns: Returns the enum corresponding to error_idx.
)trtdoc";

constexpr char const* get_error_desc = R"trtdoc(
    Returns description of the error.

    For the error specified by the idx value, return description of the error. In the safety context there is a
    constant length requirement to remove any dynamic memory allocations and the error message
    may be truncated. The format of the error description is "<EnumAsStr> - <Description>".

    :arg error_idx: A 32bit integer that indexes into the error array.

    :returns: Returns description of the error.
)trtdoc";

constexpr char const* clear = R"trtdoc(
    Clear the error stack on the error recorder.

    Removes all the tracked errors by the error recorder.  This function must guarantee that after
    this function is called, and as long as no error occurs, :attr:`num_errors` will be zero.
)trtdoc";

constexpr char const* report_error = R"trtdoc(
    Clear the error stack on the error recorder.

    Report an error to the user that has a given value and human readable description. The function returns false
    if processing can continue, which implies that the reported error is not fatal. This does not guarantee that
    processing continues, but provides a hint to TensorRT.

    :arg val: The error code enum that is being reported.
    :arg desc: The description of the error.

    :returns: True if the error is determined to be fatal and processing of the current function must end.
)trtdoc";
} // namespace IErrorRecorderDoc

namespace IExecutionContextDoc
{
constexpr char const* descr = R"trtdoc(
    Context for executing inference using an :class:`ICudaEngine` .
    Multiple :class:`IExecutionContext` s may exist for one :class:`ICudaEngine` instance, allowing the same
    :class:`ICudaEngine` to be used for the execution of multiple batches simultaneously.

    :ivar debug_sync: :class:`bool` The debug sync flag. If this flag is set to true, the :class:`ICudaEngine` will log the successful execution for each kernel during execute_v2().
    :ivar profiler: :class:`IProfiler` The profiler in use by this :class:`IExecutionContext` .
    :ivar engine: :class:`ICudaEngine` The associated :class:`ICudaEngine` .
    :ivar name: :class:`str` The name of the :class:`IExecutionContext` .
    :ivar device_memory: :class:`capsule` The device memory for use by this execution context. The memory must be aligned with cuda memory alignment property (using :func:`cuda.cudart.cudaGetDeviceProperties()`), and its size must be large enough for performing inference with the given network inputs. :func:`engine.device_memory_size` and :func:`engine.get_device_memory_size_for_profile` report upper bounds of the size. Setting memory to nullptr is acceptable if the reported size is 0. If using :func:`execute_async_v3()` to run the network, the memory is in use from the invocation of :func:`execute_async_v3()` until network execution is complete. If using :func:`execute_v2()`, it is in use until :func:`execute_v2()` returns. Releasing or otherwise using the memory for other purposes, including using it in another execution context running in parallel, during this time will result in undefined behavior.
    :ivar active_optimization_profile: :class:`int` The active optimization profile for the context. The selected profile will be used in subsequent calls to :func:`execute_v2()`. Profile 0 is selected by default. This is a readonly property and active optimization profile can be changed with :func:`set_optimization_profile_async()`. Changing this value will invalidate all dynamic bindings for the current execution context, so that they have to be set again using :func:`set_input_shape` before calling either :func:`execute_v2()`.
    :ivar all_binding_shapes_specified: :class:`bool` Whether all dynamic dimensions of input tensors have been specified by calling :func:`set_input_shape` . Trivially true if network has no dynamically shaped input tensors. Does not work with name-base interfaces eg. :func:`set_input_shape()`. Use :func:`infer_shapes()` instead.
    :ivar all_shape_inputs_specified: :class:`bool` Whether values for all input shape tensors have been specified by calling :func:`set_shape_input` . Trivially true if network has no input shape bindings. Does not work with name-base interfaces eg. :func:`set_input_shape()`. Use :func:`infer_shapes()` instead.
    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
    :ivar enqueue_emits_profile: :class:`bool` Whether enqueue emits layer timing to the profiler. The default value is :class:`True`. If set to :class:`False`, enqueue will be asynchronous if there is a profiler attached. An extra method :func:`IExecutionContext::report_to_profiler()` needs to be called to obtain the profiling data and report to the profiler attached.
    :ivar persistent_cache_limit: The maximum size of persistent L2 cache that this execution context may use for activation caching. Activation caching is not supported on all architectures - see "How TensorRT uses Memory" in the developer guide for details. The default is 0 Bytes.
    :ivar nvtx_verbosity: The NVTX verbosity of the execution context. Building with DETAILED verbosity will generally increase latency in enqueueV3(). Call this method to select NVTX verbosity in this execution context at runtime. The default is the verbosity with which the engine was built, and the verbosity may not be raised above that level. This function does not affect how IEngineInspector interacts with the engine.
    :ivar temporary_allocator: :class:`IGpuAllocator` The GPU allocator used for internal temporary storage.
)trtdoc";

constexpr char const* execute_v2 = R"trtdoc(
    Synchronously execute inference on a batch.
    This method requires a array of input and output buffers.

    :arg bindings: A list of integers representing input and output buffer addresses for the network.

    :returns: True if execution succeeded.
)trtdoc";

// TODO: Check if this makes sense to have.
constexpr char const* set_device_memory = R"trtdoc(
    The device memory for use by this :class:`IExecutionContext` .

    :arg memory: 256-byte aligned device memory.
    :arg size: Size of the provided memory. This must be at least as large as CudaEngine.get_device_memory_size_v2

    If using :func:`enqueue_v3()`, it is in use until :func:`enqueue_v3()` returns. Releasing or otherwise using the memory for other
    purposes during this time will result in undefined behavior. This includes using the same memory for a parallel execution context.
)trtdoc";

constexpr char const* set_optimization_profile_async = R"trtdoc(
    Set the optimization profile with async semantics

    :arg profile_index: The index of the optimization profile

    :arg stream_handle: cuda stream on which the work to switch optimization profile can be enqueued

    When an optimization profile is switched via this API, TensorRT may require that data is copied via cudaMemcpyAsync. It is the
    application’s responsibility to guarantee that synchronization between the profile sync stream and the enqueue stream occurs.

    :returns: :class:`True` if the optimization profile was set successfully
)trtdoc";

constexpr char const* report_to_profiler = R"trtdoc(
    Calculate layer timing info for the current optimization profile in IExecutionContext and update the profiler after one iteration of inference launch.

    If the enqueue_emits_profiler flag was set to true, the enqueue function will calculate layer timing implicitly if a profiler is provided. There is no need to call this function.
    If the enqueue_emits_profiler flag was set to false, the enqueue function will record the CUDA event timers if a profiler is provided. But it will not perform the layer timing calculation. This function needs to be called explicitly to calculate layer timing for the previous inference launch.

    In the CUDA graph launch scenario, it will record the same set of CUDA events as in regular enqueue functions if the graph is captured from an :class:`IExecutionContext` with profiler enabled. This function needs to be called after graph launch to report the layer timing info to the profiler.

    Profiling CUDA graphs is only available from CUDA 11.1 onwards.

    :returns: :class:`True` if the call succeeded, else :class:`False` (e.g. profiler not provided, in CUDA graph capture mode, etc.)
)trtdoc";

constexpr char const* get_tensor_strides = R"trtdoc(
    Return the strides of the buffer for the given tensor name.

    Note that strides can be different for different execution contexts with dynamic shapes.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* set_input_shape = R"trtdoc(
    Set shape for the given input tensor.

    :arg name: The input tensor name.
    :arg shape: The input tensor shape.
)trtdoc";

constexpr char const* get_tensor_shape = R"trtdoc(
    Return the shape of the given input or output tensor.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* set_tensor_address = R"trtdoc(
    Set memory address for the given input or output tensor.

    :arg name: The tensor name.
    :arg memory: The memory address.
)trtdoc";

constexpr char const* get_tensor_address = R"trtdoc(
    Get memory address for the given input or output tensor.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* set_input_consumed_event = R"trtdoc(
    Mark all input tensors as consumed.

    :arg event: The cuda event that is triggered after all input tensors have been consumed.
)trtdoc";

constexpr char const* get_input_consumed_event = R"trtdoc(
    Return the event associated with consuming the input tensors.
)trtdoc";

constexpr char const* set_output_allocator = R"trtdoc(
    Set output allocator to use for the given output tensor.

    Pass ``None`` to unset the output allocator.

    The allocator is called by :func:`execute_async_v3`.

    :arg name: The tensor name.
    :arg output_allocator: The output allocator.
)trtdoc";

constexpr char const* get_output_allocator = R"trtdoc(
    Return the output allocator associated with given output tensor, or ``None`` if the provided name does not map to an output tensor.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_max_output_size = R"trtdoc(
    Return the upper bound on an output tensor's size, in bytes, based on the current optimization profile.

    If the profile or input shapes are not yet set, or the provided name does not map to an output, returns -1.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* infer_shapes = R"trtdoc(
    Infer shapes and return the names of any tensors that are insufficiently specified.

    An input tensor is insufficiently specified if either of the following is true:

    * It has dynamic dimensions and its runtime dimensions have not yet
      been specified via :func:`set_input_shape` .

    * is_shape_inference_io(t) is True and the tensor's address has not yet been set.

    :returns: A ``List[str]`` indicating the names of any tensors which have not been sufficiently
        specified, or an empty list on success.

    :raises: RuntimeError if shape inference fails due to reasons other than insufficiently specified tensors.
)trtdoc";

constexpr char const* execute_async_v3 = R"trtdoc(
    Asynchronously execute inference.

    Modifying or releasing memory that has been registered for the tensors before stream synchronization or the event passed to :func:`set_input_consumed_event` has been triggered results in undefined behavior.

    Input tensors can be released after the :func:`set_input_consumed_event` whereas output tensors require stream synchronization.

    :arg stream_handle: The cuda stream on which the inference kernels will be enqueued. Using default stream may lead to performance issues due to additional cudaDeviceSynchronize() calls by TensorRT to ensure correct synchronizations. Please use non-default stream instead.
)trtdoc";

constexpr char const* set_aux_streams = R"trtdoc(
    Set the auxiliary streams that TensorRT should launch kernels on in the next execute_async_v3() call.

    If set, TensorRT will launch the kernels that are supposed to run on the auxiliary streams using the streams provided by the user with this API. If this API is not called before the execute_async_v3() call, then TensorRT will use the auxiliary streams created by TensorRT internally.

    TensorRT will always insert event synchronizations between the main stream provided via execute_async_v3() call and the auxiliary streams:
     - At the beginning of the execute_async_v3() call, TensorRT will make sure that all the auxiliary streams wait on the activities on the main stream.
     - At the end of the execute_async_v3() call, TensorRT will make sure that the main stream wait on the activities on all the auxiliary streams.

    The provided auxiliary streams must not be the default stream and must all be different to avoid deadlocks.

    :arg aux_streams: A list of cuda streams. If the length of the list is greater than engine.num_aux_streams, then only the first "engine.num_aux_streams" streams will be used. If the length is less than engine.num_aux_streams, such as an empty list, then TensorRT will use the provided streams for the first few auxiliary streams, and will create additional streams internally for the rest of the auxiliary streams.
)trtdoc";

constexpr char const* set_debug_listener = R"trtdoc(
    Set debug listener for execution context.

    :arg listener: The :class:`IDebugListener`.
)trtdoc";

constexpr char const* get_debug_listener = R"trtdoc(
    Get debug listener for execution context.

    :returns: The :class:`IDebugListener` of the execution context.
)trtdoc";

constexpr char const* set_tensor_debug_state = R"trtdoc(
    Turn the debug state of a tensor on or off. The Tensor must have been marked as a debug tensor during build time.

    :arg name: The name of the target tensor.
    :arg flag: True if turning on debug state of tensor. False if turning off.
)trtdoc";

constexpr char const* get_debug_state = R"trtdoc(
    Get the debug state of the tensor.

    :arg name: The name of the tensor.
)trtdoc";

constexpr char const* update_device_memory_size_for_shapes = R"trtdoc(
    Recompute the internal activation buffer sizes based on the current input shapes, and return the total amount of memory required.

    Users can allocate the device memory based on the size returned and provided the memory to TRT with an assignment to IExecutionContext.device_memory. Must specify all input shapes and the optimization profile to use before calling this function, otherwise the partition will be invalidated.
)trtdoc";

constexpr char const* set_all_tensors_debug_state = R"trtdoc(
    Turn the debug state of all debug tensors on or off.

    :arg flag: True if turning on debug state of tensor. False if turning off.
)trtdoc";

constexpr char const* get_runtime_config = R"trtdoc(
    Get the runtime configuration. From the execution context.

    :returns: The runtime configuration.
)trtdoc";


} // namespace IExecutionContextDoc

namespace IDebugListenerDoc
{
constexpr char const* descr = R"trtdoc(
    A user-implemented class for notification when value of a debug tensor is updated.
)trtdoc";

constexpr char const* get_interface_version = R"trtdoc(
    The version of this interface.

    :returns: The version number.
)trtdoc";

constexpr char const* process_debug_tensor = R"trtdoc(
    User implemented callback function that is called when value of a debug tensor is updated and the debug state of the tensor is set to true. Content in the given address is only guaranteed to be valid for the duration of the callback.

    :arg location: TensorLocation of the tensor
    :arg addr: pointer to buffer
    :arg type: data Type of the tensor
    :arg shape: shape of the tensor
    :arg name: name name of the tensor
    :arg stream: Cuda stream object

    :returns: True on success, False otherwise.
)trtdoc";
} // namespace IDebugListenerDoc

namespace IProgressMonitorDoc
{
constexpr char const* descr = R"trtdoc(
    Application-implemented progress reporting interface for TensorRT.

    The IProgressMonitor is a user-defined object that TensorRT uses to report back when an internal algorithm has
    started or finished a phase to help provide feedback on the progress of the optimizer.

    The IProgressMonitor will trigger its start function when a phase is entered and will trigger its finish function
    when that phase is exited. Each phase consists of one or more steps. When each step is completed, the step_complete
    function is triggered. This will allow an application using the builder to communicate progress relative to when the
    optimization step is expected to complete.

    The implementation of IProgressMonitor must be thread-safe so that it can be called from multiple internal threads.
    The lifetime of the IProgressMonitor must exceed the lifetime of all TensorRT objects that use it.
)trtdoc";

constexpr char const* phase_start = R"trtdoc(
    Signal that a phase of the optimizer has started.

    :arg phase_name: The name of this phase for tracking purposes.
    :arg parent_phase: The parent phase that this phase belongs to, None if there is no parent.
    :arg num_steps: The number of steps that are involved in this phase.

    The phase_start function signals to the application that the current phase is beginning, and that it has a
    certain number of steps to perform. If phase_parent is None, then the phase_start is beginning an
    independent phase, and if phase_parent is specified, then the current phase, specified by phase_name, is
    within the scope of the parent phase. num_steps will always be a positive number. The phase_start function
    implies that the first step is being executed. TensorRT will signal when each step is complete.

    Phase names are human readable English strings which are unique within a single phase hierarchy but which can be
    reused once the previous instance has completed. Phase names and their hierarchies may change between versions
    of TensorRT.
)trtdoc";

constexpr char const* step_complete = R"trtdoc(
    Signal that a step of an optimizer phase has finished.

    :arg phase_name: The name of the innermost phase being executed.
    :arg step: The step number that was completed.

    The step_complete function signals to the application that TensorRT has finished the current step for the phase
    ``phase_name`` , and will move on to the next step if there is one. The application can return False for TensorRT to exit
    the build early. The step value will increase on subsequent calls in the range [0, num_steps).

    :returns: True to continue to the next step or False to stop the build.
)trtdoc";

constexpr char const* phase_finish = R"trtdoc(
    Signal that a phase of the optimizer has finished.

    :arg phase_name: The name of the phase that has finished.

    The phase_finish function signals to the application that the phase is complete. This function may be called before
    all steps in the range [0, num_steps) have been reported to step_complete. This scenario can be triggered by error
    handling, internal optimizations, or when step_complete returns False to request cancellation of the build.
)trtdoc";
} // namespace IProgressMonitorDoc

namespace IRuntimeConfigDoc
{
constexpr char const* descr = R"trtdoc(
    A runtime configuration for an :class:`ICudaEngine` .
)trtdoc";

constexpr char const* set_execution_context_allocation_strategy = R"trtdoc(
    Set the execution context allocation strategy.

    :arg strategy: The execution context allocation strategy.
)trtdoc";

constexpr char const* get_execution_context_allocation_strategy = R"trtdoc(
    Get the execution context allocation strategy.

    :returns: The execution context allocation strategy.
)trtdoc";

} // namespace IRuntimeConfigDoc


namespace EngineStatDoc
{
constexpr char const* descr = R"trtdoc(The kind of engine statistics that queried from the ICudaEngine.)trtdoc";
constexpr char const* TOTAL_WEIGHTS_SIZE = R"trtdoc(The total weights size in bytes in the engine.)trtdoc";
constexpr char const* STRIPPED_WEIGHTS_SIZE
    = R"trtdoc(The stripped weight size in bytes for engines built with BuilderFlag::kSTRIP_PLAN.)trtdoc";
} // namespace EngineStatDoc

namespace ICudaEngineDoc
{
constexpr char const* descr = R"trtdoc(
    An :class:`ICudaEngine` for executing inference on a built network.

    The engine can be indexed with ``[]`` . When indexed in this way with an integer, it will return the corresponding binding name. When indexed with a string, it will return the corresponding binding index.

    :ivar num_io_tensors: :class:`int` The number of IO tensors.
    :ivar has_implicit_batch_dimension: :class:`bool` [DEPRECATED] Deprecated in TensorRT 10.0. Always flase since the implicit batch dimensions support has been removed.
    :ivar num_layers: :class:`int` The number of layers in the network. The number of layers in the network is not necessarily the number in the original :class:`INetworkDefinition`, as layers may be combined or eliminated as the :class:`ICudaEngine` is optimized. This value can be useful when building per-layer tables, such as when aggregating profiling data over a number of executions.
    :ivar max_workspace_size: :class:`int` The amount of workspace the :class:`ICudaEngine` uses. The workspace size will be no greater than the value provided to the :class:`Builder` when the :class:`ICudaEngine` was built, and will typically be smaller. Workspace will be allocated for each :class:`IExecutionContext` .
    :ivar device_memory_size: :class:`int` The amount of device memory required by an :class:`IExecutionContext` .
    :ivar device_memory_size_v2: :class:`int` The amount of device memory required by an :class:`IExecutionContext`. The return value depends on the weight streaming budget if enabled.
    :ivar refittable: :class:`bool` Whether the engine can be refit.
    :ivar name: :class:`str` The name of the network associated with the engine. The name is set during network creation and is retrieved after building or deserialization.
    :ivar num_optimization_profiles: :class:`int` The number of optimization profiles defined for this engine. This is always at least 1.
    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
    :ivar engine_capability: :class:`EngineCapability` The engine capability. See :class:`EngineCapability` for details.
    :ivar tactic_sources: :class:`int` The tactic sources required by this engine.
    :ivar profiling_verbosity: The profiling verbosity the builder config was set to when the engine was built.
    :ivar hardware_compatibility_level: The hardware compatibility level of the engine.
    :ivar num_aux_streams: Read-only. The number of auxiliary streams used by this engine, which will be less than or equal to the maximum allowed number of auxiliary streams by setting builder_config.max_aux_streams when the engine is built.
    :ivar weight_streaming_budget: [DEPRECATED] Deprecated in TensorRT 10.1, superceded by weight_streaming_budget_v2. Set and get the current weight streaming budget for inference. The budget may be set to -1 disabling weight streaming at runtime, 0 (default) enabling TRT to choose to weight stream or not, or a positive value in the inclusive range [minimum_weight_streaming_budget, streamable_weights_size - 1].
    :ivar minimum_weight_streaming_budget: [DEPRECATED] Deprecated in TensorRT 10.1, superceded by weight_streaming_budget_v2. Returns the minimum weight streaming budget in bytes required to run the network successfully. The engine must have been built with kWEIGHT_STREAMING.
    :ivar streamable_weights_size: Returns the size of the streamable weights in the engine. This may not include all the weights.
    :ivar weight_streaming_budget_v2: Set and get the current weight streaming budget for inference. The budget may be set any non-negative value. A value of 0 streams the most weights. Values equal to streamable_weights_size (default) or larger will disable weight streaming.
    :ivar weight_streaming_scratch_memory_size: The amount of scratch memory required by a TensorRT ExecutionContext to perform inference. This value may change based on the current weight streaming budget. Please use the V2 memory APIs, engine.device_memory_size_v2 and ExecutionContext.set_device_memory() to provide memory which includes the current weight streaming scratch memory. Not specifying these APIs or using the V1 APIs will not include this memory, so TensorRT will resort to allocating itself.
    )trtdoc"
    ;

// Documentation bug with parameters on these three functions because they are overloaded.
constexpr char const* serialize = R"trtdoc(
    Serialize the engine to a stream.

    :returns: An :class:`IHostMemory` object containing the serialized :class:`ICudaEngine` .
)trtdoc";

constexpr char const* create_execution_context = R"trtdoc(
    Create an :class:`IExecutionContext` and specify the device memory allocation strategy.

    :returns: The newly created :class:`IExecutionContext` .
)trtdoc";

constexpr char const* create_execution_context_without_device_memory = R"trtdoc(
    Create an :class:`IExecutionContext` without any device memory allocated
    The memory for execution of this device context must be supplied by the application.

    :returns: An :class:`IExecutionContext` without device memory allocated.
)trtdoc";

constexpr char const* create_execution_context_with_runtime_config = R"trtdoc(
    Create an :class:`IExecutionContext` with a runtime configuration.

    :arg runtime_config: The runtime configuration.
    :returns: The newly created :class:`IExecutionContext` .
)trtdoc";

constexpr char const* create_runtime_config = R"trtdoc(
    Create a runtime configuration.

    :returns: The newly created :class:`IRuntimeConfig` .
)trtdoc";

constexpr char const* get_tensor_profile_values = R"trtdoc(
    Get minimum/optimum/maximum values for an input shape binding under an optimization profile. If the specified binding is not an input shape binding, an exception is raised.

    :arg name: The tensor name.
    :arg profile_index: The index of the profile.

    :returns: A ``List[List[int]]`` of length 3, containing the minimum, optimum, and maximum values, in that order. If the values have not been set yet, an empty list is returned.
)trtdoc";

constexpr char const* create_engine_inspector = R"trtdoc(
    Create an :class:`IEngineInspector` which prints out the layer information of an engine or an execution context.

    :returns: The :class:`IEngineInspector`.
)trtdoc";

// Docs for enqueueV3 related APIs
constexpr char const* get_tensor_name = R"trtdoc(
    Return the name of an input or output tensor.

    :arg index: The tensor index.
)trtdoc";

constexpr char const* get_tensor_mode = R"trtdoc(
    Determine whether a tensor is an input or output tensor.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* is_shape_inference_io = R"trtdoc(
    Determine whether a tensor is read or written by infer_shapes.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_tensor_shape = R"trtdoc(
    Return the shape of an input or output tensor.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_tensor_dtype = R"trtdoc(
    Return the required data type for a buffer from its tensor name.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_tensor_location = R"trtdoc(
    Determine whether an input or output tensor must be on GPU or CPU.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_tensor_bytes_per_component = R"trtdoc(
    Return the number of bytes per component of an element.

    The vector component size is returned if :func:`get_tensor_vectorized_dim` != -1.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_tensor_components_per_element = R"trtdoc(
    Return the number of components included in one element.

    The number of elements in the vectors is returned if :func:`get_tensor_vectorized_dim` != -1.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_tensor_format = R"trtdoc(
    Return the tensor format.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_tensor_format_desc = R"trtdoc(
    Return the human readable description of the tensor format.

    The description includes the order, vectorization, data type, strides, etc. For example:

    |   Example 1: CHW + FP32
    |       "Row major linear FP32 format"
    |   Example 2: CHW2 + FP16
    |       "Two wide channel vectorized row major FP16 format"
    |   Example 3: HWC8 + FP16 + Line Stride = 32
    |       "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_tensor_vectorized_dim = R"trtdoc(
    Return the dimension index that the buffer is vectorized.

    Specifically -1 is returned if scalars per vector is 1.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_tensor_profile_shape = R"trtdoc(
    Get the minimum/optimum/maximum dimensions for a particular tensor under an optimization profile.

    :arg name: The tensor name.
    :arg profile_index: The index of the profile.
)trtdoc";

constexpr char const* get_device_memory_size_for_profile = R"trtdoc(
    Return the device memory size required for a certain profile.

    :arg profile_index: The index of the profile.
)trtdoc";

constexpr char const* get_device_memory_size_for_profile_v2 = R"trtdoc(
    Return the device memory size required for a certain profile.

    The return value will change depending on the following API calls
    1. setWeightStreamingBudgetV2

    :arg profile_index: The index of the profile.
)trtdoc";

constexpr char const* create_serialization_config = R"trtdoc(
    Create a serialization configuration object.
)trtdoc";

constexpr char const* serialize_with_config = R"trtdoc(
    Serialize the network to a stream.
)trtdoc";

constexpr char const* get_weight_streaming_automatic_budget = R"trtdoc(
    Get an automatic weight streaming budget based on available device memory. This value may change between TensorRT major and minor versions.
    Please use CudaEngine.weight_streaming_budget_v2 to set the returned budget.
)trtdoc";

constexpr char const* is_debug_tensor = R"trtdoc(
    Determine whether the given name corresponds to a debug tensor.

    :arg name: The tensor name.
)trtdoc";

constexpr char const* get_engine_stat = R"trtdoc(
    Return the engine statistics specified by the given enum value.
    If STRIPPED_WEIGHTS_SIZE is passed to query a normal engine, this function will
    return -1 to indicate invalid enum value.

    :arg stat: The engine statistic kind to get.
)trtdoc";
} // namespace ICudaEngineDoc

namespace OutputAllocatorDoc
{
constexpr char const* descr = R"trtdoc(
Application-implemented class for controlling output tensor allocation.

To implement a custom output allocator, ensure that you explicitly instantiate the base class in :func:`__init__` :
::

    class MyOutputAllocator(trt.IOutputAllocator):
        def __init__(self):
            trt.IOutputAllocator.__init__(self)

        def reallocate_output(self, tensor_name, memory, size, alignment):
            ... # Your implementation here

        def reallocate_output_async(self, tensor_name, memory, size, alignment, stream):
            ... # Your implementation here

        def notify_shape(self, tensor_name, shape):
            ... # Your implementation here

)trtdoc";

constexpr char const* reallocate_output = R"trtdoc(
    A callback implemented by the application to handle acquisition of output tensor memory.

    If an allocation request cannot be satisfied, ``None`` should be returned.

    :arg tensor_name: The output tensor name.
    :arg memory: The output tensor memory address.
    :arg size: The number of bytes required.
    :arg alignment: The required alignment of memory.

    :returns: The address of the output tensor memory.
)trtdoc";

constexpr char const* reallocate_output_async = R"trtdoc(
    A callback implemented by the application to handle acquisition of output tensor memory.

    If an allocation request cannot be satisfied, ``None`` should be returned.

    :arg tensor_name: The output tensor name.
    :arg memory: The output tensor memory address.
    :arg size: The number of bytes required.
    :arg alignment: The required alignment of memory.
    :arg stream: CUDA stream

    :returns: The address of the output tensor memory.
)trtdoc";

constexpr char const* notify_shape = R"trtdoc(
    Called by TensorRT when the shape of the output tensor is known.

    :arg tensor_name: The output tensor name.
    :arg shape: The output tensor shape.
)trtdoc";

} // namespace OutputAllocatorDoc

namespace StreamReaderDoc
{
constexpr char const* descr = R"trtdoc(
Application-implemented class for reading data from a stream.

To implement a custom stream reader, ensure that you explicitly instantiate the base class in :func:`__init__` :
::

    class MyStreamReader(trt.IStreamReader):
        def __init__(self):
            trt.IStreamReader.__init__(self)

        def read(self, size: int) -> bytes:
            ... # Your implementation here

)trtdoc";

constexpr char const* read = R"trtdoc(
    A callback implemented by the application to read a particular chunk of memory.

    If an allocation request cannot be satisfied, ``0`` should be returned.

    :arg size: The number of bytes required.

    :returns: A buffer containing the bytes read.
)trtdoc";
} // namespace StreamReaderDoc

namespace StreamReaderV2Doc
{
constexpr char const* descr = R"trtdoc(
    Application-implemented class for asynchronously reading data from a stream. Implementation does not need to be
    asynchronous or use the provided cuda stream. Python users are unlikely to see performance gains over IStreamReader
    or deserialization from a glob.

    To implement a custom stream reader, ensure that you explicitly instantiate the base class in :func:`__init__` :
    ::
        class MyStreamReader(trt.IStreamReaderV2):
            def __init__(self):
                trt.IStreamReaderV2.__init__(self)

            def read(self, num_bytes: int, stream: int) -> bytes:
                ... # Your implementation here

            def seek(self, offset: int, where: SeekPosition) -> bool:
                ... # Your implementation here

    To implement a read operation that retrieves the entire buffer, use the following approach:
    ::
        def read(self, num_bytes: int, stream: int) -> bytes:
            data = read_file_with_size(num_bytes)
            return data

    To read the buffer in chunks to reduce peak host memory usage (e.g., splitting the read into 200MiB chunks):
    ::
        def read(self, num_bytes: int, stream: int) -> bytes:
            chunk_size = 200 * 1024 * 1024  # 200MiB
            num_bytes = min(num_bytes, chunk_size)
            data = read_file_with_size(num_bytes)
            return data
)trtdoc";

constexpr char const* read = R"trtdoc(
    A callback implemented by the application to read a particular chunk of memory.

    :arg num_bytes: The number of bytes to read. If N bytes are requested but only M bytes (M < N) are read, additional read requests will be made until all N bytes are retrieved.
    :arg stream: A handle to the cudaStream your implementation can use for reading.

    :returns: A buffer containing the bytes read.
)trtdoc";

constexpr char const* seek = R"trtdoc(
    A callback implemented by the application to set the stream location.

    :arg offset: The offset within the stream to seek to.
    :arg where: A `SeekPosition` enum specifying where the offset is relative to.

    :returns: A buffer containing the bytes read.
)trtdoc";
} // namespace StreamReaderV2Doc

namespace StreamWriterDoc
{
constexpr char const* descr = R"trtdoc(
Application-implemented class for writing data to a stream.

To implement a custom stream writer, ensure that you explicitly instantiate the base class in :func:`__init__` :
::

    class MyStreamWriter(trt.IStreamWriter):
        def __init__(self):
            trt.IStreamWriter.__init__(self)

        def write(self, data: bytes) -> int:
            ... # Your implementation here

)trtdoc";

constexpr char const* write = R"trtdoc(
    A callback implemented by the application to write a particular chunk of memory.

    :arg data: The data to be written out in bytes.

    :returns: The total bytes actually be written.
)trtdoc";
} // namespace StreamWriterDoc

namespace SeekPositionDoc
{
constexpr char const* descr
    = R"trtdoc(Specifies what the offset is relative to when calling `seek` on an `IStreamReaderV2`.)trtdoc";
constexpr char const* SET = R"trtdoc(Offsets forward from the start of the stream.)trtdoc";
constexpr char const* CUR = R"trtdoc(Offsets forward from the current position within the stream.)trtdoc";
constexpr char const* END = R"trtdoc(Offsets backward from the end of the stream.)trtdoc";
} // namespace SeekPositionDoc

namespace BuilderFlagDoc
{
constexpr char const* descr
    = R"trtdoc(Valid modes that the builder can enable when creating an engine from a network definition.)trtdoc";

constexpr char const* FP16
     = R"trtdoc(Enable FP16 layer selection.
                [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.)trtdoc";
constexpr char const* BF16
    = R"trtdoc(Enable BF16 layer selection.
               [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.)trtdoc";
constexpr char const* INT8
    = R"trtdoc(Enable Int8 layer selection.
               [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.)trtdoc";
constexpr char const* DEBUG = R"trtdoc(Enable debugging of layers via synchronizing after every layer)trtdoc";
constexpr char const* GPU_FALLBACK
    = R"trtdoc(Enable layers marked to execute on GPU if layer cannot execute on DLA)trtdoc";
constexpr char const* REFIT = R"trtdoc(Enable building a refittable engine)trtdoc";
constexpr char const* DISABLE_TIMING_CACHE
    = R"trtdoc(Disable reuse of timing information across identical layers.)trtdoc";
constexpr char const* EDITABLE_TIMING_CACHE = R"trtdoc(Enable the editable timing cache.)trtdoc";
constexpr char const* TF32
    = R"trtdoc(Allow (but not require) computations on tensors of type DataType.FLOAT to use TF32. TF32 computes inner products by rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas. Enabled by default.)trtdoc";
constexpr char const* SPARSE_WEIGHTS
    = R"trtdoc(Allow the builder to examine weights and use optimized functions when weights have suitable sparsity.)trtdoc";
constexpr char const* SAFETY_SCOPE
    = R"trtdoc(Change the allowed parameters in the EngineCapability.STANDARD flow to match the restrictions that EngineCapability.SAFETY check against for DeviceType.GPU and EngineCapability.DLA_STANDALONE check against the DeviceType.DLA case. This flag is forced to true if EngineCapability.SAFETY at build time if it is unset.)trtdoc";
constexpr char const* OBEY_PRECISION_CONSTRAINTS
    = R"trtdoc(Require that layers execute in specified precisions. Build fails otherwise.
               [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.)trtdoc";
constexpr char const* PREFER_PRECISION_CONSTRAINTS
    = R"trtdoc(Prefer that layers execute in specified precisions. Fall back (with warning) to another precision if build would otherwise fail.
               [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.)trtdoc";
constexpr char const* DIRECT_IO
    = R"trtdoc(Require that no reformats be inserted between a layer and a network I/O tensor for which ``ITensor.allowed_formats`` was set. Build fails if a reformat is required for functional correctness.
               [DEPRECATED] Deprecated in TensorRT 10.7.))trtdoc";
constexpr char const* REJECT_EMPTY_ALGORITHMS
    = R"trtdoc([DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead. Fail if IAlgorithmSelector.select_algorithms returns an empty set of algorithms.)trtdoc";
constexpr char const* VERSION_COMPATIBLE
    = R"trtdoc(Restrict to lean runtime operators to provide version forward compatibility for the plan files.)trtdoc";
constexpr char const* EXCLUDE_LEAN_RUNTIME = R"trtdoc(Exclude lean runtime from the plan.)trtdoc";
constexpr char const* FP8
 = R"trtdoc(Enable plugins with FP8 input/output
    [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.)trtdoc";
constexpr char const* ERROR_ON_TIMING_CACHE_MISS
    = R"trtdoc(Emit error when a tactic being timed is not present in the timing cache.)trtdoc";
constexpr char const* DISABLE_COMPILATION_CACHE
    = R"trtdoc(Disable caching JIT compilation results during engine build.)trtdoc";
constexpr char const* WEIGHTLESS
    = R"trtdoc(Strip the perf-irrelevant weights from the plan file, update them later using refitting for better file size.
               [DEPRECATED] Deprecated in TensorRT 10.0.
               )trtdoc";
constexpr char const* STRIP_PLAN = R"trtdoc(Strip the refittable weights from the engine plan file.)trtdoc";
constexpr char const* REFIT_IDENTICAL
    = R"trtdoc(Create a refittable engine using identical weights. Different weights during refits yield unpredictable behavior.)trtdoc";
constexpr char const* REFIT_INDIVIDUAL
    = R"trtdoc(Create a refittable engine and allows the users to specify which weights are refittable and which are not.)trtdoc";
constexpr char const* WEIGHT_STREAMING
    = R"trtdoc(Enable building with the ability to stream varying amounts of weights during Runtime. This decreases GPU memory of TRT at the expense of performance.)trtdoc";
constexpr char const* INT4
    = R"trtdoc(Enable plugins with INT4 input/output
               [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.)trtdoc";
constexpr char const* STRICT_NANS
    = R"trtdoc(Disable floating-point optimizations: 0*x => 0, x-x => 0, or x/x => 1. These identities are not true when x is a NaN or Inf, and thus might hide propagation or generation of NaNs.)trtdoc";
constexpr char const* FP4
    = R"trtdoc(Enable plugins with FP4 input/output
               [DEPRECATED] Deprecated in TensorRT 10.12. Superseded by strong typing.)trtdoc";
constexpr char const* MONITOR_MEMORY = R"trtdoc(Enable memory monitor during build time.)trtdoc";
constexpr char const* DISTRIBUTIVE_INDEPENDENCE = R"trtdoc(Enable distributive independence.)trtdoc";
} // namespace BuilderFlagDoc

namespace MemoryPoolTypeDoc
{
constexpr char const* descr = R"trtdoc(The type for memory pools used by TensorRT.)trtdoc";
constexpr char const* WORKSPACE = R"trtdoc(
    WORKSPACE is used by TensorRT to store intermediate buffers within an operation.
    This defaults to max device memory. Set to a smaller value to restrict tactics that use over the threshold en masse.
    For more targeted removal of tactics use the IAlgorithmSelector interface ([DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead).
)trtdoc";
constexpr char const* DLA_MANAGED_SRAM = R"trtdoc(
    DLA_MANAGED_SRAM is a fast software managed RAM used by DLA to communicate within a layer.
    The size of this pool must be at least 4 KiB and must be a power of 2.
    This defaults to 1 MiB.
    Orin has capacity of 1 MiB per core.
)trtdoc";
constexpr char const* DLA_LOCAL_DRAM = R"trtdoc(
    DLA_LOCAL_DRAM is host RAM used by DLA to share intermediate tensor data across operations.
    The size of this pool must be at least 4 KiB and must be a power of 2.
    This defaults to 1 GiB.
)trtdoc";
constexpr char const* DLA_GLOBAL_DRAM = R"trtdoc(
    DLA_GLOBAL_DRAM is host RAM used by DLA to store weights and metadata for execution.
    The size of this pool must be at least 4 KiB and must be a power of 2.
    This defaults to 512 MiB.
)trtdoc";
constexpr char const* TACTIC_DRAM = R"trtdoc(
    TACTIC_DRAM is the host DRAM used by the optimizer to
    run tactics. On embedded devices, where host and device memory are unified, this includes all device
    memory required by TensorRT to build the network up to the point of each memory allocation.
    This defaults to 75% of totalGlobalMem as reported by cudaGetDeviceProperties when
    cudaGetDeviceProperties.embedded is true, and 100% otherwise.
)trtdoc";
constexpr char const* TACTIC_SHARED_MEMORY = R"trtdoc(
    TACTIC_SHARED_MEMORY defines the maximum shared memory size utilized for driver reserved and executing
    the backend CUDA kernel implementation. Adjust this value to restrict tactics that exceed
    the specified threshold en masse. The default value is device max capability. This value must
    be less than 1GiB.

    Updating this flag will override the shared memory limit set by \ref HardwareCompatibilityLevel,
    which defaults to 48KiB.
)trtdoc";

} // namespace MemoryPoolTypeDoc

namespace QuantizationFlagDoc
{
constexpr char const* descr = R"trtdoc(List of valid flags for quantizing the network to int8.)trtdoc";

constexpr char const* CALIBRATE_BEFORE_FUSION
    = R"trtdoc(Run int8 calibration pass before layer fusion. Only valid for IInt8LegacyCalibrator and IInt8EntropyCalibrator. We always run int8 calibration pass before layer fusion for IInt8MinMaxCalibrator and IInt8EntropyCalibrator2. Disabled by default.)trtdoc";
} // namespace QuantizationFlagDoc

namespace PreviewFeatureDoc
{
constexpr char const* descr = R"trtdoc(
    List of Preview Features that can be enabled. Preview Features have been fully tested but are not yet as stable as other features in TensorRT.
    They are provided as opt-in features for at least one release.
    For example, to enable faster dynamic shapes, call :func:`set_preview_feature` with ``PreviewFeature.PROFILE_SHARING_0806``
)trtdoc";
constexpr char const* PROFILE_SHARING_0806 = R"trtdoc(
    [DEPRECATED] Allows optimization profiles to be shared across execution contexts. The default value for this flag is on in TensorRT 10.0. Turning if off is deprecated.
)trtdoc";
constexpr char const* ALIASED_PLUGIN_IO_10_03 = R"trtdoc(
    Allows plugin I/O to be aliased when using IPluginV3OneBuildV2.
)trtdoc";
constexpr char const* RUNTIME_ACTIVATION_RESIZE_10_10 = R"trtdoc(
    Allow update_device_memory_size_for_shapes to resize runner internal activation memory by changing the allocation algorithm. Using this feature can reduce runtime memory requirement when the actual input tensor shapes are smaller than the maximum input tensor dimensions.
)trtdoc";
} // namespace PreviewFeatureDoc


namespace HardwareCompatibilityLevelDoc
{
constexpr char const* descr = R"trtdoc(
    Describes requirements of compatibility with GPU architectures other than that of the GPU on which the engine was built.
    Note that compatibility with future hardware depends on CUDA forward compatibility support.
)trtdoc";
constexpr char const* NONE = R"trtdoc(
    Do not require hardware compatibility with GPU architectures other than that of the GPU on which the engine was built.
)trtdoc";
constexpr char const* AMPERE_PLUS = R"trtdoc(
    Require that the engine is compatible with Ampere and newer GPUs. This will limit the combined usage of driver reserved and backend kernel max shared memory to
    48KiB, may reduce the number of available tactics for each layer, and may prevent some fusions from occurring.
    Thus this can decrease the performance, especially for tf32 models.
    This option will disable cuDNN, cuBLAS, and cuBLASLt as tactic sources.
    This option is only supported for engines built on NVIDIA Ampere and later GPUs.
)trtdoc";
constexpr char const* SAME_COMPUTE_CAPABILITY = R"trtdoc(
    Require that the engine is compatible with GPUs that have the same Compute Capability version as the one it was built on.
    This may decrease the performance compared to an engine with no compatibility.
    This option will disable cuDNN, cuBLAS, and cuBLASLt as tactic sources.
    This option is only supported for engines built on NVIDIA Turing and later GPUs.
)trtdoc";
} // namespace HardwareCompatibilityLevelDoc

namespace RuntimePlatformDoc
{
constexpr char const* descr = R"trtdoc(
    Describes the intended runtime platform for the execution of the TensorRT engine.
    TensorRT provides support for cross-platform engine compatibility when the target runtime platform is different from the build platform.

    **NOTE:** The cross-platform engine will not be able to run on the host platform it was built on.

    **NOTE:** When building a cross-platform engine that also requires version forward compatibility, EXCLUDE_LEAN_RUNTIME must be set to exclude the target platform lean runtime.

    **NOTE:** The cross-platform engine might have performance differences compared to the natively built engine on the target platform.
)trtdoc";
constexpr char const* SAME_AS_BUILD = R"trtdoc(
    No requirement for cross-platform compatibility. The engine constructed by TensorRT can only run on the identical platform it was built on.
)trtdoc";
constexpr char const* WINDOWS_AMD64 = R"trtdoc(
    Designates the target platform for engine execution as Windows AMD64 system. Currently this flag can only be enabled when building engines on Linux AMD64 platforms.
)trtdoc";
} // namespace RuntimePlatformDoc

namespace TilingOptimizationLevelDoc
{
constexpr char const* descr = R"trtdoc(
    Describes the optimization level of tiling strategies. A higher level allows TensorRT to spend more time searching for better tiling strategy.
)trtdoc";
constexpr char const* NONE = R"trtdoc(
    Do not apply any tiling strategy.
)trtdoc";
constexpr char const* FAST = R"trtdoc(
    Use a fast algorithm and heuristic based strategy. Slightly increases engine build time.
)trtdoc";
constexpr char const* MODERATE = R"trtdoc(
    Increase search space and use a mixed heuristic/profiling strategy. Moderately increases engine build time.
)trtdoc";
constexpr char const* FULL = R"trtdoc(
    Increase search space even wider. Significantly increases engine build time.
)trtdoc";
} // namespace TilingOptimizationLevelDoc

namespace NetworkDefinitionCreationFlagDoc
{
constexpr char const* descr
    = R"trtdoc(List of immutable network properties expressed at network creation time. For example, to enable strongly typed mode, pass a value of ``1 << int(NetworkDefinitionCreationFlag.STRONGLY_TYPED)`` to :func:`create_network` )trtdoc";
constexpr char const* EXPLICIT_BATCH
    = R"trtdoc([DEPRECATED] Ignored because networks are always "explicit batch" in TensorRT 10.0.)trtdoc";
constexpr char const* STRONGLY_TYPED
    = R"trtdoc(Specify that every tensor in the network has a data type defined in the network following only type inference rules and the inputs/operator annotations. Setting layer precision and layer output types is not allowed, and the network output types will be inferred based on the input types and the type inference rules)trtdoc";
constexpr char const* PREFER_AOT_PYTHON_PLUGINS
    = R"trtdoc(If set, for a Python plugin with both AOT and JIT implementations, the AOT implementation will be used.)trtdoc";
constexpr char const* PREFER_JIT_PYTHON_PLUGINS
    = R"trtdoc(If set, for a Python plugin with both AOT and JIT implementations, the JIT implementation will be used.)trtdoc";
} // namespace NetworkDefinitionCreationFlagDoc

namespace ISerializationConfigDoc
{
constexpr char const* descr
    = R"trtdoc(Class to hold properties for configuring an engine to serialize the binary.)trtdoc";
constexpr char const* clear_flag = R"trtdoc(Clears the serialization flag from the config.)trtdoc";
constexpr char const* get_flag = R"trtdoc(Check if a serialization flag is set.)trtdoc";
constexpr char const* set_flag = R"trtdoc(Add the input serialization flag to the already enabled flags.)trtdoc";
} // namespace ISerializationConfigDoc

namespace DeviceTypeDoc
{
constexpr char const* descr = R"trtdoc(Device types that TensorRT can execute on)trtdoc";

constexpr char const* GPU = R"trtdoc(GPU device)trtdoc";
constexpr char const* DLA = R"trtdoc(DLA core)trtdoc";
} // namespace DeviceTypeDoc

namespace TempfileControlFlagDoc
{
constexpr char const* descr = R"trtdoc(
Flags used to control TensorRT's behavior when creating executable temporary files.

On some platforms the TensorRT runtime may need to create files in a temporary directory or use platform-specific
APIs to create files in-memory to load temporary DLLs that implement runtime code. These flags allow the
application to explicitly control TensorRT's use of these files. This will preclude the use of certain TensorRT
APIs for deserializing and loading lean runtimes.

These should be treated as bit offsets, e.g. in order to allow in-memory files for a given :class:`IRuntime`:

.. code-block:: python

    runtime.tempfile_control_flags |= (1 << int(TempfileControlFlag.ALLOW_IN_MEMORY_FILES))

)trtdoc";

constexpr char const* ALLOW_IN_MEMORY_FILES
    = R"trtdoc(Allow creating and loading files in-memory (or unnamed files).)trtdoc";
constexpr char const* ALLOW_TEMPORARY_FILES
    = R"trtdoc(Allow creating and loading named files in a temporary directory on the filesystem.)trtdoc";

} // namespace TempfileControlFlagDoc

namespace ProfilingVerbosityDoc
{
constexpr char const* descr = R"trtdoc(Profiling verbosity in NVTX annotations and the engine inspector)trtdoc";

constexpr char const* LAYER_NAMES_ONLY = R"trtdoc(Print only the layer names. This is the default setting.)trtdoc";
constexpr char const* DETAILED
    = R"trtdoc(Print detailed layer information including layer names and layer parameters.)trtdoc";
constexpr char const* NONE = R"trtdoc(Do not print any layer information.)trtdoc";
} // namespace ProfilingVerbosityDoc

namespace TensorIOModeDoc
{
constexpr char const* descr = R"trtdoc(IO tensor modes for TensorRT.)trtdoc";

constexpr char const* NONE = R"trtdoc(Tensor is not an input or output.)trtdoc";
constexpr char const* INPUT = R"trtdoc(Tensor is input to the engine.)trtdoc";
constexpr char const* OUTPUT = R"trtdoc(Tensor is output to the engine.)trtdoc";
} // namespace TensorIOModeDoc

namespace TacticSourceDoc
{
constexpr char const* descr = R"trtdoc(Tactic sources that can provide tactics for TensorRT.)trtdoc";

constexpr char const* CUBLAS = R"trtdoc(
        Enables cuBLAS tactics. Disabled by default.
        [DEPRECATED] Deprecated in TensorRT 10.0.
        **NOTE:** Disabling CUBLAS tactic source will cause the cuBLAS handle passed to plugins in attachToContext to be null.
    )trtdoc";
constexpr char const* CUBLAS_LT = R"trtdoc(
        Enables cuBLAS LT tactics. Disabled by default.
        [DEPRECATED] Deprecated in TensorRT 9.0.
    )trtdoc";
constexpr char const* CUDNN = R"trtdoc(
        Enables cuDNN tactics. Disabled by default.
        [DEPRECATED] Deprecated in TensorRT 10.0.
        **NOTE:** Disabling CUDNN tactic source will cause the cuDNN handle passed to plugins in attachToContext to be null.
    )trtdoc";
constexpr char const* EDGE_MASK_CONVOLUTIONS = R"trtdoc(
        Enables convolution tactics implemented with edge mask tables. These tactics tradeoff memory for performance
        by consuming additional memory space proportional to the input size. Enabled by default.
    )trtdoc";
constexpr char const* JIT_CONVOLUTIONS = R"trtdoc(
        Enables convolution tactics implemented with source-code JIT fusion. The engine building time may increase
        when this is enabled. Enabled by default.
    )trtdoc";
} // namespace TacticSourceDoc

namespace EngineCapabilityDoc
{
constexpr char const* descr = R"trtdoc(
    List of supported engine capability flows.
    The EngineCapability determines the restrictions of a network during build time and what runtime
    it targets. When BuilderFlag::kSAFETY_SCOPE is not set (by default), EngineCapability.STANDARD does not provide any restrictions on functionality and the resulting
    serialized engine can be executed with TensorRT's standard runtime APIs in the nvinfer1 namespace.
    EngineCapability.SAFETY provides a restricted subset of network operations that are safety certified and
    the resulting serialized engine can be executed with TensorRT's safe runtime APIs in the `nvinfer1::safe` namespace.
    EngineCapability.DLA_STANDALONE provides a restricted subset of network operations that are DLA compatible and
    the resulting serialized engine can be executed using standalone DLA runtime APIs. See sampleCudla for an
    example of integrating cuDLA APIs with TensorRT APIs.)trtdoc";

constexpr char const* STANDARD
    = R"trtdoc(Standard: TensorRT flow without targeting the standard runtime. This flow supports both DeviceType::kGPU and DeviceType::kDLA.)trtdoc";

constexpr char const* SAFETY
    = R"trtdoc(Safety: TensorRT flow with restrictions targeting the safety runtime. See safety documentation for list of supported layers and formats. This flow supports only DeviceType::kGPU.)trtdoc";

constexpr char const* DLA_STANDALONE
    = R"trtdoc(DLA Standalone: TensorRT flow with restrictions targeting external, to TensorRT, DLA runtimes. See DLA documentation for list of supported layers and formats. This flow supports only DeviceType::kDLA.)trtdoc";

} // namespace EngineCapabilityDoc

namespace LayerInformationFormatDoc
{
constexpr char const* descr = R"trtdoc(The format in which the IEngineInspector prints the layer information.)trtdoc";
constexpr char const* ONELINE = R"trtdoc(Print layer information in one line per layer.)trtdoc";
constexpr char const* JSON = R"trtdoc(Print layer information in JSON format.)trtdoc";
} // namespace LayerInformationFormatDoc

namespace TimingCacheKeyDoc
{
constexpr char const* descr = R"trtdoc(
        The key to retrieve timing cache entries.
    )trtdoc";

constexpr char const* parse = R"trtdoc(
        Parse the text into a `TimingCacheKey` object.

        :arg text: The input text.

        :returns: A `TimingCacheKey` object.
    )trtdoc";

constexpr char const* convertTimingCacheKeyToString = R"trtdoc(
        Convert a `TimingCacheKey` object into text.

        :returns: A `str` object.
    )trtdoc";
} // namespace TimingCacheKeyDoc

namespace TimingCacheValueDoc
{
constexpr char const* descr = R"trtdoc(
        The values in the cache entry.
    )trtdoc";
}

namespace ITimingCacheDoc
{
constexpr char const* descr = R"trtdoc(
        Class to handle tactic timing info collected from builder.
    )trtdoc";

constexpr char const* serialize = R"trtdoc(
        Serialize a timing cache to a :class:`IHostMemory` object.

        :returns: An :class:`IHostMemory` object that contains a serialized timing cache.
    )trtdoc";

constexpr char const* combine = R"trtdoc(
        Combine input timing cache into local instance.

        Append entries in input cache to local cache. Conflicting entries will be skipped. The input
        cache must be generated by a TensorRT build of exact same version, otherwise combine will be
        skipped and return false. ``bool(ignore_mismatch) == True`` if combining a timing cache
        created from a different device.

        :arg input_cache: The input timing cache
        :arg ignore_mismatch: Whether or not to allow cache verification header mismatch

        :returns: A `bool` indicating whether the combine operation is done successfully.
    )trtdoc";

constexpr char const* reset = R"trtdoc(
        Empty the timing cache

        :returns: A `bool` indicating whether the reset operation is done successfully.
    )trtdoc";

constexpr char const* queryKeys = R"trtdoc(
    Query cache keys from Timing Cache.

    If an error occurs, a RuntimeError will be raised.

    :returns A list containing the cache keys.
    )trtdoc";

constexpr char const* query = R"trtdoc(
        Query value in a cache entry.

        If the key exists, write the value out, otherwise return an invalid value.

        :arg key: The query key.
        :cache cache: value if the key exists, otherwise an invalid value.

        :returns A :class:`TimingCacheValue` object.
    )trtdoc";

constexpr char const* update = R"trtdoc(
        Update values in a cache entry.

        Update the value of the given cache key. If the key does not exist, return False.
        If the key exists and the new tactic timing is NaN, delete the cache entry and
        return True. If tactic timing is not NaN and the new value is valid, override the
        cache value and return True. False is returned when the new value is invalid.
        If this layer cannot use the new tactic, build errors will be reported when
        building the next engine.

        :arg key: The key to the entry to be updated.
        :arg value: New cache value.

        :returns True if update succeeds, otherwise False.
    )trtdoc";

} // namespace ITimingCacheDoc

namespace IBuilderConfigDoc
{
constexpr char const* descr = R"trtdoc(

        :ivar avg_timing_iterations: :class:`int` The number of averaging iterations used when timing layers. When timing layers, the builder minimizes over a set of average times for layer execution. This parameter controls the number of iterations used in averaging. By default the number of averaging iterations is 1.
        :ivar int8_calibrator: :class:`IInt8Calibrator` [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization. Int8 Calibration interface. The calibrator is to minimize the information loss during the INT8 quantization process.
        :ivar flags: :class:`int` The build mode flags to turn on builder options for this network. The flags are listed in the BuilderFlags enum. The flags set configuration options to build the network. This should be in integer consisting of one or more :class:`BuilderFlag` s, combined via binary OR. For example, ``1 << BuilderFlag.FP16 | 1 << BuilderFlag.DEBUG``.
        :ivar profile_stream: :class:`int` The handle for the CUDA stream that is used to profile this network.
        :ivar num_optimization_profiles: :class:`int` The number of optimization profiles.
        :ivar default_device_type: :class:`tensorrt.DeviceType` The default DeviceType to be used by the Builder.
        :ivar DLA_core: :class:`int` The DLA core that the engine executes on. Must be between 0 and N-1 where N is the number of available DLA cores.
        :ivar profiling_verbosity: Profiling verbosity in NVTX annotations.
        :ivar engine_capability: The desired engine capability. See :class:`EngineCapability` for details.
        :ivar algorithm_selector: [DEPRECATED] Deprecated in TensorRT 10.8. Please use editable mode in ITimingCache instead. The :class:`IAlgorithmSelector` to use.
        :ivar builder_optimization_level: The builder optimization level which TensorRT should build the engine at. Setting a higher optimization level allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a lower optimization level. The default optimization level is 3. Valid values include integers from 0 to the maximum optimization level, which is currently 5. Setting it to be greater than the maximum level results in identical behavior to the maximum level.
        :ivar max_num_tactics: The maximum number of tactics to time when there is a choice of tactics. Setting a larger number allows TensorRT to spend longer engine building time searching for more optimization options. The resulting engine may have better performance compared to an engine built with a smaller number of tactics. Valid values include integers from -1 to the maximum 32-bit integer. Default value -1 indicates that TensorRT can decide the number of tactics based on its own heuristic.
        :ivar hardware_compatibility_level: Hardware compatibility allows an engine compatible with GPU architectures other than that of the GPU on which the engine was built.
        :ivar plugins_to_serialize: The plugin libraries to be serialized with forward-compatible engines.
        :ivar max_aux_streams: The maximum number of auxiliary streams that TRT is allowed to use. If the network contains operators that can run in parallel, TRT can execute them using auxiliary streams in addition to the one provided to the IExecutionContext::enqueueV3() call. The default maximum number of auxiliary streams is determined by the heuristics in TensorRT on whether enabling multi-stream would improve the performance. This behavior can be overridden by calling this API to set the maximum number of auxiliary streams explicitly. Set this to 0 to enforce single-stream inference. The resulting engine may use fewer auxiliary streams than the maximum if the network does not contain enough parallelism or if TensorRT determines that using more auxiliary streams does not help improve the performance. Allowing more auxiliary streams does not always give better performance since there will be synchronizations overhead between streams. Using CUDA graphs at runtime can help reduce the overhead caused by cross-stream synchronizations. Using more auxiliary leads to more memory usage at runtime since some activation memory blocks will not be able to be reused.
        :ivar progress_monitor: The :class:`IProgressMonitor` to use.
        :ivar tiling_optimization_level: The optimization level of tiling strategies. A Higher level allows TensorRT to spend more time searching for better optimization strategy.
        :ivar l2_limit_for_tiling: The target L2 cache usage for tiling optimization.
        :ivar remote_auto_tuning_config: The config string to be used during remote auto-tuning. Remote auto-tuning is only enabled for engines built with EngineCapability.SAFETY.

        Below are the descriptions about each builder optimization level:

        - Level 0: This enables the fastest compilation by disabling dynamic kernel generation and selecting the first tactic that succeeds in execution. This will also not respect a timing cache.
        - Level 1: Available tactics are sorted by heuristics, but only the top are tested to select the best. If a dynamic kernel is generated its compile optimization is low.
        - Level 2: Available tactics are sorted by heuristics, but only the fastest tactics are tested to select the best.
        - Level 3: Apply heuristics to see if a static precompiled kernel is applicable or if a new one has to be compiled dynamically.
        - Level 4: Always compiles a dynamic kernel.
        - Level 5: Always compiles a dynamic kernel and compares it to static kernels.
    )trtdoc";

constexpr char const* set_memory_pool_limit = R"trtdoc(
        Set the memory size for the memory pool.

        TensorRT layers access different memory pools depending on the operation.
        This function sets in the :class:`IBuilderConfig` the size limit, specified by pool_size, for the corresponding memory pool, specified by pool.
        TensorRT will build a plan file that is constrained by these limits or report which constraint caused the failure.

        If the size of the pool, specified by pool_size, fails to meet the size requirements for the pool,
        this function does nothing and emits the recoverable error, ErrorCode.INVALID_ARGUMENT, to the registered :class:`IErrorRecorder` .

        If the size of the pool is larger than the maximum possible value for the configuration,
        this function does nothing and emits ErrorCode.UNSUPPORTED_STATE.

        If the pool does not exist on the requested device type when building the network,
        a warning is emitted to the logger, and the memory pool value is ignored.

        Refer to MemoryPoolType to see the size requirements for each pool.

        :arg pool: The memory pool to limit the available memory for.
        :arg pool_size: The size of the pool in bytes.
    )trtdoc";

constexpr char const* get_memory_pool_limit = R"trtdoc(
        Retrieve the memory size limit of the corresponding pool in bytes.
        If :func:`set_memory_pool_limit` for the pool has not been called, this returns the default value used by TensorRT.
        This default value is not necessarily the maximum possible value for that configuration.

        :arg pool: The memory pool to get the limit for.

        :returns: The size of the memory limit, in bytes, for the corresponding pool.
    )trtdoc";

constexpr char const* num_compute_capabilities = R"trtdoc(
        :ivar num_compute_capabilities: The number of target compute capabilities for building the engine.
    )trtdoc";

constexpr char const* get_compute_capability = R"trtdoc(
        Get one target compute capability for building the engine.

        :arg index: The index of the compute capability to get. Must be between 0 and the number of compute capabilities - 1.

        :returns: The compute capability at the specified index.
    )trtdoc";

constexpr char const* set_compute_capability = R"trtdoc(
        Set one target compute capability for building the engine.

        :arg compute_capability: The compute capability to set. Cannot be kNONE. When set to kCURRENT, the index must be 0 and there must be only one compute capability.
        :arg index: The index at which to set the compute capability. Must be between 0 and the number of compute capabilities - 1.

        :returns: True if successful, false otherwise.
    )trtdoc";

constexpr char const* clear_flag = R"trtdoc(
        Clears the builder mode flag from the enabled flags.

        :arg flag: The flag to clear.
    )trtdoc";

constexpr char const* set_flag = R"trtdoc(
        Add the input builder mode flag to the already enabled flags.

        :arg flag: The flag to set.
    )trtdoc";

constexpr char const* get_flag = R"trtdoc(
        Check if a build mode flag is set.

        :arg flag: The flag to check.

        :returns: A `bool` indicating whether the flag is set.
    )trtdoc";

constexpr char const* clear_quantization_flag = R"trtdoc(
        Clears the quantization flag from the enabled quantization flags.

        :arg flag: The flag to clear.
    )trtdoc";

constexpr char const* set_quantization_flag = R"trtdoc(
        Add the input quantization flag to the already enabled quantization flags.

        :arg flag: The flag to set.
    )trtdoc";

constexpr char const* get_quantization_flag = R"trtdoc(
        Check if a quantization flag is set.

        :arg flag: The flag to check.

        :returns: A `bool` indicating whether the flag is set.
    )trtdoc";

constexpr char const* reset = R"trtdoc(
        Resets the builder configuration to defaults. When initializing a builder config object, we can call this function.
    )trtdoc";

constexpr char const* add_optimization_profile = R"trtdoc(
    Add an optimization profile.

    This function must be called at least once if the network has dynamic or shape input tensors.

    :arg profile: The new optimization profile, which must satisfy ``bool(profile) == True``

    :returns: The index of the optimization profile (starting from 0) if the input is valid, or -1 if the input is
                not valid.
)trtdoc";

constexpr char const* set_calibration_profile = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

    Set a calibration profile.

    Calibration optimization profile must be set if int8 calibration is used to set scales for a network with runtime dimensions.

    :arg profile: The new calibration profile, which must satisfy ``bool(profile) == True`` or be None. MIN and MAX values will be overwritten by OPT.

    :returns: True if the calibration profile was set correctly.
)trtdoc";

constexpr char const* get_calibration_profile = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

    Get the current calibration profile.

    :returns: The current calibration profile or None if calibrartion profile is unset.
)trtdoc";

constexpr char const* set_device_type = R"trtdoc(
    Set the device that this layer must execute on. If DeviceType is not set or is reset, TensorRT will use the
    default DeviceType set in the builder.

    The DeviceType for a layer must be compatible with the safety flow (if specified). For example a layer
    cannot be marked for DLA execution while the builder is configured for SAFE_GPU.


    :arg layer: The layer to set the DeviceType of
    :arg device_type: The DeviceType the layer must execute on
)trtdoc";

constexpr char const* get_device_type = R"trtdoc(
    Get the device that the layer executes on.

    :arg layer: The layer to get the DeviceType for

    :returns: The DeviceType of the layer
)trtdoc";

constexpr char const* is_device_type_set = R"trtdoc(
    Check if the DeviceType for a layer is explicitly set.

    :arg layer: The layer to check for DeviceType

    :returns: True if DeviceType is not default, False otherwise
)trtdoc";

constexpr char const* reset_device_type = R"trtdoc(
    Reset the DeviceType for the given layer.

    :arg layer: The layer to reset the DeviceType for
)trtdoc";

constexpr char const* can_run_on_DLA = R"trtdoc(
    Check if the layer can run on DLA.

    :arg layer: The layer to check

    :returns: A `bool` indicating whether the layer can run on DLA
)trtdoc";

constexpr char const* set_tactic_sources = R"trtdoc(
    Set tactic sources.

    This bitset controls which tactic sources TensorRT is allowed to use for tactic selection.

    Multiple tactic sources may be combined with a bitwise OR operation. For example,
    to enable cublas and cublasLt as tactic sources, use a value of:
    ``1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT)``

    :arg tactic_sources: The tactic sources to set

    :returns: A `bool` indicating whether the tactic sources in the build configuration were updated. The tactic sources in the build configuration will not be updated if the provided value is invalid.
)trtdoc";

constexpr char const* get_tactic_sources = R"trtdoc(
    Get the tactic sources currently set in the engine build configuration.
)trtdoc";

constexpr char const* create_timing_cache = R"trtdoc(
    Create timing cache

    Create :class:`ITimingCache` instance from serialized raw data. The created timing cache doesn't belong to
    a specific builder config. It can be shared by multiple builder instances

    :arg serialized_timing_cache: The serialized timing cache. If an empty cache is provided (i.e. ``b""``),  a new cache will be created.

    :returns: The created :class:`ITimingCache` object.
)trtdoc";

constexpr char const* set_timing_cache = R"trtdoc(
    Attach a timing cache to IBuilderConfig

    The timing cache has verification header to make sure the provided cache can be used in current environment.
    A failure will be reported if the CUDA device property in the provided cache is different from current environment.
    ``bool(ignore_mismatch) == True`` skips strict verification and allows loading cache created from a different device.
    The cache must not be destroyed until after the engine is built.

    :arg cache: The timing cache to be used
    :arg ignore_mismatch: Whether or not allow using a cache that contains different CUDA device property

    :returns: A `BOOL` indicating whether the operation is done successfully.
)trtdoc";

constexpr char const* get_timing_cache = R"trtdoc(
    Get the timing cache from current IBuilderConfig

    :returns: The timing cache used in current IBuilderConfig, or `None` if no timing cache is set.
)trtdoc";

constexpr char const* set_preview_feature = R"trtdoc(
    Enable or disable a specific preview feature.

    Allows enabling or disabling experimental features, which are not enabled by default in the current release.
    Preview Features have been fully tested but are not yet as stable as other features in TensorRT.
    They are provided as opt-in features for at least one release.

    Refer to PreviewFeature for additional information, and a list of the available features.

    :arg feature: the feature to enable
    :arg enable: whether to enable or disable
)trtdoc";

constexpr char const* get_preview_feature = R"trtdoc(
    Check if a preview feature is enabled.

    :arg feature: the feature to query

    :returns: true if the feature is enabled, false otherwise
)trtdoc";
} // namespace IBuilderConfigDoc

namespace SerializationFlagDoc
{
constexpr char const* descr = R"trtdoc(Valid flags that can be use to creating binary file from engine.)trtdoc";
constexpr char const* EXCLUDE_WEIGHTS = R"trtdoc(Exclude weights that can be refitted.)trtdoc";
constexpr char const* EXCLUDE_LEAN_RUNTIME = R"trtdoc(Exclude lean runtime from the plan.)trtdoc";
constexpr char const* INCLUDE_REFIT = R"trtdoc(Remain refittable if originally so.)trtdoc";
} // namespace SerializationFlagDoc

namespace ExecutionContextAllocationStrategyDoc
{
constexpr char const* descr = R"trtdoc(Different memory allocation behaviors for IExecutionContext.)trtdoc";
constexpr char const* STATIC = R"trtdoc(Default static allocation with the maximum size across all profiles.)trtdoc";
constexpr char const* ON_PROFILE_CHANGE = R"trtdoc(Reallocate for a profile when it's selected.)trtdoc";
constexpr char const* USER_MANAGED = R"trtdoc(The user supplies custom allocation to the execution context.)trtdoc";
} // namespace ExecutionContextAllocationStrategyDoc


namespace BuilderDoc
{
constexpr char const* descr = R"trtdoc(
    Builds an :class:`ICudaEngine` from a :class:`INetworkDefinition` .

    :ivar platform_has_tf32: :class:`bool` Whether the platform has tf32 support.
    :ivar platform_has_fast_fp16: :class:`bool` Whether the platform has fast native fp16.
    :ivar platform_has_fast_int8: :class:`bool` Whether the platform has fast native int8.
    :ivar max_DLA_batch_size: :class:`int` The maximum batch size DLA can support. For any tensor the total volume of index dimensions combined(dimensions other than CHW) with the requested batch size should not exceed the value returned by this function.
    :ivar num_DLA_cores: :class:`int` The number of DLA engines available to this builder.
    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
    :ivar gpu_allocator: :class:`IGpuAllocator` The GPU allocator to be used by the :class:`Builder` . All GPU
        memory acquired will use this allocator. If set to ``None``, the default allocator will be used.
    :ivar logger: :class:`ILogger` The logger provided when creating the refitter.
    :ivar max_threads: :class:`int` The maximum thread that can be used by the :class:`Builder`.
)trtdoc";

constexpr char const* init = R"trtdoc(
    :arg logger: The logger to use.
)trtdoc";

constexpr char const* create_network = R"trtdoc(
    Create a :class:`INetworkDefinition` object.

    :arg flags: :class:`NetworkDefinitionCreationFlag` s combined using bitwise OR.

    :returns: An empty TensorRT :class:`INetworkDefinition` .
)trtdoc";

constexpr char const* create_optimization_profile = R"trtdoc(
    Create a new optimization profile.

    If the network has any dynamic input tensors, the appropriate calls to :func:`IOptimizationProfile.set_shape` must be made. Likewise, if there are any shape input tensors, the appropriate calls to :func:`IOptimizationProfile.set_shape_input` are required.

    See :class:`IOptimizationProfile`
)trtdoc";

constexpr char const* create_builder_config = R"trtdoc(
    Create a builder configuration object.

    See :class:`IBuilderConfig`
)trtdoc";

constexpr char const* build_serialized_network = R"trtdoc(
    Builds and serializes a network for the given :class:`INetworkDefinition` and :class:`IBuilderConfig` .

    This function allows building and serialization of a network without creating an engine.

    :arg network: Network definition.
    :arg config: Builder configuration.

    :returns: A pointer to a :class:`IHostMemory` object that contains a serialized network.
)trtdoc";

constexpr char const* build_serialized_network_to_stream = R"trtdoc(
    Builds and serializes a network for the given :class:`INetworkDefinition` and :class:`IBuilderConfig` and write the serialized network to a writer stream.

    This function allows building and serialization of a network without creating an engine.

    :arg network: Network definition.
    :arg config: Builder configuration.
    :arg writer: Output stream writer.

    :returns: True if build succeed, otherwise False.
)trtdoc";

constexpr char const* build_engine_with_config = R"trtdoc(
    Builds a network for the given :class:`INetworkDefinition` and :class:`IBuilderConfig` .

    This function allows building a network and creating an engine.

    :arg network: Network definition.
    :arg config: Builder configuration.

    :returns: A pointer to a :class:`ICudaEngine` object that contains a built engine.
)trtdoc";

constexpr char const* is_network_supported = R"trtdoc(
    Checks that a network is within the scope of the :class:`IBuilderConfig` settings.

    :arg network: The network definition to check for configuration compliance.
    :arg config: The configuration of the builder to use when checking the network.

    Given an :class:`INetworkDefinition` and an :class:`IBuilderConfig` , check if
    the network falls within the constraints of the builder configuration based on the
    :class:`EngineCapability` , :class:`BuilderFlag` , and :class:`DeviceType` .

    :returns: ``True`` if network is within the scope of the restrictions specified by the builder config, ``False`` otherwise.
        This function reports the conditions that are violated to the registered :class:`ErrorRecorder` .

    NOTE: This function will synchronize the cuda stream returned by ``config.profile_stream`` before returning.

)trtdoc";

constexpr char const* reset = R"trtdoc(
    Resets the builder state to default values.
)trtdoc";

constexpr char const* get_plugin_registry = R"trtdoc(
    Get the local plugin registry that can be used by the builder.

    :returns: The local plugin registry that can be used by the builder.
)trtdoc";
} // namespace BuilderDoc

namespace RuntimeDoc
{
constexpr char const* descr = R"trtdoc(
    Allows a serialized :class:`ICudaEngine` to be deserialized.

    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
    :ivar gpu_allocator: :class:`IGpuAllocator` The GPU allocator to be used by the :class:`Runtime` . All GPU memory
        acquired will use this allocator. If set to None, the default allocator will be used (Default: cudaMalloc/cudaFree).
    :ivar DLA_core: :class:`int` The DLA core that the engine executes on. Must be between 0 and N-1 where N is the number of available DLA cores.
    :ivar num_DLA_cores: :class:`int` The number of DLA engines available to this builder.
    :ivar logger: :class:`ILogger` The logger provided when creating the refitter.
    :ivar max_threads: :class:`int` The maximum thread that can be used by the :class:`Runtime`.
    :ivar temporary_directory: :class:`str` The temporary directory to use when loading executable code for engines.  If set to None (the default), TensorRT will
                                            attempt to find a suitable directory for use using platform-specific heuristics:
                                            - On UNIX/Linux platforms, TensorRT will first try the TMPDIR environment variable, then fall back to /tmp
                                            - On Windows, TensorRT will try the TEMP environment variable.
    :ivar tempfile_control_flags: :class:`int` Flags which control whether TensorRT is allowed to create in-memory or temporary files.
                                               See :class:`TempfileControlFlag` for details.
    :ivar engine_host_code_allowed: :class:`bool` Whether this runtime is allowed to deserialize engines that contain host executable code (Default: False).

)trtdoc";

constexpr char const* init = R"trtdoc(
    :arg logger: The logger to use.
)trtdoc";

constexpr char const* deserialize_cuda_engine = R"trtdoc(
    Deserialize an :class:`ICudaEngine` from host memory.

    :arg serialized_engine: The :class:`buffer` that holds the serialized :class:`ICudaEngine`.

    :returns: The :class:`ICudaEngine`, or None if it could not be deserialized.
)trtdoc";

constexpr char const* deserialize_cuda_engine_reader = R"trtdoc(
    Deserialize an :class:`ICudaEngine` from a stream reader.

    :arg stream_reader: The :class:`PyStreamReader` that will read the serialized :class:`ICudaEngine`. This enables
        deserialization from a file directly.

    :returns: The :class:`ICudaEngine`, or None if it could not be deserialized.
)trtdoc";

constexpr char const* deserialize_cuda_engine_reader_v2 = R"trtdoc(
    Deserialize an :class:`ICudaEngine` from a stream reader v2.

    :arg stream_reader: The :class:`PyStreamReaderV2` that will read the serialized :class:`ICudaEngine`. This
        enables deserialization from a file directly, with possible benefits to performance.

    :returns: The :class:`ICudaEngine`, or None if it could not be deserialized.
)trtdoc";

constexpr char const* get_plugin_registry = R"trtdoc(
    Get the local plugin registry that can be used by the runtime.

    :returns: The local plugin registry that can be used by the runtime.
)trtdoc";

constexpr char const* load_runtime = R"trtdoc(
    Load IRuntime from the file.

    This method loads a runtime library from a shared library file. The runtime can
    then be used to execute a plan file built with BuilderFlag.VERSION_COMPATIBLE
    and BuilderFlag.EXCLUDE_LEAN_RUNTIME both set and built with the same version
    of TensorRT as the loaded runtime library.

    :ivar path: Path to the runtime lean library.

    :returns: The :class:`IRuntime`, or None if it could not be loaded.
)trtdoc";

} // namespace RuntimeDoc

namespace RuntimeInspectorDoc
{
constexpr char const* descr = R"trtdoc(
    An engine inspector which prints out the layer information of an engine or an execution context.

    The amount of printed information depends on the profiling verbosity setting of the builder config when the engine is built.
    By default, the profiling verbosity is set to ProfilingVerbosity.LAYER_NAMES_ONLY, and only layer names will be printed.
    If the profiling verbosity is set to ProfilingVerbosity.DETAILED, layer names and layer parameters will be printed.
    If the profiling verbosity is set to ProfilingVerbosity.NONE, no layer information will be printed.

    :ivar execution_context: :class:`IExecutionContext` Set or get context currently being inspected.
    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
)trtdoc";

constexpr char const* get_layer_information = R"trtdoc(
    Get a string describing the information about a specific layer in the current engine or the execution context.

    :arg layer_index: The index of the layer. It must lie in [0, engine.num_layers].
    :arg format: :class:`LayerInformationFormat` The format the layer information should be printed in.

    :returns: A string describing the information about a specific layer in the current engine or the execution context.
)trtdoc";

constexpr char const* get_engine_information = R"trtdoc(
    Get a string describing the information about all the layers in the current engine or the execution context.

    :arg format: :class:`LayerInformationFormat` The format the layer information should be printed in.

    :returns: A string describing the information about all the layers in the current engine or the execution context.
)trtdoc";

constexpr char const* clear_inspection_source = R"trtdoc(
    Clear the inspection srouce.

    :returns: A boolean indicating whether the action succeeds.
)trtdoc";

} // namespace RuntimeInspectorDoc

namespace TensorLocationDoc
{
constexpr char const* descr = R"trtdoc(The physical location of the data.)trtdoc";

constexpr char const* DEVICE = R"trtdoc(Data is stored on the device.)trtdoc";
constexpr char const* HOST = R"trtdoc(Data is stored on the host.)trtdoc";
} // namespace TensorLocationDoc

namespace RefitterDoc
{
constexpr char const* descr = R"trtdoc(
    Updates weights in an :class:`ICudaEngine` .

    :ivar error_recorder: :class:`IErrorRecorder` Application-implemented error reporting interface for TensorRT objects.
    :ivar logger: :class:`ILogger` The logger provided when creating the refitter.
    :ivar max_threads: :class:`int` The maximum thread that can be used by the :class:`Refitter`.
    :ivar weights_validation: :class:`bool` The flag to indicate whether to validate weights in the refitting process.
)trtdoc";

constexpr char const* init = R"trtdoc(
    :arg engine: The engine to refit.
    :arg logger: The logger to use.
)trtdoc";

constexpr char const* set_weights = R"trtdoc(
    Specify new weights for a layer of given name.
    Possible reasons for rejection are:

    * There is no such layer by that name.
    * The layer does not have weights with the specified role.
    * The size of weights is inconsistent with the layer’s original specification.

    Modifying the weights before :func:`refit_cuda_engine` or :func:`refit_cuda_engine_async` returns
    will result in undefined behavior.

    :arg layer_name: The name of the layer.
    :arg role: The role of the weights. See :class:`WeightsRole` for more information.
    :arg weights: The weights to refit with.

    :returns: ``True`` on success, or ``False`` if new weights are rejected.
)trtdoc";

constexpr char const* set_named_weights = R"trtdoc(
    Specify new weights of given name.
    Possible reasons for rejection are:

    * The name of weights is empty or does not correspond to any refittable weights.
    * The size of the weights is inconsistent with the size returned from calling :func:`get_weights_prototype` with the same name.
    * The dtype of the weights is inconsistent with the dtype returned from calling :func:`get_weights_prototype` with the same name.

    Modifying the weights before :func:`refit_cuda_engine` or :func:`refit_cuda_engine_async` returns
    will result in undefined behavior.

    :arg name: The name of the weights to be refitted.
    :arg weights: The new weights to associate with the name.

    :returns: ``True`` on success, or ``False`` if new weights are rejected.
)trtdoc";

constexpr char const* set_named_weights_with_location = R"trtdoc(
    Specify new weights on a specified device of given name.
    Possible reasons for rejection are:

    * The name of weights is empty or does not correspond to any refittable weights.
    * The size of the weights is inconsistent with the size returned from calling :func:`get_weights_prototype` with the same name.
    * The dtype of the weights is inconsistent with the dtype returned from calling :func:`get_weights_prototype` with the same name.

    It is allowed to provide some weights on CPU and others on GPU.
    Modifying the weights before :func:`refit_cuda_engine` or :func:`refit_cuda_engine_async` returns
    will result in undefined behavior.

    :arg name: The name of the weights to be refitted.
    :arg weights: The new weights on the specified device.
    :arg location: The location (host vs. device) of the new weights.

    :returns: ``True`` on success, or ``False`` if new weights are rejected.
)trtdoc";

constexpr char const* get_named_weights = R"trtdoc(
    Get weights associated with the given name.

    If the weights were never set, returns null weights and reports an error to the refitter errorRecorder.

    :arg weights_name: The name of the weights to be refitted.

    :returns: Weights associated with the given name.
)trtdoc";

constexpr char const* get_weights_location = R"trtdoc(
    Get location for the weights associated with the given name.

    If the weights were never set, returns TensorLocation.HOST and reports an error to the refitter errorRecorder.

    :arg weights_name: The name of the weights to be refitted.

    :returns: Location for the weights associated with the given name.
)trtdoc";

constexpr char const* get_weights_prototype = R"trtdoc(
    Get the weights prototype associated with the given name.

    The dtype and size of weights prototype is the same as weights used for engine building.
    The size of the weights prototype is -1 when the name of the weights is None or does not correspond to any refittable weights.

    :arg weights_name: The name of the weights to be refitted.

    :returns: weights prototype associated with the given name.
)trtdoc";

constexpr char const* unset_named_weights = R"trtdoc(
    Unset weights associated with the given name.

    Unset weights before releasing them.

    :arg weights_name: The name of the weights to be refitted.

    :returns: ``False`` if the weights were never set, returns ``True`` otherwise.
)trtdoc";

constexpr char const* refit_cuda_engine = R"trtdoc(
   Refits associated engine.

   If ``False`` is returned, a subset of weights may have been refitted.

   The behavior is undefined if the engine has pending enqueued work.
   Provided weights on CPU or GPU can be unset and released, or updated after refit_cuda_engine returns.

   IExecutionContexts associated with the engine remain valid for use afterwards. There is no need to set the same
   weights repeatedly for multiple refit calls as the weights memory can be updated directly instead.

   :returns: ``True`` on success, or ``False`` if new weights validation fails or get_missing_weights() != 0 before the call.
)trtdoc";

constexpr char const* refit_cuda_engine_async = R"trtdoc(
    Enqueue weights refitting of the associated engine on the given stream.

    If ``False`` is returned, a subset of weights may have been refitted.

    The behavior is undefined if the engine has pending enqueued work on a different stream from the provided one.
    Provided weights on CPU can be unset and released, or updated after refit_cuda_engine_async returns.
    Freeing or updating of the provided weights on GPU can be enqueued on the same stream after refit_cuda_engine_async returns.

    IExecutionContexts associated with the engine remain valid for use afterwards. There is no need to set the same
    weights repeatedly for multiple refit calls as the weights memory can be updated directly instead. The weights
    updating task should use the the same stream as the one used for the refit call.

    :arg stream: The stream to enqueue the weights updating task.

    :returns: ``True`` on success, or ``False`` if new weights validation fails or get_missing_weights() != 0 before the call.
)trtdoc";

constexpr char const* get_missing = R"trtdoc(
    Get description of missing weights.

    For example, if some Weights have been set, but the engine was optimized
    in a way that combines weights, any unsupplied Weights in the combination
    are considered missing.

    :returns: The names of layers with missing weights, and the roles of those weights.
)trtdoc";

constexpr char const* get_missing_weights = R"trtdoc(
    Get names of missing weights.

    For example, if some Weights have been set, but the engine was optimized
    in a way that combines weights, any unsupplied Weights in the combination
    are considered missing.

    :returns: The names of missing weights, empty string for unnamed weights.
)trtdoc";

constexpr char const* get_all = R"trtdoc(
    Get description of all weights that could be refitted.

    :returns: The names of layers with refittable weights, and the roles of those weights.
)trtdoc";

constexpr char const* get_all_weights = R"trtdoc(
    Get names of all weights that could be refitted.

    :returns: The names of refittable weights.
)trtdoc";

constexpr char const* get_dynamic_range = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

    Gets the dynamic range of a tensor. If the dynamic range was never set, returns the range computed during calibration.

    :arg tensor_name: The name of the tensor whose dynamic range to retrieve.

    :returns: :class:`Tuple[float, float]` A tuple containing the [minimum, maximum] of the dynamic range.
)trtdoc";

constexpr char const* set_dynamic_range = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

    Update dynamic range for a tensor.

    :arg tensor_name: The name of the tensor whose dynamic range to update.
    :arg range: The new range.

    :returns: :class:`True` if successful, :class:`False` otherwise.

    Returns false if there is no Int8 engine tensor derived from a network tensor of that name.  If successful, then :func:`get_missing` may report that some weights need to be supplied.
)trtdoc";

constexpr char const* get_tensors_with_dynamic_range = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.1. Superseded by explicit quantization.

    Get names of all tensors that have refittable dynamic ranges.

    :returns: The names of tensors with refittable dynamic ranges.
)trtdoc";

} // namespace RefitterDoc

namespace AllocatorFlagDoc
{
constexpr char const* descr = R"trtdoc()trtdoc";

constexpr char const* RESIZABLE = R"trtdoc(TensorRT may call realloc() on this allocation)trtdoc";
} // namespace AllocatorFlagDoc

namespace GpuAllocatorDoc
{
constexpr char const* descr = R"trtdoc(Application-implemented class for controlling allocation on the GPU.

To implement a custom allocator, ensure that you explicitly instantiate the base class in :func:`__init__` :
::

    class MyAllocator(trt.IGpuAllocator):
        def __init__(self):
            trt.IGpuAllocator.__init__(self)

        ...

Note that all methods below (allocate, reallocate, deallocate, allocate_async, deallocate_async) must be overridden in the custom allocator, or else pybind11 would not be able to call the method from a custom allocator.
)trtdoc";

constexpr char const* allocate = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.0. Please use allocate_async instead.
    A callback implemented by the application to handle acquisition of GPU memory.
    If an allocation request of size 0 is made, ``None`` should be returned.

    If an allocation request cannot be satisfied, ``None`` should be returned.

    :arg size: The size of the memory required.
    :arg alignment: The required alignment of memory. Alignment will be zero
        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
        An alignment value of zero indicates any alignment is acceptable.
    :arg flags: Allocation flags. See :class:`AllocatorFlag`

    :returns: The address of the allocated memory
)trtdoc";

constexpr char const* reallocate = R"trtdoc(
    A callback implemented by the application to resize an existing allocation.

    Only allocations which were allocated with AllocatorFlag.RESIZABLE will be resized.

    Options are one of:
    - resize in place leaving min(old_size, new_size) bytes unchanged and return the original address
    - move min(old_size, new_size) bytes to a new location of sufficient size and return its address
    - return None, to indicate that the request could not be fulfilled.

    If None is returned, TensorRT will assume that resize() is not implemented, and that the
    allocation at address is still valid.

    This method is made available for use cases where delegating the resize
    strategy to the application provides an opportunity to improve memory management.
    One possible implementation is to allocate a large virtual device buffer and
    progressively commit physical memory with cuMemMap. CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    is suggested in this case.

    TensorRT may call realloc to increase the buffer by relatively small amounts.

    :arg address: the address of the original allocation.
    :arg alignment: The alignment used by the original allocation.
    :arg new_size: The new memory size required.

    :returns: The address of the reallocated memory
)trtdoc";

constexpr char const* deallocate = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.0. Please use dealocate_async instead;
    A callback implemented by the application to handle release of GPU memory.

    TensorRT may pass a 0 to this function if it was previously returned by ``allocate()``.

    :arg memory: The memory address of the memory to release.

    :returns: True if the acquired memory is released successfully.
)trtdoc";

constexpr char const* allocate_async = R"trtdoc(
    A callback implemented by the application to handle acquisition of GPU memory asynchronously.
    This is just a wrapper around a syncronous method allocate.
    For the asynchronous allocation please use the corresponding IGpuAsyncAllocator class.
    If an allocation request of size 0 is made, ``None`` should be returned.

    If an allocation request cannot be satisfied, ``None`` should be returned.

    :arg size: The size of the memory required.
    :arg alignment: The required alignment of memory. Alignment will be zero
        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
        An alignment value of zero indicates any alignment is acceptable.
    :arg flags: Allocation flags. See :class:`AllocatorFlag`
    :arg stream: CUDA stream

    :returns: The address of the allocated memory
)trtdoc";

constexpr char const* deallocate_async = R"trtdoc(
    A callback implemented by the application to handle release of GPU memory asynchronously.
    This is just a wrapper around a syncronous method deallocate.
    For the asynchronous deallocation please use the corresponding IGpuAsyncAllocator class.

    TensorRT may pass a 0 to this function if it was previously returned by ``allocate()``.

    :arg memory: The memory address of the memory to release.
    :arg stream: CUDA stream

    :returns: True if the acquired memory is released successfully.
)trtdoc";

} // namespace GpuAllocatorDoc

namespace GpuAsyncAllocatorDoc
{
constexpr char const* descr = R"trtdoc(Application-implemented class for controlling allocation on the GPU.

To implement a custom allocator, ensure that you explicitly instantiate the base class in :func:`__init__` :
::

    class MyAllocator(trt.IGpuAsyncAllocator):
        def __init__(self):
            trt.IGpuAllocator.__init__(self)

        ...

Note that all methods below (allocate, reallocate, deallocate, allocate_async, reallocate_async, deallocate_async) must be overridden in the custom allocator, or else pybind11 would not be able to call the method from a custom allocator.
)trtdoc";

constexpr char const* allocate = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.0. Please use allocate_async instead.
    A callback implemented by the application to handle acquisition of GPU memory.
    This is just a wrapper around a synchronous method allocate_async passing the default stream.

    If an allocation request of size 0 is made, ``None`` should be returned.

    If an allocation request cannot be satisfied, ``None`` should be returned.

    :arg size: The size of the memory required.
    :arg alignment: The required alignment of memory. Alignment will be zero
        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
        An alignment value of zero indicates any alignment is acceptable.
    :arg flags: Allocation flags. See :class:`AllocatorFlag`

    :returns: The address of the allocated memory
)trtdoc";

constexpr char const* deallocate = R"trtdoc(
    [DEPRECATED] Deprecated in TensorRT 10.0. Please use deallocate_async instead.
    A callback implemented by the application to handle release of GPU memory.
    This is just a wrapper around a synchronous method deallocate_async passing the default stream.

    TensorRT may pass a 0 to this function if it was previously returned by ``allocate()``.

    :arg memory: The memory address of the memory to release.

    :returns: True if the acquired memory is released successfully.
)trtdoc";

constexpr char const* allocate_async = R"trtdoc(
    A callback implemented by the application to handle acquisition of GPU memory asynchronously.
    If an allocation request of size 0 is made, ``None`` should be returned.

    If an allocation request cannot be satisfied, ``None`` should be returned.

    :arg size: The size of the memory required.
    :arg alignment: The required alignment of memory. Alignment will be zero
        or a power of 2 not exceeding the alignment guaranteed by cudaMalloc.
        Thus this allocator can be safely implemented with cudaMalloc/cudaFree.
        An alignment value of zero indicates any alignment is acceptable.
    :arg flags: Allocation flags. See :class:`AllocatorFlag`
    :arg stream: CUDA stream

    :returns: The address of the allocated memory
)trtdoc";

constexpr char const* deallocate_async = R"trtdoc(
    A callback implemented by the application to handle release of GPU memory asynchronously.

    TensorRT may pass a 0 to this function if it was previously returned by ``allocate()``.

    :arg memory: The memory address of the memory to release.
    :arg stream: CUDA stream

    :returns: True if the acquired memory is released successfully.
)trtdoc";

} // namespace GpuAsyncAllocatorDoc

} // namespace tensorrt
