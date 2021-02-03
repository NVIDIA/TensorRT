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

// This file contains all Core docstrings, since these are typically too long to keep in the binding code.
#pragma once

namespace tensorrt
{
    namespace LoggerDoc
    {
        constexpr const char* descr = R"trtdoc(
            Logger for the :class:`Builder`, :class:`ICudaEngine` and :class:`Runtime` .

            :arg min_severity: The initial minimum severity of this Logger.

            :ivar min_severity: :class:`Logger.Severity` This minimum required severity of messages for the logger to log them.

            Note that although a logger is passed on creation to each instance of a :class:`Builder` or :class:`Runtime` interface, the logger is internally considered a singleton, and thus
            multiple instances of :class:`Runtime` and/or :class:`Builder` must all use the same logger.
        )trtdoc";

        constexpr const char* log = R"trtdoc(
            Logs a message to `stderr` .

            :arg severity: The severity of the message.
            :arg msg: The log message.

            Derived classes should generally overload this function.
        )trtdoc";
    } /* LoggerDoc */

    namespace SeverityDoc
    {
        constexpr const char* internal_error = R"trtdoc(
            Represents an internal error. Execution is unrecoverable.
        )trtdoc";

        constexpr const char* error = R"trtdoc(
            Represents an application error.
        )trtdoc";

        constexpr const char* warning = R"trtdoc(
            Represents an application error that TensorRT has recovered from or fallen back to a default.
        )trtdoc";

        constexpr const char* info = R"trtdoc(
            Represents informational messages.
        )trtdoc";

        constexpr const char* verbose = R"trtdoc(
            Verbose messages with debugging information.
        )trtdoc";
    } /* SeverityDoc */

    namespace ProfilerDoc
    {
        constexpr const char* descr = R"trtdoc(
            When this class is added to an :class:`IExecutionContext`, the profiler will be called once per layer for each invocation of :func:`IExecutionContext.execute()` .
            Note that :func:`IExecutionContext.execute_async()` does not currently support profiling.

            The profiler will only be called after execution is complete. It has a small impact on execution time.
        )trtdoc";

        constexpr const char* report_layer_time = R"trtdoc(
            Reports time in milliseconds for each layer. This function should be overloaded by classes derived from IProfiler.

            :arg layer_name: The name of the layer, set when constructing the :class:`INetworkDefinition` .
            :arg ms: The time in milliseconds to execute the layer.
        )trtdoc";
    } /* ProfilerDoc */

    namespace IOptimizationProfileDoc
    {
    constexpr const char* descr = R"trtdoc(
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

    constexpr const char* set_shape = R"trtdoc(
            Set the minimum/optimum/maximum dimensions for a dynamic input tensor.

            This function must be called for any network input tensor that has dynamic dimensions. If ``min``, ``opt``, and ``max`` are the minimum, optimum, and maximum dimensions, and ``real_shape`` is the shape for this input tensor provided to the :class:`INetworkDefinition` ,then the following conditions must hold:

            (1) ``len(min)`` == ``len(opt)`` == ``len(max)`` == ``len(real_shape)``
            (2) 1 <= ``min[i]`` <= ``opt[i]`` <= ``max[i]`` for all ``i``
            (3) if ``real_shape[i]`` != -1, then ``min[i]`` == ``opt[i]`` == ``max[i]`` == ``real_shape[i]``

            This function may (but need not be) called for an input tensor that does not have dynamic dimensions. In this
            case, all shapes must equal ``real_shape``.

            :arg input: The name of the input tensor.
            :arg min: The minimum dimensions for this input tensor.
            :arg opt: The optimum dimensions for this input tensor.
            :arg max: The maximum dimensions for this input tensor.

            :raises: :class:`ValueError` if an inconsistency was detected. Note that inputs can be validated only partially; a full validation is performed at engine build time.
        )trtdoc";

    constexpr const char* get_shape = R"trtdoc(
            Get the minimum/optimum/maximum dimensions for a dynamic input tensor.
            If the dimensions have not been previously set via :func:`set_shape`, return an invalid :class:`Dims` with a length of -1.

            :returns: A ``List[Dims]`` of length 3, containing the minimum, optimum, and maximum shapes, in that order. If the shapes have not been set yet, an empty list is returned.
        )trtdoc";

    constexpr const char* set_shape_input = R"trtdoc(
            Set the minimum/optimum/maximum values for a shape input tensor.

            This function must be called for every input tensor ``t`` that is a shape tensor (``t.is_shape`` == ``True``).
            This implies that the datatype of ``t`` is ``int32``, the rank is either 0 or 1, and the dimensions of ``t``
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

    constexpr const char* get_shape_input = R"trtdoc(
            Get the minimum/optimum/maximum values for a shape input tensor.

            :returns: A ``List[List[int]]`` of length 3, containing the minimum, optimum, and maximum values, in that order. If the values have not been set yet, an empty list is returned.
        )trtdoc";
    } // IOptimizationProfileDoc

    namespace ErrorCodeDoc
    {
    constexpr const char* descr = R"trtdoc(Error codes that can be returned by TensorRT during execution.)trtdoc";

    constexpr const char* SUCCESS = R"trtdoc(Execution completed successfully.)trtdoc";

    constexpr const char* UNSPECIFIED_ERROR = R"trtdoc(
            An error that does not fall into any other category. This error is included for forward compatibility.
        )trtdoc";

    constexpr const char* INTERNAL_ERROR = R"trtdoc(A non-recoverable TensorRT error occurred.)trtdoc";

    constexpr const char* INVALID_ARGUMENT = R"trtdoc(
            An argument passed to the function is invalid in isolation. This is a violation of the API contract.
        )trtdoc";

    constexpr const char* INVALID_CONFIG = R"trtdoc(
            An error occurred when comparing the state of an argument relative to other arguments. For example, the
            dimensions for concat differ between two tensors outside of the channel dimension. This error is triggered
            when an argument is correct in isolation, but not relative to other arguments. This is to help to distinguish
            from the simple errors from the more complex errors.
            This is a violation of the API contract.
        )trtdoc";

    constexpr const char* FAILED_ALLOCATION = R"trtdoc(
            An error occurred when performing an allocation of memory on the host or the device.
            A memory allocation error is normally fatal, but in the case where the application provided its own memory
            allocation routine, it is possible to increase the pool of available memory and resume execution.
        )trtdoc";

    constexpr const char* FAILED_INITIALIZATION = R"trtdoc(
            One, or more, of the components that TensorRT relies on did not initialize correctly.
            This is a system setup issue.
        )trtdoc";

    constexpr const char* FAILED_EXECUTION = R"trtdoc(
            An error occurred during execution that caused TensorRT to end prematurely, either an asynchronous error or
            other execution errors reported by CUDA/DLA. In a dynamic system, the
            data can be thrown away and the next frame can be processed or execution can be retried.
            This is either an execution error or a memory error.
        )trtdoc";

    constexpr const char* FAILED_COMPUTATION = R"trtdoc(
            An error occurred during execution that caused the data to become corrupted, but execution finished. Examples
            of this error are NaN squashing or integer overflow. In a dynamic system, the data can be thrown away and the
            next frame can be processed or execution can be retried.
            This is either a data corruption error, an input error, or a range error.
        )trtdoc";

    constexpr const char* INVALID_STATE = R"trtdoc(
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

    constexpr const char* UNSUPPORTED_STATE = R"trtdoc(
            An error occurred due to the network not being supported on the device due to constraints of the hardware or
            system. An example is running a unsafe layer in a safety certified context, or a resource requirement for the
            current network is greater than the capabilities of the target device. The network is otherwise correct, but
            the network and hardware combination is problematic. This can be recoverable.
            Examples:
            * Scratch space requests larger than available device memory and can be recovered by increasing allowed workspace size.
            * Tensor size exceeds the maximum element count and can be recovered by reducing the maximum batch size.
        )trtdoc";
    } // ErrorCodeDoc

    namespace IErrorRecorderDoc
    {
    constexpr const char* descr = R"trtdoc(
            Reference counted application-implemented error reporting interface for TensorRT objects.

            The error reporting mechanism is a user defined object that interacts with the internal state of the object
            that it is assigned to in order to determine information about abnormalities in execution. The error recorder
            gets both an error enum that is more descriptive than pass/fail and also a description that gives more
            detail on the exact failure modes. In the safety context, the error strings are all limited to 128 characters
            in length.
            The ErrorRecorder gets passed along to any class that is created from another class that has an ErrorRecorder
            assigned to it. For example, assigning an ErrorRecorder to an IBuilder allows all INetwork's, ILayer's, and
            ITensor's to use the same error recorder. For functions that have their own ErrorRecorder accessor functions.
            This allows registering a different error recorder or de-registering of the error recorder for that specific
            object.

            The ErrorRecorder object implementation must be thread safe if the same ErrorRecorder is passed to different
            interface objects being executed in parallel in different threads. All locking and synchronization is
            pushed to the interface implementation and TensorRT does not hold any synchronization primitives when accessing
            the interface functions.
        )trtdoc";

    constexpr const char* has_overflowed = R"trtdoc(
            Determine if the error stack has overflowed.

            In the case when the number of errors is large, this function is used to query if one or more
            errors have been dropped due to lack of storage capacity. This is especially important in the
            automotive safety case where the internal error handling mechanisms cannot allocate memory.

            :returns: True if errors have been dropped due to overflowing the error stack.
        )trtdoc";

    constexpr const char* get_num_errors = R"trtdoc(
            Return the number of errors

            Determines the number of errors that occurred between the current point in execution
            and the last time that the clear() was executed. Due to the possibility of asynchronous
            errors occuring, a TensorRT API can return correct results, but still register errors
            with the Error Recorder. The value of getNbErrors must monotonically increases until clear()
            is called.

            :returns: Returns the number of errors detected, or 0 if there are no errors.
        )trtdoc";

    constexpr const char* get_error_code = R"trtdoc(
            Returns the ErrorCode enumeration.

            The error_idx specifies what error code from 0 to :attr:`num_errors`-1 that the application
            wants to analyze and return the error code enum.

            :arg error_idx: A 32bit integer that indexes into the error array.

            :returns: Returns the enum corresponding to error_idx.
        )trtdoc";

    constexpr const char* get_error_desc = R"trtdoc(
            Returns description of the error.

            For the error specified by the idx value, return description of the error. In the safety context there is a
            constant length requirement to remove any dynamic memory allocations and the error message
            may be truncated. The format of the error description is "<EnumAsStr> - <Description>".

            :arg error_idx: A 32bit integer that indexes into the error array.

            :returns: Returns description of the error.
        )trtdoc";

    constexpr const char* clear = R"trtdoc(
            Clear the error stack on the error recorder.

            Removes all the tracked errors by the error recorder.  This function must guarantee that after
            this function is called, and as long as no error occurs, :attr:`num_errors` will be zero.
        )trtdoc";

    constexpr const char* report_error = R"trtdoc(
            Clear the error stack on the error recorder.

            Report an error to the user that has a given value and human readable description. The function returns false
            if processing can continue, which implies that the reported error is not fatal. This does not guarantee that
            processing continues, but provides a hint to TensorRT.

            :arg val: The error code enum that is being reported.
            :arg desc: The description of the error.

            :returns: True if the error is determined to be fatal and processing of the current function must end.
        )trtdoc";
    } // IErrorRecorderDoc

    namespace IExecutionContextDoc
    {
        constexpr const char* descr = R"trtdoc(
            Context for executing inference using an :class:`ICudaEngine` .
            Multiple :class:`IExecutionContext` s may exist for one :class:`ICudaEngine` instance, allowing the same
            :class:`ICudaEngine` to be used for the execution of multiple batches simultaneously.

            :ivar debug_sync: :class:`bool` The debug sync flag. If this flag is set to true, the :class:`ICudaEngine` will log the successful execution for each kernel during execute(). It has no effect when using execute_async().
            :ivar profiler: :class:`IProfiler` The profiler in use by this :class:`IExecutionContext` .
            :ivar engine: :class:`ICudaEngine` The associated :class:`ICudaEngine` .
            :ivar name: :class:`str` The name of the :class:`IExecutionContext` .
            :ivar device_memory: :class:`capsule` The device memory for use by this execution context. The memory must be aligned on a 256-byte boundary, and its size must be at least :attr:`engine.device_memory_size`. If using :func:`execute_async` to run the network, The memory is in use from the invocation of :func:`execute_async` until network execution is complete. If using :func:`execute`, it is in use until :func:`execute` returns. Releasing or otherwise using the memory for other purposes during this time will result in undefined behavior.
            :ivar active_optimization_profile: :class:`int` The active optimization profile for the context. The selected profile will be used in subsequent calls to :func:`execute` or :func:`execute_async` . Profile 0 is selected by default. Changing this value will invalidate all dynamic bindings for the current execution context, so that they have to be set again using :func:`set_binding_shape` before calling either :func:`execute` or :func:`execute_async` .
            :ivar all_binding_shapes_specified: :class:`bool` Whether all dynamic dimensions of input tensors have been specified by calling :func:`set_binding_shape` . Trivially true if network has no dynamically shaped input tensors.
            :ivar all_shape_inputs_specified: :class:`bool` Whether values for all input shape tensors have been specified by calling :func:`set_shape_input` . Trivially true if network has no input shape bindings.
        )trtdoc";

        constexpr const char* execute = R"trtdoc(
            Synchronously execute inference on a batch.
            This method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using :func:`ICudaEngine.get_binding_index()` .

            :arg batch_size: The batch size. This is at most the value supplied when the :class:`ICudaEngine` was built.
            :arg bindings: A list of integers representing input and output buffer addresses for the network.

            :returns: True if execution succeeded.
        )trtdoc";

        constexpr const char* execute_async = R"trtdoc(
            Asynchronously execute inference on a batch.
            This method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using :func:`ICudaEngine::get_binding_index()` .

            :arg batch_size: The batch size. This is at most the value supplied when the :class:`ICudaEngine` was built.
            :arg bindings: A list of integers representing input and output buffer addresses for the network.
            :arg stream_handle: A handle for a CUDA stream on which the inference kernels will be executed.
            :arg input_consumed: An optional event which will be signaled when the input buffers can be refilled with new data

            :returns: True if the kernels were executed successfully.
        )trtdoc";

        constexpr const char* execute_v2 = R"trtdoc(
            Synchronously execute inference on a batch.
            This method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using :func:`ICudaEngine.get_binding_index()` .
            This method only works for execution contexts built from networks with no implicit batch dimension.

            :arg bindings: A list of integers representing input and output buffer addresses for the network.

            :returns: True if execution succeeded.
        )trtdoc";

        constexpr const char* execute_async_v2 = R"trtdoc(
            Asynchronously execute inference on a batch.
            This method requires a array of input and output buffers. The mapping from tensor names to indices can be queried using :func:`ICudaEngine::get_binding_index()` .
            This method only works for execution contexts built from networks with no implicit batch dimension.

            :arg bindings: A list of integers representing input and output buffer addresses for the network.
            :arg stream_handle: A handle for a CUDA stream on which the inference kernels will be executed.
            :arg input_consumed: An optional event which will be signaled when the input buffers can be refilled with new data

            :returns: True if the kernels were executed successfully.
        )trtdoc";

        constexpr const char* device_memory = R"trtdoc(
            The device memory for use by this :class:`IExecutionContext` .

            The memory must be aligned on a 256-byte boundary, and its size must be at least that
            returned by getDeviceMemorySize(). If using :func:`execute_async()` to run the network, The memory is in
            use from the invocation of :func:`execute_async()` until network execution is complete. If using :func:`execute()`,
            it is in use until :func:`execute()` returns. Releasing or otherwise using the memory for other
            purposes during this time will result in undefined behavior.
        )trtdoc";

        constexpr const char* get_strides = R"trtdoc(
            Return the strides of the buffer for the given binding.

            Note that strides can be different for different execution contexts with dynamic shapes.

            :arg binding: The binding index.
        )trtdoc";

        constexpr const char* set_binding_shape = R"trtdoc(
            Set the dynamic shape of a binding.

            Requires the engine to be built without an implicit batch dimension.
            The binding must be an input tensor, and all dimensions must be compatible with
            the network definition (i.e. only the wildcard dimension -1 can be replaced with a
            new dimension > 0). Furthermore, the dimensions must be in the valid range for the
            currently selected optimization profile.

            For all dynamic non-output bindings (which have at least one wildcard dimension of -1),
            this method needs to be called after setting :attr:`active_optimization_profile` before
            either :func:`execute_async` or :func:`execute` may be called. When all input shapes have been
            specified, :attr:`all_binding_shapes_specified` is set to :class:`True` .

            :arg binding: The binding index.
            :arg shape: The shape to set.

            :returns: :class:`False` if an error occurs (e.g. index out of range), else :class:`True` .
        )trtdoc";

        constexpr const char* get_binding_shape = R"trtdoc(
            Get the dynamic shape of a binding.

            If :func:`set_binding_shape` has been called on this binding (or if there are no
            dynamic dimensions), all dimensions will be positive. Otherwise, it is necessary to
            call :func:`set_binding_shape` before :func:`execute_async` or :func:`execute` may be called.

            If the ``binding`` is out of range, an invalid Dims with nbDims == -1 is returned.

            If ``ICudaEngine.binding_is_input(binding)`` is :class:`False` , then both
            :attr:`all_binding_shapes_specified` and :attr:`all_shape_inputs_specified` must be :class:`True`
            before calling this method.

            :arg binding: The binding index.

            :returns: A :class:`Dims` object representing the currently selected shape.
        )trtdoc";

        constexpr const char* set_shape_input = R"trtdoc(
            Set values of an input shape tensor required by shape calculations.

            :arg binding: The binding index of an input tensor for which ``ICudaEngine.is_shape_binding(binding)`` and ``ICudaEngine.binding_is_input(binding)`` are both true.
            :arg shape: An iterable containing the values of the input shape tensor. The number of values should be the product of the dimensions returned by ``get_binding_shape(binding)``.

            If ``ICudaEngine.is_shape_binding(binding)`` and ``ICudaEngine.binding_is_input(binding)`` are both true, this method must be called before :func:`execute_async` or :func:`execute` may be called. Additionally, this method must not be called if either ``ICudaEngine.is_shape_binding(binding)`` or ``ICudaEngine.binding_is_input(binding)`` are false.

            :returns: :class:`True` if the values were set successfully.
        )trtdoc";

        constexpr const char* get_shape = R"trtdoc(
            Get values of an input shape tensor required for shape calculations or an output tensor produced by shape calculations.

            :arg binding: The binding index of an input tensor for which ``ICudaEngine.is_shape_binding(binding)`` is true.

            If ``ICudaEngine.binding_is_input(binding) == False``, then both
            :attr:`all_binding_shapes_specified` and :attr:`all_shape_inputs_specified` must be :class:`True`
            before calling this method.

            :returns: An iterable containing the values of the shape tensor.
        )trtdoc";

        constexpr const char* set_optimization_profile_async = R"trtdoc(
            Set the optimization profile with async semantics

            :arg profile_index: The index of the optimization profile

            :arg stream_handle: cuda stream on which the work to switch optimization profile can be enqueued

            When an optimization profile is switched via this API, TensorRT may require that data is copied via cudaMemcpyAsync. It is the
            application’s responsibility to guarantee that synchronization between the profile sync stream and the enqueue stream occurs.

            :returns: :class:`True` if the optimization profile was set successfully
        )trtdoc";
    } //IExecutionContextDoc

    namespace ICudaEngineDoc
    {
    constexpr const char* descr = R"trtdoc(
            An :class:`ICudaEngine` for executing inference on a built network.

            The engine can be indexed with ``[]`` . When indexed in this way with an integer, it will return the corresponding binding name. When indexed with a string, it will return the corresponding binding index.

            :ivar num_bindings: :class:`int` The number of binding indices.
            :ivar max_batch_size: :class:`int` The maximum batch size which can be used for inference. For an engine built from an :class:`INetworkDefinition` without an implicit batch dimension, this will always be ``1`` .
            :ivar has_implicit_batch_dimension: :class:`bool` Whether the engine was built with an implicit batch dimension.. This is an engine-wide property. Either all tensors in the engine have an implicit batch dimension or none of them do. This is True if and only if the :class:`INetworkDefinition` from which this engine was built was created with the ``NetworkDefinitionCreationFlag.EXPLICIT_BATCH`` flag.
            :ivar num_layers: :class:`int` The number of layers in the network. The number of layers in the network is not necessarily the number in the original :class:`INetworkDefinition`, as layers may be combined or eliminated as the :class:`ICudaEngine` is optimized. This value can be useful when building per-layer tables, such as when aggregating profiling data over a number of executions.
            :ivar max_workspace_size: :class:`int` The amount of workspace the :class:`ICudaEngine` uses. The workspace size will be no greater than the value provided to the :class:`Builder` when the :class:`ICudaEngine` was built, and will typically be smaller. Workspace will be allocated for each :class:`IExecutionContext` .
            :ivar device_memory_size: :class:`int` The amount of device memory required by an :class:`IExecutionContext` .
            :ivar refittable: :class:`bool` Whether the engine can be refit.
            :ivar name: :class:`str` The name of the network associated with the engine. The name is set during network creation and is retrieved after building or deserialization.
            :ivar num_optimization_profiles: :class:`int` The number of optimization profiles defined for this engine. This is always at least 1.
        )trtdoc";

    constexpr const char* get_binding_index = R"trtdoc(
            Retrieve the binding index for a named tensor.

            You can also use engine's :func:`__getitem__` with ``engine[name]``. When invoked with a :class:`str` , this will return the corresponding binding index.

            :func:`IExecutionContext.execute_async()` and :func:`IExecutionContext.execute()` require an array of buffers.
            Engine bindings map from tensor names to indices in this array.
            Binding indices are assigned at :class:`ICudaEngine` build time, and take values in the range [0 ... n-1] where n is the total number of inputs and outputs.

            :arg name: The tensor name.

            :returns: The binding index for the named tensor, or -1 if the name is not found.
        )trtdoc";

    constexpr const char* get_binding_name = R"trtdoc(
            Retrieve the name corresponding to a binding index.

            You can also use engine's :func:`__getitem__` with ``engine[index]``. When invoked with an :class:`int` , this will return the corresponding binding name.

            This is the reverse mapping to that provided by :func:`get_binding_index()` .

            :arg index: The binding index.

            :returns: The name corresponding to the binding index.
        )trtdoc";

    // Documentation bug with parameters on these three functions because they are overloaded.
    constexpr const char* binding_is_input = R"trtdoc(
            Determine whether a binding is an input binding.

            :index: The binding index.

            :returns: True if the index corresponds to an input binding and the index is in range.
        )trtdoc";

    constexpr const char* binding_is_input_str = R"trtdoc(
            Determine whether a binding is an input binding.

            :name: The name of the tensor corresponding to an engine binding.

            :returns: True if the index corresponds to an input binding and the index is in range.
        )trtdoc";

    constexpr const char* get_binding_shape = R"trtdoc(
            Get the shape of a binding.

            :index: The binding index.

            :Returns: The shape of the binding if the index is in range, otherwise Dims()
        )trtdoc";

    constexpr const char* get_binding_shape_str = R"trtdoc(
            Get the shape of a binding.

            :name: The name of the tensor corresponding to an engine binding.

            :Returns: The shape of the binding if the tensor is present, otherwise Dims()
        )trtdoc";

    constexpr const char* get_binding_dtype = R"trtdoc(
            Determine the required data type for a buffer from its binding index.

            :index: The binding index.

            :Returns: The type of data in the buffer.
        )trtdoc";

    constexpr const char* get_binding_dtype_str = R"trtdoc(
            Determine the required data type for a buffer from its binding index.

            :name: The name of the tensor corresponding to an engine binding.

            :Returns: The type of data in the buffer.
        )trtdoc";

    constexpr const char* serialize = R"trtdoc(
            Serialize the engine to a stream.

            :returns: An :class:`IHostMemory` object containing the serialized :class:`ICudaEngine` .
        )trtdoc";

    constexpr const char* create_execution_context = R"trtdoc(
            Create an :class:`IExecutionContext` .

            :returns: The newly created :class:`IExecutionContext` .
        )trtdoc";

    constexpr const char* get_location = R"trtdoc(
            Get location of binding.
            This lets you know whether the binding should be a pointer to device or host memory.

            :index: The binding index.

            :returns: The location of the bound tensor with given index.
        )trtdoc";

    constexpr const char* get_location_str = R"trtdoc(
            Get location of binding.
            This lets you know whether the binding should be a pointer to device or host memory.

            :name: The name of the tensor corresponding to an engine binding.

            :returns: The location of the bound tensor with given index.
        )trtdoc";

    constexpr const char* create_execution_context_without_device_memory = R"trtdoc(
            Create an :class:`IExecutionContext` without any device memory allocated
            The memory for execution of this device context must be supplied by the application.

            :returns: An :class:`IExecutionContext` without device memory allocated.
        )trtdoc";

    constexpr const char* get_profile_shape = R"trtdoc(
            Get the minimum/optimum/maximum dimensions for a particular binding under an optimization profile.

            :arg profile_index: The index of the profile.
            :arg binding: The binding index or name.

            :returns: A ``List[Dims]`` of length 3, containing the minimum, optimum, and maximum shapes, in that order.
        )trtdoc";

    constexpr const char* get_profile_shape_input = R"trtdoc(
            Get minimum/optimum/maximum values for an input shape binding under an optimization profile. If the specified binding is not an input shape binding, an exception is raised.

            :arg profile_index: The index of the profile.
            :arg binding: The binding index or name.

            :returns: A ``List[List[int]]`` of length 3, containing the minimum, optimum, and maximum values, in that order. If the values have not been set yet, an empty list is returned.
        )trtdoc";

    constexpr const char* is_shape_binding = R"trtdoc(
            Returns :class:`True` if tensor is required as input for shape calculations or output from them.

            TensorRT evaluates a network in two phases:

            1. Compute shape information required to determine memory allocation requirements and validate that runtime sizes make sense.

            2. Process tensors on the device.

            Some tensors are required in phase 1. These tensors are called "shape tensors", and always
            have type :class:`tensorrt.int32` and no more than one dimension. These tensors are not always shapes
            themselves, but might be used to calculate tensor shapes for phase 2.

            :func:`is_shape_binding` returns true if the tensor is a required input or an output computed in phase 1.
            :func:`is_execution_binding` returns true if the tensor is a required input or an output computed in phase 2.

            For example, if a network uses an input tensor with binding ``i`` as an input to an IElementWiseLayer that computes the reshape dimensions for an :class:`IShuffleLayer` , ``is_shape_binding(i) == True``

            It's possible to have a tensor be required by both phases. For instance, a tensor can be used as a shape in an :class:`IShuffleLayer` and as the indices for an :class:`IGatherLayer` collecting floating-point data.

            It's also possible to have a tensor required by neither phase that shows up in the engine's inputs. For example, if an input tensor is used only as an input to an :class:`IShapeLayer` , only its shape matters and its values are irrelevant.

            :arg binding: The binding index.
        )trtdoc";

    constexpr const char* is_execution_binding = R"trtdoc(
            Returns :class:`True` if tensor is required for execution phase, false otherwise.

            For example, if a network uses an input tensor with binding i ONLY as the reshape dimensions for an :class:`IShuffleLayer` , then ``is_execution_binding(i) == False``, and a binding of `0` can be supplied for it when calling :func:`IExecutionContext.execute` or :func:`IExecutionContext.execute_async` .

            :arg binding: The binding index.
        )trtdoc";

    constexpr const char* get_binding_bytes_per_component = R"trtdoc(
            Return the number of bytes per component of an element.
            The vector component size is returned if :func:`get_binding_vectorized_dim` != -1.

            :arg index: The binding index.
        )trtdoc";

    constexpr const char* get_binding_components_per_element = R"trtdoc(
            Return the number of components included in one element.

            The number of elements in the vectors is returned if :func:`get_binding_vectorized_dim` != -1.

            :arg index: The binding index.
        )trtdoc";

    constexpr const char* get_binding_format = R"trtdoc(
            Return the binding format.

            :arg index: The binding index.
        )trtdoc";

    constexpr const char* get_binding_format_desc = R"trtdoc(
            Return the human readable description of the tensor format.

            The description includes the order, vectorization, data type, strides, etc. For example:

            |   Example 1: kCHW + FP32
            |       "Row major linear FP32 format"
            |   Example 2: kCHW2 + FP16
            |       "Two wide channel vectorized row major FP16 format"
            |   Example 3: kHWC8 + FP16 + Line Stride = 32
            |       "Channel major FP16 format where C % 8 == 0 and H Stride % 32 == 0"

            :arg index: The binding index.
        )trtdoc";

    constexpr const char* get_binding_vectorized_dim = R"trtdoc(
            Return the dimension index that the buffer is vectorized.

            Specifically -1 is returned if scalars per vector is 1.

            :arg index: The binding index.
        )trtdoc";

    } // ICudaEngineDoc

    namespace BuilderFlagDoc
    {
    constexpr const char* descr
        = R"trtdoc(Valid modes that the builder can enable when creating an engine from a network definition.)trtdoc";

    constexpr const char* FP16 = R"trtdoc(Enable FP16 layer selection)trtdoc";
    constexpr const char* INT8 = R"trtdoc(Enable Int8 layer selection)trtdoc";
    constexpr const char* DEBUG = R"trtdoc(Enable debugging of layers via synchronizing after every layer)trtdoc";
    constexpr const char* GPU_FALLBACK
        = R"trtdoc(Enable layers marked to execute on GPU if layer cannot execute on DLA)trtdoc";
    constexpr const char* STRICT_TYPES = R"trtdoc(Enables strict type constraints)trtdoc";
    constexpr const char* REFIT = R"trtdoc(Enable building a refittable engine)trtdoc";
    constexpr const char* DISABLE_TIMING_CACHE
        = R"trtdoc(Disable reuse of timing information across identical layers.)trtdoc";
    constexpr const char* TF32
        = R"trtdoc(Allow (but not require) computations on tensors of type DataType.FLOAT to use TF32. TF32 computes inner products by rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas. Enabled by default.)trtdoc";
    } // namespace BuilderFlagDoc

    namespace QuantizationFlagDoc
    {
    constexpr const char* descr = R"trtdoc(List of valid flags for quantizing the network to int8.)trtdoc";

    constexpr const char* CALIBRATE_BEFORE_FUSION
        = R"trtdoc(Run int8 calibration pass before layer fusion. Only valid for IInt8LegacyCalibrator and IInt8EntropyCalibrator. We always run int8 calibration pass before layer fusion for IInt8MinMaxCalibrator and IInt8EntropyCalibrator2. Disabled by default.)trtdoc";
    } // namespace QuantizationFlagDoc

    namespace NetworkDefinitionCreationFlagDoc
    {
        constexpr const char* descr
            = R"trtdoc(List of immutable network properties expressed at network creation time. For example, to enable explicit batch mode, pass a value of ``1 << NetworkDefinitionCreationFlag.EXPLICIT_BATCH`` to :func:`create_network` )trtdoc";
        constexpr const char* EXPLICIT_BATCH = R"trtdoc(Specify that the network should be created with an explicit batch dimension.)trtdoc";
        constexpr const char* EXPLICIT_PRECISION
            = R"trtdoc(Specify that the network contains explicit quantization and dequantization scale layers.)trtdoc";
    } // NetworkDefinitionCreationFlagDoc

    namespace DeviceTypeDoc
    {
        constexpr const char* descr = R"trtdoc(Device types that TensorRT can execute on)trtdoc";

        constexpr const char* GPU = R"trtdoc(GPU device)trtdoc";
        constexpr const char* DLA = R"trtdoc(DLA core)trtdoc";
    } // DeviceTypeDoc

    namespace ProfilingVerbosityDoc
    {
        constexpr const char* descr = R"trtdoc(Profiling verbosity in NVTX annotations)trtdoc";

        constexpr const char* DEFAULT = R"trtdoc(Register layer names in NVTX message field)trtdoc";
        constexpr const char* NONE = R"trtdoc(Turn off NVTX traces)trtdoc";
        constexpr const char* VERBOSE = R"trtdoc(Register layer names in NVTX message field and register layer detail in NVTX JSON payload field)trtdoc";
    } // DeviceTypeDoc

    namespace TacticSourceDoc
    {
    constexpr const char* descr
        = R"trtdoc(List of tactic sources that can provide tactics for TensorRT.)trtdoc";

    constexpr const char* CUBLAS = R"trtdoc(
            Enables cuBLAS tactics.
            **NOTE:** Disabling this value will cause the cublas handle passed to plugins in attachToContext to be null.
        )trtdoc";
    constexpr const char* CUBLAS_LT = R"trtdoc(
            Enables cuBLAS LT tactics
        )trtdoc";
    } // namespace TacticSourceDoc

    namespace IBuilderConfigDoc
    {
    constexpr const char* descr = R"trtdoc(

                :ivar min_timing_iterations: :class:`int` The number of minimization iterations used when timing layers. When timing layers, the builder minimizes over a set of average times for layer execution. This parameter controls the number of iterations used in minimization.
                :ivar avg_timing_iterations: :class:`int` The number of averaging iterations used when timing layers. When timing layers, the builder minimizes over a set of average times for layer execution. This parameter controls the number of iterations used in averaging.
                :ivar int8_calibrator: :class:`IInt8Calibrator` Int8 Calibration interface. The calibrator is to minimize the information loss during the INT8 quantization process.
                :ivar max_workspace_size: :class:`int` The maximum workspace size. The maximum GPU temporary memory which the engine can use at execution time.
                :ivar flags: :class:`int` The build mode flags to turn on builder options for this network. The flags are listed in the BuilderFlags enum. The flags set configuration options to build the network. This should be in integer consisting of one or more :class:`BuilderFlag` s, combined via binary OR. For example, ``1 << BuilderFlag.FP16 | 1 << BuilderFlag.DEBUG``.
                :ivar profile_stream: :class:`int` The handle for the CUDA stream that is used to profile this network.
                :ivar num_optimization_profiles: :class:`int` The number of optimization profiles.
                :ivar default_device_type: :class:`tensorrt.DeviceType` The default DeviceType to be used by the Builder.
                :ivar DLA_core: :class:`int` The DLA core that the engine executes on. Must be between 0 and N-1 where N is the number of available DLA cores.
                :ivar profiling_verbosity: Profiling verbosity in NVTX annotations.
                :ivar algorithm_selector: :class:`IAlgorithmSelector` The algorithm slector to be set/get in the build config.
            )trtdoc";

    constexpr const char* clear_flag = R"trtdoc(
                clears the builder mode flag from the enabled flags.

                :arg flag: The flag to clear.
            )trtdoc";

    constexpr const char* set_flag = R"trtdoc(
                Add the input builder mode flag to the already enabled flags.

                :arg flag: The flag to set.
            )trtdoc";

    constexpr const char* get_flag = R"trtdoc(
                Check if a build mode flag is set.

                :arg flag: The flag to check.

                :returns: A `bool` indicating whether the flag is set.
            )trtdoc";

    constexpr const char* clear_quantization_flag = R"trtdoc(
                Clears the quantization flag from the enabled quantization flags.

                :arg flag: The flag to clear.
            )trtdoc";

    constexpr const char* set_quantization_flag = R"trtdoc(
                Add the input quantization flag to the already enabled quantization flags.

                :arg flag: The flag to set.
            )trtdoc";

    constexpr const char* get_quantization_flag = R"trtdoc(
                Check if a quantization flag is set.

                :arg flag: The flag to check.

                :returns: A `bool` indicating whether the flag is set.
            )trtdoc";

    constexpr const char* reset = R"trtdoc(
                Resets the builder configuration to defaults. When initializing a builder config object, we can call this function.
            )trtdoc";

    constexpr const char* add_optimization_profile = R"trtdoc(
            Add an optimization profile.

            This function must be called at least once if the network has dynamic or shape input tensors.

            :arg profile: The new optimization profile, which must satisfy ``bool(profile) == True``

            :returns: The index of the optimization profile (starting from 0) if the input is valid, or -1 if the input is
                     not valid.
        )trtdoc";

    constexpr const char* set_calibration_profile = R"trtdoc(
            Set a calibration profile.

            Calibration optimization profile must be set if int8 calibration is used to set scales for a network with runtime dimensions.

            :arg profile: The new calibration profile, which must satisfy ``bool(profile) == True`` or be nullptr. MIN and MAX values will be overwritten by kOPT.

            :returns: True if the calibration profile was set correctly.
        )trtdoc";

    constexpr const char* get_calibration_profile = R"trtdoc(
            Get the current calibration profile.

            :returns: The current calibration profile or nullptr if calibrartion profile is unset.
        )trtdoc";

    constexpr const char* set_device_type = R"trtdoc(
            Set the device that this layer must execute on. If DeviceType is not set or is reset, TensorRT will use the
            default DeviceType set in the builder.

            The DeviceType for a layer must be compatible with the safety flow (if specified). For example a layer
            cannot be marked for DLA execution while the builder is configured for kSAFE_GPU.


            :arg layer: The layer to set the DeviceType of
            :arg device_type: The DeviceType the layer must execute on
        )trtdoc";

    constexpr const char* get_device_type = R"trtdoc(
            Get the device that the layer executes on.

            :arg layer: The layer to get the DeviceType for

            :returns: The DeviceType of the layer
        )trtdoc";

    constexpr const char* is_device_type_set = R"trtdoc(
            Check if the DeviceType for a layer is explicitly set.

            :arg layer: The layer to check for DeviceType

            :returns: True if DeviceType is not default, False otherwise
        )trtdoc";

    constexpr const char* reset_device_type = R"trtdoc(
            Reset the DeviceType for the given layer.

            :arg layer: The layer to reset the DeviceType for
        )trtdoc";

    constexpr const char* can_run_on_DLA = R"trtdoc(
            Check if the layer can run on DLA.

            :arg layer: The layer to check

            :returns: A `bool` indicating whether the layer can run on DLA
        )trtdoc";

    constexpr const char* set_tactic_sources = R"trtdoc(
            Set tactic sources.

            This bitset controls which tactic sources TensorRT is allowed to use for tactic
            selection. By default, kCUBLAS is always enabled, and kCUBLAS_LT is enabled for x86
            platforms, as well as non-x86 platforms if CUDA >= 11.0

            Multiple tactic sources may be combined with a bitwise OR operation. For example,
            to enable cublas and cublasLt as tactic sources, use a value of:
            ``1U << static_cast<uint32_t>(TacticSource::kCUBLAS) | 1U << static_cast<uint32_t>(TacticSource::kCUBLAS_LT)``

            :arg tactic_sources: The tactic sources to set

            :returns: A `bool` indicating whether the tactic sources in the build configuration were updated. The tactic sources in the build configuration will not be updated if the provided value is invalid.
        )trtdoc";

    constexpr const char* get_tactic_sources = R"trtdoc(
            Get the tactic sources currently set in the engine build configuration.
        )trtdoc";

    } // namespace IBuilderConfigDoc

    namespace BuilderDoc
    {
    constexpr const char* descr = R"trtdoc(
            Builds an :class:`ICudaEngine` from a :class:`INetworkDefinition` .

            :ivar max_batch_size: :class:`int` The maximum batch size which can be used at execution time, and also the batch size for which the :class:`ICudaEngine` will be optimized.
            :ivar max_workspace_size: :class:`int` The maximum GPU temporary memory which the :class:`ICudaEngine` can use at execution time.
            :ivar debug_sync: :class:`bool` Whether the :class:`Builder` should use debug synchronization. If this is true, the :class:`Builder` will synchronize after timing each layer, and report the layer name. It can be useful when diagnosing issues at build time.
            :ivar min_find_iterations: :class:`int` The number of minimization iterations used when timing layers. When timing layers, the :class:`Builder` minimizes over a set of average times for layer execution. This parameter controls the number of iterations used in minimization.
            :ivar average_find_iterations: :class:`int` The number of averaging iterations used when timing layers. When timing layers, the :class:`Builder` minimizes over a set of average times for layer execution. This parameter controls the number of iterations used in averaging.
            :ivar platform_has_tf32: :class:`bool` Whether the platform has tf32 support.
            :ivar platform_has_fast_fp16: :class:`bool` Whether the platform has fast native fp16.
            :ivar platform_has_fast_int8: :class:`bool` Whether the platform has fast native int8.
            :ivar int8_mode: :class:`bool` Whether Int8 mode is used.
            :ivar int8_calibrator: :class:`IInt8Calibrator` The Int8 Calibration interface.
            :ivar fp16_mode: :class:`bool` Whether or not 16-bit kernels are permitted. During :class:`ICudaEngine` build fp16 kernels will also be tried when this mode is enabled.
            :ivar strict_type_constraints: :class:`bool` When strict type constraints is set, TensorRT will choose the type constraints that conforms to type constraints. If the flag is not enabled higher precision implementation may be chosen if it results in higher performance.
            :ivar refittable: :class:`bool` Whether an :class:`ICudaEngine` will be refittable.
            :ivar error_recorder: :class:`IErrorRecorder` Reference counted application-implemented error reporting interface for TensorRT objects.
        )trtdoc";

    // :ivar gpu_allocator: :class:`IGpuAllocator` The GPU allocator to be used by the :class:`Builder` . All GPU
    // memory acquired will use this allocator. If set to ``None``, the default allocator will be used.

    constexpr const char* init = R"trtdoc(
            :arg logger: The logger to use.
        )trtdoc";

    constexpr const char* create_network = R"trtdoc(
            Create a :class:`INetworkDefinition` object.

            :arg flags: :class:`NetworkDefinitionCreationFlag` s combined using bitwise OR. Default value is 0. This mimics the behavior of create_network() in TensorRT 5.1.

            :returns: An empty TensorRT :class:`INetworkDefinition` .
        )trtdoc";

    constexpr const char* build_cuda_engine = R"trtdoc(
            Builds an :class:`ICudaEngine` from a :class:`INetworkDefinition` .

            :arg network: The TensorRT :class:`INetworkDefinition` .

            :returns: A new :class:`ICudaEngine` .
        )trtdoc";

    constexpr const char* create_optimization_profile = R"trtdoc(
            Create a new optimization profile.

            If the network has any dynamic input tensors, the appropriate calls to :func:`IOptimizationProfile.set_shape` must be made. Likewise, if there are any shape input tensors, the appropriate calls to :func:`IOptimizationProfile.set_shape_input` are required.

            See :class:`IOptimizationProfile`
        )trtdoc";

    constexpr const char* create_builder_config = R"trtdoc(
        Create a builder configuration object.

        See :class:`IBuilderConfig`
    )trtdoc";

    constexpr const char* build_engine = R"trtdoc(
            Builds an engine for the given :class:`INetworkDefinition` and :class:`IBuilderConfig` .

            This enables the builder to build multiple engines based on the same network definition, but with different builder configurations.

            :arg network: The TensorRT :class:`INetworkDefinition` .
            :arg config: The TensorRT :class:`IBuilderConfig` .

            :returns: A new :class:`ICudaEngine` .
        )trtdoc";

    } /* BuilderDoc */

    namespace RuntimeDoc
    {
        constexpr const char* descr = R"trtdoc(
            Allows a serialized :class:`ICudaEngine` to be deserialized.
        )trtdoc";

        // :ivar gpu_allocator: :class:`IGpuAllocator` The GPU allocator to be used by the :class:`Runtime` . All GPU memory acquired will use this allocator. If set to None, the default allocator will be used (Default: cudaMalloc/cudaFree).

        constexpr const char* init = R"trtdoc(
            :arg logger: The logger to use.
        )trtdoc";

        constexpr const char* deserialize_cuda_engine = R"trtdoc(
            Deserialize an :class:`ICudaEngine` from a stream.

            :arg serialized_engine: The :class:`buffer` that holds the serialized :class:`ICudaEngine` .
            :arg plugin_factory: The :class:`IPluginFactory` , if any plugins are used by the network, otherwise None.

            :returns: The :class:`ICudaEngine`, or None if it could not be deserialized.
        )trtdoc";

    } /* RuntimeDoc */

    namespace RefitterDoc
    {
        constexpr const char* descr = R"trtdoc(
            Updates weights in an :class:`ICudaEngine` .
        )trtdoc";

        constexpr const char* init = R"trtdoc(
            :arg engine: The engine to refit.
            :arg logger: The logger to use.
        )trtdoc";

        constexpr const char* set_weights = R"trtdoc(
            Specify new weights for a layer of given name.
            Possible reasons for rejection are:

            * There is no such layer by that name.
            * The layer does not have weights with the specified role.
            * The number of weights is inconsistent with the layer’s original specification.

            Modifying the weights before :func:`refit_cuda_engine` completes will result in undefined behavior.

            :arg layer_name: The name of the layer.
            :arg role: The role of the weights. See :class:`WeightsRole` for more information.
            :arg weights: The weights to refit with.

            :returns: ``True`` on success, or ``False`` if new weights are rejected.
        )trtdoc";

        constexpr const char* refit_cuda_engine = R"trtdoc(
            Updates associated engine.  Return ``True`` if successful.

            Failure occurs if :func:`get_missing` != 0 before the call.
        )trtdoc";

        constexpr const char* get_missing = R"trtdoc(
            Get description of missing weights.

            For example, if some Weights have been set, but the engine was optimized
            in a way that combines weights, any unsupplied Weights in the combination
            are considered missing.

            :returns: The names of layers with missing weights, and the roles of those weights.
        )trtdoc";

        constexpr const char* get_all = R"trtdoc(
            Get description of all weights that could be refit.

            :returns: The names of layers with refittable weights, and the roles of those weights.
        )trtdoc";

        constexpr const char* get_dynamic_range = R"trtdoc(
           Gets the dynamic range of a tensor. If the dynamic range was never set, returns the range computed during calibration.

           :arg tensor_name: The name of the tensor whose dynamic range to retrieve.

           :returns: :class:`Tuple[float, float]` A tuple containing the [minimum, maximum] of the dynamic range.
       )trtdoc";

        constexpr const char* set_dynamic_range = R"trtdoc(
           Update dynamic range for a tensor.

           :arg tensor_name: The name of the tensor whose dynamic range to update.
           :arg range: The new range.

           :returns: :class:`True` if successful, :class:`False` otherwise.

           Returns false if there is no Int8 engine tensor derived from a network tensor of that name.  If successful, then :func:`get_missing` may report that some weights need to be supplied.
       )trtdoc";

        constexpr const char* get_tensors_with_dynamic_range = R"trtdoc(
           Get names of all tensors that have refittable dynamic ranges.

           :returns: The names of tensors with refittable dynamic ranges.
       )trtdoc";
    } /* RefitterDoc */

} /* tensorrt */
