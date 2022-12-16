#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import contextlib
import copy
import time
from collections import OrderedDict

from polygraphy import cuda, mod, util
from polygraphy.backend.base import BaseRunner
from polygraphy.backend.trt import util as trt_util
from polygraphy.common import FormattedArray
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")
trt = mod.lazy_import("tensorrt")


def _make_output_allocator():
    if mod.version(trt.__version__) <= mod.version("8.5.0.9"):
        G_LOGGER.internal_error("This function should only be called in TensorRT 8.5 and newer")

    class OutputAllocator(trt.IOutputAllocator):
        def __init__(self):
            trt.IOutputAllocator.__init__(self)
            self.buffers = {}
            self.shapes = {}

        def reallocate_output(self, tensor_name, memory, size, alignment):
            shape = (size,)
            if tensor_name not in self.buffers:
                self.buffers[tensor_name] = cuda.DeviceArray.raw(shape)
            else:
                self.buffers[tensor_name].resize(shape)
            G_LOGGER.extra_verbose(f"Reallocated output tensor: {tensor_name} to: {self.buffers[tensor_name]}")
            return self.buffers[tensor_name].ptr

        def notify_shape(self, tensor_name, shape):
            self.shapes[tensor_name] = tuple(shape)

    return OutputAllocator()


@mod.export()
class TrtRunner(BaseRunner):
    """
    Runs inference using TensorRT.

    Note that runners are not designed for production deployment and should generally
    be used only for prototyping, testing, and debugging.
    """

    def __init__(self, engine, name: str = None, optimization_profile: int = None):
        """
        Args:
            engine (Union[Union[trt.ICudaEngine, trt.IExecutionContext], Callable() -> Union[trt.ICudaEngine, trt.IExecutionContext]]):
                    A TensorRT engine or execution context or a callable that returns one.
                    If an engine is provided, the runner will create a context automatically.

            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
            optimization_profile (int):
                    The index of the optimization profile to set each time this runner is activated.
                    When this is not provided, the profile is not set explicitly and will default to the 0th profile.
                    You can also change the profile after the runner is active using the ``set_profile()`` method.
        """
        super().__init__(name=name, prefix="trt-runner")
        self._engine_or_context = engine
        self.optimization_profile = optimization_profile

    def activate_impl(self):
        engine_or_context, owning = util.invoke_if_callable(self._engine_or_context)

        if isinstance(engine_or_context, trt.ICudaEngine):
            self.engine = engine_or_context
            self.owns_engine = owning
            self.context = self.engine.create_execution_context()
            self.owns_context = True
            if not self.context:
                G_LOGGER.critical("Invalid Context. See error log for details.")
        elif isinstance(engine_or_context, trt.IExecutionContext):
            self.context = engine_or_context
            self.owns_context = owning
            self.engine = self.context.engine
            self.owns_engine = False
        else:
            G_LOGGER.critical(
                "Invalid Engine or Context. Please ensure the engine was built correctly. See error log for details."
            )

        if not owning:
            G_LOGGER.verbose(
                "Object was provided directly instead of via a Callable. This runner will not assume ownership. "
                "Please ensure it is freed."
            )

        def make_buffers_legacy():
            """
            Creates empty host and device buffers for the specified engine.
            Always uses binding names from Profile 0.
            """
            device_buffers = OrderedDict()
            host_output_buffers = OrderedDict()

            for idx in range(trt_util.get_bindings_per_profile(self.engine)):
                binding = self.engine[idx]
                dtype = trt_util.np_dtype_from_trt(self.engine.get_binding_dtype(binding))
                device_buffers[binding] = cuda.DeviceArray(dtype=dtype)
                if not self.engine.binding_is_input(binding):
                    host_output_buffers[binding] = np.empty(shape=tuple(), dtype=dtype)

            G_LOGGER.extra_verbose(f"Initialized device buffers: {device_buffers}")
            return device_buffers, host_output_buffers, None

        def make_buffers():
            """
            Creates empty host buffers for outputs and empty device buffers for inputs.
            """
            device_buffers = OrderedDict()
            host_output_buffers = OrderedDict()
            output_allocator = _make_output_allocator()

            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)

                # NOTE: We use raw arrays to enable vectorized formats.
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    device_buffers[name] = cuda.DeviceArray.raw(shape=tuple())
                elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    host_output_buffers[name] = np.empty(shape=tuple(), dtype=np.byte)
                    if not self.context.set_output_allocator(name, output_allocator):
                        G_LOGGER.critical(f"For output: {name}, failed to set output allocator")
                else:
                    G_LOGGER.internal_error(
                        f"Unexpected tensor I/O mode encountered during inference: {self.engine.get_tensor_mode(name)}.\n"
                        "Please update this implementation!"
                    )

            G_LOGGER.extra_verbose(f"Initialized device buffers: {device_buffers}")
            return device_buffers, host_output_buffers, output_allocator

        self.device_buffers, self.host_output_buffers, self.output_allocator = (
            make_buffers() if trt_util._should_use_v3_api() else make_buffers_legacy()
        )
        self.stream = cuda.Stream()

        if self.optimization_profile is not None:
            self.set_profile(self.optimization_profile)

    def set_profile(self, index: int):
        """
        Sets the active optimization profile for this runner.
        The runner must already be active (see ``__enter__()`` or ``activate()``).

        This only applies if your engine was built with multiple
        optimization profiles.

        In TensorRT 8.0 and newer, the profile will be set asynchronously
        using this runner's CUDA stream (``runner.stream``).

        By default, the runner uses the first profile (profile 0).

        Args:
            index (int):
                    The index of the optimization profile to use.
        """
        if not hasattr(self, "context") or self.context is None:
            G_LOGGER.critical(f"{self.name:35} | Must be activated prior to calling set_profile()")

        try:
            self.context.set_optimization_profile_async
        except AttributeError:
            self.context.active_optimization_profile = index
        else:
            if not self.context.set_optimization_profile_async(index, self.stream.ptr):
                G_LOGGER.critical(f"Failed to set optimization profile to: {index}")

    def get_input_metadata_impl(self):
        if trt_util._should_use_v3_api():
            return trt_util.get_metadata_from_engine(self.engine, mode=trt.TensorIOMode.INPUT)
        else:
            start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
            # This function always uses binding names of the 0th profile.
            return trt_util.get_input_metadata_from_engine(self.engine, start_binding, end_binding)

    def _set_shapes_from_feed_dict_legacy(self, feed_dict):
        """
        Sets context shapes according to the provided feed_dict.

        Note that ``infer()`` will call this function automatically, and hence
        you should only use it if you plan to use this runner's context manually.

        Args:
            feed_dict (OrderedDict[str, numpy.ndarray]):
                    A mapping of input tensor names to corresponding input NumPy arrays.

        Returns:
            Tuple[int, int]: The start and end binding indices of the modified bindings.
        """

        def is_dynamic_shape_input(binding):
            return self.engine.is_shape_binding(binding) and self.engine.binding_is_input(binding)

        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        for name, inp in feed_dict.items():
            binding = start_binding + self.engine[name]
            # Only set shapes if required.
            # get_shape/get_binding_shape will return what a shape input/data input is currently set to.
            if is_dynamic_shape_input(binding):  # For input shape tensors
                if isinstance(inp, cuda.DeviceView):
                    G_LOGGER.critical(
                        f"A DeviceView was provided for input: {name}, but since this is a shape tensor, "
                        "it must reside in host memory. Please use a NumPy array instead. "
                    )

                if tuple(self.context.get_shape(binding)) != tuple(inp):
                    G_LOGGER.verbose(lambda: f"Setting shape binding: {name} (index: {binding}) to: {inp}")
                    if not self.context.set_shape_input(binding, inp):
                        G_LOGGER.critical(
                            f"Failed to set shape binding: {name} (index: {binding}) to: {inp}. "
                            "Are these values valid for the binding?"
                        )

            elif util.is_shape_dynamic(self.engine.get_binding_shape(binding)):
                shape = inp.shape
                if tuple(self.context.get_binding_shape(binding)) != tuple(shape):
                    G_LOGGER.verbose(lambda: f"Setting binding: {name} (index: {binding}) to shape: {shape}")
                    if not self.context.set_binding_shape(binding, shape):
                        G_LOGGER.critical(
                            f"Failed to set binding: {name} (index: {binding}) to shape: {shape}. "
                            "Is this shape valid for the binding?"
                        )

        if not self.context.all_binding_shapes_specified:
            G_LOGGER.critical(
                f"Some input shapes were not specified.\nNote: Network inputs are: {self.get_input_metadata()}"
            )
        if not self.context.all_shape_inputs_specified:
            G_LOGGER.critical(
                f"Some shape inputs were not specified.\nNote: Network inputs are: {self.get_input_metadata()}"
            )

        return start_binding, end_binding

    def _infer_impl_legacy(self, feed_dict, copy_outputs_to_host):
        start_binding, end_binding = self._set_shapes_from_feed_dict_legacy(feed_dict)

        # Resize output device buffers - host buffers will be automatically resized by copy_to
        for binding in range(start_binding, end_binding):
            if not self.engine.binding_is_input(binding):
                name = self.engine[binding - start_binding]  # Use profile 0 binding names for all buffers.
                shape = tuple(self.context.get_binding_shape(binding))
                self.device_buffers[name].resize(shape)

        # Use a shallow copy in case we need to replace our allocated buffers with provided DeviceViews.
        dev_bufs = copy.copy(self.device_buffers)
        for name, buffer in feed_dict.items():
            if isinstance(buffer, cuda.DeviceView):
                dev_bufs[name] = buffer
            elif isinstance(buffer, np.ndarray):
                dev_bufs[name].resize(buffer.shape)
                buffer = util.make_contiguous(buffer)
                dev_bufs[name].copy_from(buffer, self.stream)
            else:
                G_LOGGER.critical(
                    f"For input: {name}, unrecognized type in feed_dict: {type(buffer).__name__}.\n"
                    "Please provide either a NumPy array or Polygraphy DeviceView. "
                )

        # Need to offset bindings in case the active profile is not 0.
        bindings = [0] * start_binding + [buf.ptr for buf in dev_bufs.values()]
        success = self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.ptr)
        if not success:
            G_LOGGER.critical("Model execution failed. Please see the log messages above for details")

        output_buffers = OrderedDict()
        for name, buffer in self.host_output_buffers.items():
            if copy_outputs_to_host:
                self.host_output_buffers[name] = util.resize_buffer(buffer, dev_bufs[name].shape)
                dev_bufs[name].copy_to(self.host_output_buffers[name], self.stream)
                output_buffers[name] = self.host_output_buffers[name]
            else:
                output_buffers[name] = dev_bufs[name].view()

        self.stream.synchronize()
        return output_buffers

    def _infer_impl_v3(self, feed_dict, copy_outputs_to_host):
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)

            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue

            # Set up input tensor shapes and copy from host memory if needed
            array = feed_dict[name]
            if not isinstance(array, FormattedArray):
                array = FormattedArray(array, shape=array.shape, dtype=array.dtype)

            underlying_array = array.array

            ptr = None
            if self.engine.is_shape_inference_io(name):
                if not isinstance(underlying_array, np.ndarray):
                    G_LOGGER.critical(
                        f"A {type(underlying_array).__name__} was provided for input: {name}, but since this is a shape tensor, "
                        "it must reside in host memory. Please use a NumPy array instead. "
                    )

                ptr = underlying_array.ctypes.data
            else:
                if isinstance(underlying_array, cuda.DeviceView):
                    ptr = underlying_array.ptr
                elif isinstance(underlying_array, np.ndarray):
                    underlying_array = util.make_contiguous(underlying_array)
                    dev_array = self.device_buffers[name]
                    dev_array.resize(shape=(underlying_array.nbytes,))

                    # For scalars, we need to reshape the array to 1D before we can use `view()` or NumPy complains.
                    if not underlying_array.shape:
                        view = underlying_array.reshape(-1).view(np.byte)
                    else:
                        view = underlying_array.view(np.byte)

                    dev_array.copy_from(view, stream=self.stream)
                    ptr = dev_array.ptr
                else:
                    G_LOGGER.critical(
                        f"For input: {name}, unrecognized type in feed_dict: {type(underlying_array).__name__}.\n"
                        "Please provide either a NumPy array or Polygraphy DeviceView. "
                    )

            # Only update the input shape/address if something has changed. Otherwise, we'd be
            # doing extra work unnecessarily.
            # We retrieve the semantic shape from the FormattedArray, *not* the underlying array.
            if self.context.get_tensor_shape(name) != array.shape:
                G_LOGGER.ultra_verbose(f"Setting {name} input shape to: {array.shape}")
                if not self.context.set_input_shape(name, array.shape):
                    G_LOGGER.critical(f"For input: {name}, failed to set shape to: {array.shape}")

            if self.context.get_tensor_address(name) != ptr:
                if not self.context.set_tensor_address(name, ptr):
                    G_LOGGER.critical(f"For input: {name}, failed to set tensor address to: {ptr}")

        if not self.context.execute_async_v3(self.stream.ptr):
            G_LOGGER.critical("`execute_async_v3()` failed. Please see the logging output above for details.")

        output_buffers = OrderedDict()
        for name in self.host_output_buffers.keys():
            # If we're dealing with vectorized formats, we need to return a FormattedArray.
            # Otherwise, we create a view instead with the correct shape/dtype.
            raw_array = self.output_allocator.buffers[name]
            shape = self.output_allocator.shapes[name]
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))

            using_nonlinear_format = self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
            # The memory allocated by the output allocator may be larger than actually required.
            # If we're using a vectorized format, then we need to copy the whole thing.
            # Otherwise, we can determine how much we actually need.
            nbytes = raw_array.nbytes if using_nonlinear_format else (util.volume(shape) * dtype.itemsize)

            if copy_outputs_to_host:
                self.host_output_buffers[name] = util.resize_buffer(self.host_output_buffers[name], (nbytes,))
                raw_array.view(shape=(nbytes,)).copy_to(self.host_output_buffers[name], stream=self.stream)
                raw_array = self.host_output_buffers[name]

            if using_nonlinear_format:
                array = FormattedArray(raw_array, shape=shape, dtype=dtype)
            else:
                if copy_outputs_to_host:
                    array = raw_array.view(dtype).reshape(shape)
                else:
                    array = cuda.DeviceView(raw_array.ptr, shape, dtype)
            output_buffers[name] = array

        self.stream.synchronize()
        return output_buffers

    def infer_impl(self, feed_dict, copy_outputs_to_host=None):
        """
        Implementation for running inference with TensorRT.
        Do not call this method directly - use ``infer()`` instead,
        which will forward unrecognized arguments to this method.

        In addition to accepting NumPy arrays in the feed_dict, this runner can also
        accept Polygraphy DeviceViews. In that case, no host-to-device copy is necessary for the inputs.

        Args:
            feed_dict (OrderedDict[str, Union[numpy.ndarray, DeviceView]]):
                    A mapping of input tensor names to corresponding input NumPy arrays
                    or Polygraphy DeviceViews.

            copy_outputs_to_host (bool):
                    Whether to copy inference outputs back to host memory.
                    If this is False, Polygraphy DeviceViews are returned
                    instead of NumPy arrays.
                    Defaults to True.

        Returns:
            OrderedDict[str, Union[numpy.ndarray, DeviceView]]:
                    A mapping of output tensor names to corresponding output NumPy arrays
                    or Polygraphy DeviceViews.
        """
        copy_outputs_to_host = util.default(copy_outputs_to_host, True)

        start = time.time()
        if trt_util._should_use_v3_api():
            output_buffers = self._infer_impl_v3(feed_dict, copy_outputs_to_host)
        else:
            output_buffers = self._infer_impl_legacy(feed_dict, copy_outputs_to_host)
        end = time.time()
        self.inference_time = end - start

        return output_buffers

    def deactivate_impl(self):
        with contextlib.ExitStack() as stack:
            if self.owns_engine:
                stack.enter_context(self.engine)
            if self.owns_context:
                stack.enter_context(self.context)

            [buf.free() for buf in self.device_buffers.values()]
            self.stream.free()

        del (
            self.engine,
            self.owns_engine,
            self.context,
            self.owns_context,
            self.device_buffers,
            self.host_output_buffers,
            self.output_allocator,
            self.stream,
        )
