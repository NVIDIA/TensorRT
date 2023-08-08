#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time
from collections import OrderedDict

from polygraphy import cuda, mod, util
from polygraphy.backend.base import BaseRunner
from polygraphy.backend.trt import util as trt_util
from polygraphy.common import FormattedArray
from polygraphy.datatype import DataType
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")
torch = mod.lazy_import("torch>=1.13.0")
trt = mod.lazy_import("tensorrt>=8.5")


def _make_output_allocator(use_torch):
    class OutputAllocator(trt.IOutputAllocator):
        def __init__(self):
            trt.IOutputAllocator.__init__(self)
            self.buffers = {}
            self.shapes = {}

        def reallocate_output(self, tensor_name, memory, size, alignment):
            shape = (size,)
            if tensor_name not in self.buffers:
                self.buffers[tensor_name] = (
                    cuda.DeviceArray.raw(shape)
                    if not use_torch
                    else torch.empty(shape, dtype=torch.uint8, device="cuda")
                )
            else:
                self.buffers[tensor_name] = util.array.resize_or_reallocate(self.buffers[tensor_name], shape)
            G_LOGGER.extra_verbose(f"Reallocated output tensor: {tensor_name} to: {self.buffers[tensor_name]}")
            return util.array.data_ptr(self.buffers[tensor_name])

        def notify_shape(self, tensor_name, shape):
            self.shapes[tensor_name] = tuple(shape)

    return OutputAllocator()


def _get_array_on_cpu(arr, name, host_buffers, stream, nbytes, use_torch):
    """
    Copies the provided array to CPU memory and returns it.
    If sufficient CPU memory has not been allocated for the array in
    ``host_bufffers``, this function will allocate new memory.

    If the input is a `torch.Tensor`, then a `torch.Tensor` is returned.
    Otherwise, if the input is a `DeviceView`, a `NumPy` array is returned.

    Args:
        arr (Union[DeviceView, torch.Tensor]): The array.
        name (str): The name of the array.
        host_buffers (Dict[str, Union[numpy.ndarray, torch.Tensor]]):
                A mapping of names to host buffers.
        stream (cuda.Stream): The CUDA stream to use.
        nbytes (int): The number of bytes to copy. This may be smaller than the size of the GPU memory.
        use_torch (bool): Whether to use PyTorch tensors instead of NumPy arrays.

    Returns:
        Union[numpy.ndarray, torch.Tensor]: The host buffer as a flat array of bytes.
    """
    if not util.array.is_on_gpu(arr):
        G_LOGGER.internal_error(f"_get_array_on_cpu() should only be called with input arrays on the GPU!")

    # The host buffer will always be a "raw" array, i.e. a flat array of bytes.
    shape = (nbytes,)
    dtype = DataType.UINT8
    # If we switch between torch tensors and DeviceViews between inferences, we need to reallocate the host buffer.
    if name not in host_buffers or util.array.is_torch(host_buffers[name]) != use_torch:
        host_buffers[name] = (
            np.empty(shape, dtype=DataType.to_dtype(dtype, "numpy"))
            if not use_torch
            else torch.empty(shape, dtype=DataType.to_dtype(dtype, "torch"), device="cpu")
        )

    host_buffers[name] = util.array.resize_or_reallocate(host_buffers[name], shape)
    cuda.wrapper().memcpy(
        dst=util.array.data_ptr(host_buffers[name]),
        src=util.array.data_ptr(arr),
        nbytes=nbytes,
        kind=cuda.MemcpyKind.DeviceToHost,
        stream_ptr=stream.ptr,
    )
    return host_buffers[name]


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

    @util.check_called_by("activate")
    def activate_impl(self):
        engine_or_context, _ = util.invoke_if_callable(self._engine_or_context)

        if isinstance(engine_or_context, trt.ICudaEngine):
            self.engine = engine_or_context
            self.context = self.engine.create_execution_context()
            if not self.context:
                G_LOGGER.critical("Invalid Context. See error log for details.")
        elif isinstance(engine_or_context, trt.IExecutionContext):
            self.context = engine_or_context
            self.engine = self.context.engine
        else:
            G_LOGGER.critical(
                "Invalid Engine or Context. Please ensure the engine was built correctly. See error log for details."
            )

        self.device_input_buffers = OrderedDict()
        self.host_output_buffers = OrderedDict()
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

    @util.check_called_by("get_input_metadata")
    def get_input_metadata_impl(self):
        return trt_util.get_metadata_from_engine(self.engine, self.context, mode=trt.TensorIOMode.INPUT)

    def _infer_impl(self, feed_dict, copy_outputs_to_host):
        def get_io(mode):
            for idx in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(idx)

                if self.engine.get_tensor_mode(name) == mode:
                    yield name

        use_torch = False

        for name in get_io(trt.TensorIOMode.INPUT):
            # Set up input tensor shapes and copy from host memory if needed
            array = feed_dict[name]
            if not isinstance(array, FormattedArray):
                array = FormattedArray(array, shape=util.array.shape(array))

            underlying_array = array.array
            use_torch = use_torch or util.array.is_torch(underlying_array)

            ptr = None
            if self.engine.is_shape_inference_io(name):
                if not util.array.is_on_cpu(underlying_array):
                    G_LOGGER.critical(
                        f"A {type(underlying_array).__name__} was provided for input: {name}, but since this is a shape tensor, "
                        "it must reside in host memory. "
                    )

                ptr = util.array.data_ptr(underlying_array)
            else:
                ptr = trt_util._get_array_on_gpu(underlying_array, name, self.device_input_buffers, self.stream)

            # If the format is HWC, make sure array.shape is considered after transposing back to CHW
            if trt_util.get_tensor_format(self.engine, self.context, name) == trt.TensorFormat.HWC:
                array_shape = trt_util.get_chw_shape_from_hwc(array.shape, self.context.get_tensor_strides(name))
            else:
                array_shape = array.shape

            # Only update the input shape/address if something has changed. Otherwise, we'd be
            # doing extra work unnecessarily.
            # We retrieve the semantic shape from the FormattedArray, *not* the underlying array.
            if self.context.get_tensor_shape(name) != array_shape:
                G_LOGGER.ultra_verbose(f"Setting {name} input shape to: {array_shape}")
                if not self.context.set_input_shape(name, array_shape):
                    G_LOGGER.critical(f"For input: {name}, failed to set shape to: {array_shape}")

            if self.context.get_tensor_address(name) != ptr:
                if not self.context.set_tensor_address(name, ptr):
                    G_LOGGER.critical(f"For input: {name}, failed to set tensor address to: {ptr}")

        # Set up the output allocator before running inference.
        output_allocator = _make_output_allocator(use_torch and torch.cuda.is_available())
        for name in get_io(trt.TensorIOMode.OUTPUT):
            if not self.context.set_output_allocator(name, output_allocator):
                G_LOGGER.critical(f"For output: {name}, failed to set output allocator")

        if not self.context.execute_async_v3(self.stream.ptr):
            G_LOGGER.critical("`execute_async_v3()` failed. Please see the logging output above for details.")

        output_buffers = OrderedDict()
        for name in get_io(trt.TensorIOMode.OUTPUT):
            # If we're dealing with vectorized formats, we need to return a FormattedArray.
            # Otherwise, we create a view instead with the correct shape/dtype.
            raw_array = output_allocator.buffers[name]

            shape = output_allocator.shapes[name]
            # If the format is HWC, make sure the result is shaped accordingly
            tensor_format = trt_util.get_tensor_format(self.engine, self.context, name)
            if tensor_format == trt.TensorFormat.HWC:
                shape = trt_util.get_hwc_shape_from_chw(shape, self.context.get_tensor_strides(name))
            using_vectorized_format = tensor_format != trt.TensorFormat.LINEAR and tensor_format != trt.TensorFormat.HWC

            dtype = DataType.from_dtype(self.engine.get_tensor_dtype(name), source_module="tensorrt")

            # The memory allocated by the output allocator may be larger than actually required.
            # If we're using a vectorized format, then we need to copy the whole thing.
            # Otherwise, we can determine how much we actually need.
            nbytes = util.array.nbytes(raw_array) if using_vectorized_format else (util.volume(shape) * dtype.itemsize)

            if copy_outputs_to_host:
                raw_array = _get_array_on_cpu(
                    raw_array, name, self.host_output_buffers, self.stream, nbytes, use_torch=use_torch
                )

            if using_vectorized_format:
                array = FormattedArray(raw_array, shape=shape)
            else:
                array = util.array.view(raw_array, dtype, shape)
            output_buffers[name] = array

        self.stream.synchronize()
        return output_buffers

    @util.check_called_by("infer")
    def infer_impl(self, feed_dict, copy_outputs_to_host=None):
        """
        Implementation for running inference with TensorRT.
        Do not call this method directly - use ``infer()`` instead,
        which will forward unrecognized arguments to this method.

        Args:
            feed_dict (OrderedDict[str, Union[numpy.ndarray, DeviceView, torch.Tensor]]):
                    A mapping of input tensor names to corresponding input NumPy arrays,
                    Polygraphy DeviceViews, or PyTorch tensors.
                    If PyTorch tensors are provided in the feed_dict, then this function
                    will return the outputs also as PyTorch tensors.
                    If the provided inputs already reside in GPU memory, no additional copies are made.

            copy_outputs_to_host (bool):
                    Whether to copy inference outputs back to host memory.
                    If this is False, PyTorch GPU tensors or Polygraphy DeviceViews
                    are returned instead of PyTorch CPU tensors or NumPy arrays respectively.
                    Defaults to True.

        Returns:
            OrderedDict[str, Union[numpy.ndarray, DeviceView, torch.Tensor]]:
                    A mapping of output tensor names to corresponding output NumPy arrays,
                    Polygraphy DeviceViews, or PyTorch tensors.
        """
        copy_outputs_to_host = util.default(copy_outputs_to_host, True)

        start = time.time()
        output_buffers = self._infer_impl(feed_dict, copy_outputs_to_host)
        end = time.time()
        self.inference_time = end - start

        return output_buffers

    @util.check_called_by("deactivate")
    def deactivate_impl(self):
        [buf.free() for buf in self.device_input_buffers.values()]
        self.stream.free()

        del (
            self.engine,
            self.context,
            self.device_input_buffers,
            self.host_output_buffers,
            self.stream,
        )
