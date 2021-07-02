#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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

from polygraphy import cuda, func, mod, util
from polygraphy.backend.base import BaseRunner
from polygraphy.backend.trt import util as trt_util
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")
trt = mod.lazy_import("tensorrt")


@mod.export()
class TrtRunner(BaseRunner):
    """
    Runs inference using TensorRT.

    Note that runners are not designed for production deployment and should generally
    be used only for prototyping, testing, and debugging.
    """

    def __init__(self, engine, name=None):
        """
        Args:
            engine (Callable() -> Union[trt.ICudaEngine, trt.IExecutionContext]):
                    A callable that can supply either a TensorRT engine or execution context.
                    If an engine is provided, the runner will create a context automatically.
                    This callable is invoked whenever the runner is activated.

                    Alternatively, the engine or context may be supplied directly instead of
                    through a callable, in which case the runner will *not* take ownership of it,
                    and therefore will not destroy it.


            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        super().__init__(name=name, prefix="trt-runner")
        self._engine_or_context = engine

    @func.constantmethod
    def get_input_metadata_impl(self):
        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # This function always uses binding names of the 0th profile.
        return trt_util.get_input_metadata_from_engine(self.context.engine, start_binding, end_binding)

    def activate_impl(self):
        def make_buffers(engine):
            """
            Creates empty host and device buffers for the specified engine.
            Always uses binding names from Profile 0.
            """
            device_buffers = OrderedDict()
            host_output_buffers = OrderedDict()

            for idx in range(trt_util.get_bindings_per_profile(engine)):
                binding = engine[idx]
                dtype = trt_util.np_dtype_from_trt(engine.get_binding_dtype(binding))
                device_buffers[binding] = cuda.DeviceArray(dtype=dtype)
                if not engine.binding_is_input(binding):
                    host_output_buffers[binding] = np.empty(shape=tuple(), dtype=dtype)
            G_LOGGER.extra_verbose("Created device buffers: {:}".format(device_buffers))
            return device_buffers, host_output_buffers

        engine_or_context, owning = util.invoke_if_callable(self._engine_or_context)

        if isinstance(engine_or_context, trt.ICudaEngine):
            self.engine = engine_or_context
            self.owns_engine = owning
            self.context = self.engine.create_execution_context()
            self.owns_context = True
            if not self.context:
                G_LOGGER.critical("Invalid Context. See error log for details.")
        elif isinstance(engine_or_context, trt.IExecutionContext):
            self.engine = None
            self.owns_engine = False
            self.context = engine_or_context
            self.owns_context = owning
        else:
            G_LOGGER.critical(
                "Invalid Engine or Context. Please ensure the engine was built correctly. See error log for details."
            )

        if not owning:
            G_LOGGER.verbose(
                "Object was provided directly instead of via a Callable. This runner will not assume ownership. "
                "Please ensure it is freed."
            )

        self.device_buffers, self.host_output_buffers = make_buffers(self.context.engine)
        self.stream = cuda.Stream()

    def set_profile(self, index):
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
        if not self.is_active:
            G_LOGGER.critical("{:35} | Must be activated prior to calling set_profile()".format(self.name))

        try:
            self.context.set_optimization_profile_async
        except AttributeError:
            self.context.active_optimization_profile = index
        else:
            self.context.set_optimization_profile_async(index, self.stream.ptr)

    def _set_shapes_from_feed_dict(self, feed_dict):
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
            try:
                self.context.engine.get_profile_shape_input(0, binding)
                return True
            except RuntimeError:
                return False

        start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        for name, inp in feed_dict.items():
            binding = start_binding + self.context.engine[name]
            # Only set shapes if required.
            # get_shape/get_binding_shape will return what a shape input/data input is currently set to.
            if is_dynamic_shape_input(binding):  # For input shape tensors
                if isinstance(inp, cuda.DeviceView):
                    G_LOGGER.critical(
                        "A DeviceView was provided for input: {:}, but since this is a "
                        "shape tensor, it must reside in host memory. "
                        "Please use a NumPy array instead. ".format(name)
                    )

                if tuple(self.context.get_shape(binding)) != tuple(inp):
                    G_LOGGER.verbose("Setting shape binding: {:} (index: {:}) to: {:}".format(name, binding, inp))
                    self.context.set_shape_input(binding, inp)

            elif util.is_shape_dynamic(self.context.engine.get_binding_shape(binding)):
                shape = inp.shape
                if tuple(self.context.get_binding_shape(binding)) != tuple(shape):
                    G_LOGGER.verbose("Setting binding: {:} (index: {:}) to shape: {:}".format(name, binding, shape))
                    self.context.set_binding_shape(binding, shape)

        if not self.context.all_binding_shapes_specified:
            G_LOGGER.critical(
                "Some input shapes were not specified.\n"
                "Note: Network inputs are: {:}".format(self.get_input_metadata())
            )
        if not self.context.all_shape_inputs_specified:
            G_LOGGER.critical(
                "Some shape inputs were not specified.\n"
                "Note: Network inputs are: {:}".format(self.get_input_metadata())
            )

        return start_binding, end_binding

    def infer_impl(self, feed_dict):
        start_binding, end_binding = self._set_shapes_from_feed_dict(feed_dict)

        # Resize output device buffers - host buffers will be automatically resized by copy_to
        for binding in range(start_binding, end_binding):
            if not self.context.engine.binding_is_input(binding):
                name = self.context.engine[binding - start_binding]  # Use profile 0 binding names for all buffers.
                shape = tuple(self.context.get_binding_shape(binding))
                self.device_buffers[name].resize(shape)

        start = time.time()

        # Use a shallow copy in case we need to replace our allocated buffers with provided DeviceViews.
        dev_bufs = copy.copy(self.device_buffers)
        for name, buffer in feed_dict.items():
            if isinstance(buffer, cuda.DeviceView):
                dev_bufs[name] = buffer
            elif isinstance(buffer, np.ndarray):
                dev_bufs[name].copy_from(buffer, self.stream)
            else:
                G_LOGGER.critical(
                    "Unrecognized type in feed_dict: {:} for input: {:}.\n"
                    "Please provide either a NumPy array or Polygraphy DeviceView. ".format(type(buffer).__name__, name)
                )

        # Need to offset bindings in case the active profile is not 0.
        bindings = [0] * start_binding + [buf.ptr for buf in dev_bufs.values()]
        success = self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.ptr)
        if not success:
            G_LOGGER.critical("Model execution failed. Please see the log messages above for details")

        for name, buffer in self.host_output_buffers.items():
            self.host_output_buffers[name] = dev_bufs[name].copy_to(buffer, self.stream)

        self.stream.synchronize()

        end = time.time()
        self.inference_time = end - start

        return self.host_output_buffers

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
            self.stream,
        )

    # Note: This can be removed once TRT 6 support is dropped.
    def infer(self, feed_dict, check_inputs=None):
        # Disable checks by default on TRT 6.0 due to implicit batch semantics.
        if mod.version(trt.__version__) < mod.version("7.0"):
            return super().infer(feed_dict, util.default(check_inputs, False))
        return super().infer(feed_dict, util.default(check_inputs, True))
