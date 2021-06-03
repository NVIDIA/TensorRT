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
import time
from collections import OrderedDict

import numpy as np
import tensorrt as trt
from polygraphy.backend.base import BaseRunner
from polygraphy.backend.trt import util as trt_util
from polygraphy.logger.logger import G_LOGGER
from polygraphy.util import cuda, misc

misc.log_module_info(trt)


class TrtRunner(BaseRunner):
    """
    Runs inference using a TensorRT engine.
    """
    def __init__(self, engine, name=None):
        """
        Args:
            engine (Callable() -> Union[trt.ICudaEngine, trt.IExecutionContext]):
                    A callable that can supply either a TensorRT engine or execution context.
                    If an engine is provided, the runner will create a context automatically.
                    Otherwise, it will use the provided context.
                    If instead of a callable, the object is provided directly, then the runner
                    will *not* take ownership of it, and therefore will not destroy it.


            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        super().__init__(name=name, prefix="trt-runner")
        self._engine_or_context = engine


    def get_input_metadata(self):
        bindings_per_profile = trt_util.get_bindings_per_profile(self.context.engine)
        # This function always uses binding names of the 0th profile.
        return trt_util.get_input_metadata_from_engine(self.context.engine, start_binding=0, end_binding=bindings_per_profile)


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
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                device_buffers[binding] = cuda.DeviceBuffer(dtype=dtype)
                if not engine.binding_is_input(binding):
                    host_output_buffers[binding] = np.empty(shape=tuple(), dtype=dtype)
            G_LOGGER.extra_verbose("Created device buffers: {:}".format(device_buffers))
            return device_buffers, host_output_buffers


        engine_or_context, owning = misc.try_call(self._engine_or_context)

        self.engine, self.owns_engine = None, False
        self.context, self.owns_context = None, False

        if isinstance(engine_or_context, trt.ICudaEngine):
            self.engine = engine_or_context
            self.owns_engine = owning
            self.context = self.engine.create_execution_context()
            if not self.context:
                G_LOGGER.critical("Invalid Context. See error log for details.")
        elif isinstance(engine_or_context, trt.IExecutionContext):
            self.context = engine_or_context
            self.owns_context = owning
        else:
            G_LOGGER.critical("Invalid Engine or Context. Please ensure the engine was built correctly. See error log for details.")

        if not owning:
            G_LOGGER.verbose("Object was provided directly instead of via a Callable. This runner will not assume ownership. "
                             "Please ensure it is freed.")


        self.device_buffers, self.host_output_buffers = make_buffers(self.context.engine)
        self.stream = cuda.Stream()


    def set_shapes_from_feed_dict(self, feed_dict):
        """
        Sets context shapes according to the provided feed_dict, then resizes
        buffers as needed.

        Args:
            feed_dict (OrderedDict[str, numpy.ndarray]): A mapping of input tensor names to corresponding input NumPy arrays.

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
            shape = inp.shape
            # Only set shapes if required.
            # get_shape/get_binding_shape will return what a shape input/data input is currently set to.
            if is_dynamic_shape_input(binding): # For input shape tensors
                G_LOGGER.verbose("Setting shape binding: {:} (index: {:}) to: {:}".format(name, binding, inp))
                if tuple(self.context.get_shape(binding)) != tuple(inp):
                    self.context.set_shape_input(binding, inp)

            elif misc.is_shape_dynamic(self.context.engine.get_binding_shape(binding)):
                G_LOGGER.verbose("Setting binding: {:} (index: {:}) to shape: {:}".format(name, binding, shape))
                if tuple(self.context.get_binding_shape(binding)) != tuple(shape):
                    self.context.set_binding_shape(binding, shape)

        if not self.context.all_binding_shapes_specified:
            G_LOGGER.critical("Some input shapes were not specified.\nNote: Network inputs are: {:}".format(self.get_input_metadata()))
        if not self.context.all_shape_inputs_specified:
            G_LOGGER.critical("Some shape inputs were not specified.\nNote: Network inputs are: {:}".format(self.get_input_metadata()))

        # Resize device buffers - host buffers will be automatically resized by copy_to
        for binding in range(start_binding, end_binding):
            name = self.context.engine[binding - start_binding] # Use profile 0 binding names for all buffers.
            shape = tuple(self.context.get_binding_shape(binding))
            self.device_buffers[name].resize(shape)

        return start_binding, end_binding


    def infer_impl(self, feed_dict):
        start_binding, _ = self.set_shapes_from_feed_dict(feed_dict)

        start = time.time()

        for name, buffer in feed_dict.items():
            self.device_buffers[name].copy_from(buffer, self.stream)

        # Need to offset bindings in case the active profile is not 0.
        status = self.context.execute_async_v2(bindings=[0] * start_binding + [buf.address() for buf in self.device_buffers.values()], stream_handle=self.stream.address())
        if not status:
            G_LOGGER.critical("Model execution failed. Please see the log messages above for details")

        for name, buffer in self.host_output_buffers.items():
            self.host_output_buffers[name] = self.device_buffers[name].copy_to(buffer, self.stream)

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
