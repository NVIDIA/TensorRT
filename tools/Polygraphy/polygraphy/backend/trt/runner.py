#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import time

import tensorrt as trt
from polygraphy.backend.base import BaseRunner
from polygraphy.backend.trt import util as trt_util
from polygraphy.backend.trt.buffers import Buffers
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
            engine (Callable() -> trt.ICudaEngine):
                    A callable that can supply a TensorRT engine.
                    If instead of a loader, the engine is provided directly, then the runner
                    will *not* take ownership of it, and therefore will not destroy it.


            name (str):
                    The human-readable name prefix to use for this runner.
                    A runner count and timestamp will be appended to this prefix.
        """
        super().__init__(name=name, prefix="trt-runner")
        self._engine = engine


    def activate_impl(self):
        # If engine is a callable, then we own the engine
        self.engine, self.owning = misc.try_call(self._engine)

        if not self.engine:
            G_LOGGER.critical("Invalid Engine. Please ensure the engine was built correctly")

        if not self.owning:
            G_LOGGER.verbose("Engine was provided directly instead of via a Callable. This runner will not assume ownership. "
                           "Please ensure the engine is freed.")

        self.buffers = Buffers.from_engine(self.engine)
        self.stream = cuda.Stream()

        self.context = self.engine.create_execution_context()


    def get_input_metadata(self):
        bindings_per_profile = trt_util.get_bindings_per_profile(self.engine)
        # This function always uses binding names of the 0th profile.
        return trt_util.get_input_metadata_from_engine(self.engine, start_binding=0, end_binding=bindings_per_profile)


    def deactivate_impl(self):
        # Destroy the engine, and context.
        with self.context:
            pass

        if self.owning:
            with self.engine:
                pass

        self.buffers.free()
        self.stream.free()


    def infer(self, feed_dict):
        def is_dynamic_shape_input(binding):
            try:
                self.engine.get_profile_shape_input(0, binding)
                return True
            except RuntimeError:
                return False

        start_binding, end_binding = trt_util.get_active_profile_bindings(self.engine, self.context)
        for name, inp in feed_dict.items():
            binding = start_binding + self.engine[name]
            shape = inp.shape
            # Only set shapes if required.
            # get_shape/get_binding_shape will return what a shape input/data input is currently set to.
            if is_dynamic_shape_input(binding):
                G_LOGGER.verbose("Setting shape binding: {:} (index: {:}) to: {:}".format(name, binding, inp))
                if tuple(self.context.get_shape(binding)) != tuple(inp):
                    self.context.set_shape_input(binding, inp)

            elif misc.is_shape_dynamic(self.engine.get_binding_shape(binding)):
                G_LOGGER.verbose("Setting binding: {:} (index: {:}) to shape: {:}".format(name, binding, shape))
                if tuple(self.context.get_binding_shape(binding)) != tuple(shape):
                    self.context.set_binding_shape(binding, shape)

        if not self.context.all_binding_shapes_specified:
            G_LOGGER.critical("Some input shapes were not specified.\nNote: Network inputs are: {:}".format(self.get_input_metadata()))
        if not self.context.all_shape_inputs_specified:
            G_LOGGER.critical("Some shape inputs were not specified.\nNote: Network inputs are: {:}".format(self.get_input_metadata()))


        # Inference
        # Need to resize output buffers
        self.buffers.resize(self.engine, self.context, start_binding=start_binding, end_binding=end_binding)

        start = time.time()
        self.buffers.copy_inputs(feed_dict, self.stream)
        # Need to offset bindings in case the active profile is not 0.
        status = self.context.execute_async_v2(bindings=[0] * start_binding + self.buffers.bindings(), stream_handle=self.stream.address())
        if not status:
            G_LOGGER.critical("Model execution failed. Please see the log messages above for details")

        self.buffers.copy_outputs(self.stream)
        self.stream.synchronize()
        end = time.time()

        self.inference_time = end - start
        return self.buffers.outputs
