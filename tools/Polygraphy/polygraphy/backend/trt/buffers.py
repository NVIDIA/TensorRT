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
from collections import OrderedDict

import numpy as np
import tensorrt as trt
from polygraphy.backend.trt import util as trt_util
from polygraphy.logger.logger import G_LOGGER
from polygraphy.util import cuda


# This class always uses profile 0 binding names to refer to input and output buffers.
# Therefore, it is essentially agnostic to multiple profiles, which are instead handled in the runner.
class Buffers(object):
    # Currently, buffers are reused across profiles.
    @staticmethod
    def from_engine(engine):
        buffers = Buffers()
        bindings_per_profile = trt_util.get_bindings_per_profile(engine)
        for idx in range(bindings_per_profile):
            binding = engine[idx]
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            buffers.device_buffers[binding] = cuda.DeviceBuffer(dtype=dtype)
            if not engine.binding_is_input(binding):
                buffers.outputs[binding] = np.empty(shape=tuple(), dtype=dtype)
        G_LOGGER.extra_verbose("Created device buffers: {:}".format(buffers.device_buffers))
        return buffers


    def __init__(self):
        self.device_buffers = OrderedDict()
        self.outputs = OrderedDict()


    def copy_inputs(self, feed_dict, stream=None):
        for name, buffer in feed_dict.items():
            self.device_buffers[name].copy_from(buffer, stream)


    # Resizes all device buffers to match the shapes currently set on the provided context
    def resize(self, engine, context, start_binding, end_binding):
        for binding in range(start_binding, end_binding):
            shape = tuple(context.get_binding_shape(binding))
            name = engine[binding - start_binding] # Use profile 0 binding names for all buffers.
            self.device_buffers[name].resize(shape)


    # Copies outputs from the device back to host.
    def copy_outputs(self, stream=None):
        for name, buffer in self.outputs.items():
            self.outputs[name] = self.device_buffers[name].copy_to(buffer, stream)


    def bindings(self):
        return [buf.address() for buf in self.device_buffers.values()]


    def free(self):
        [buf.free() for buf in self.device_buffers.values()]
