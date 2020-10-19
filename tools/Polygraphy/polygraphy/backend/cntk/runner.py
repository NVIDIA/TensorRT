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
# EXPERIMENTAL
from polygraphy.logger.logger import G_LOGGER
from polygraphy.backend.base import BaseRunner

from collections import OrderedDict
import time

import cntk


class CNTKRunner(BaseRunner):
    def __init__(self, model, name=None):
        super().__init__(name=name, prefix="cntk-runner")
        self.model = model


    def activate_impl(self):
        self.cntk_model = cntk.Function.load(self.model)

        self.inputs = OrderedDict()
        for inp in self.cntk_model.arguments:
            self.inputs[inp] = inp.shape


    def infer(self, feed_dict):
        start = time.time()
        inference_outputs = self.cntk_model.eval(feed_dict)
        end = time.time()

        out_dict = OrderedDict()
        for out_node, out in zip(self.cntk_model.outputs, inference_outputs):
            out_dict[out_node.name] = out

        self.inference_time = end - start
        return out_dict
