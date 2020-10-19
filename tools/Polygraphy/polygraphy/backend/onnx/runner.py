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
from collections import OrderedDict

from polygraphy.backend.base import BaseRunner
from polygraphy.backend.onnx import util as onnx_util
from polygraphy.logger.logger import G_LOGGER
from polygraphy.util import misc


class OnnxTfRunner(BaseRunner):
    """
    Runs an ONNX model using the TensorFlow backend for ONNX.
    """
    def __init__(self, model, name=None):
        """
        Creates an ONNX-TF runner.
        """
        super().__init__(name=name, prefix="onnxtf-runner")
        import polygraphy.backend.tf  # To set logging callback
        self._model = model


    def activate_impl(self):
        self.model, _ = misc.try_call(self._model)

        import onnx_tf
        G_LOGGER.info("Preparing ONNX-TF backend")
        self.tf_rep = onnx_tf.backend.prepare(self.model)


    def deactivate_impl(self):
        del self.tf_rep


    def infer(self, feed_dict):
        start = time.time()
        outputs = self.tf_rep.run(list(feed_dict.values()))
        end = time.time()

        out_dict = OrderedDict()
        for name, out in zip(self.tf_rep.outputs, outputs):
            out_dict[name] = out
        self.inference_time = end - start
        return out_dict


    def get_input_metadata(self):
        return onnx_util.get_input_metadata(self.model.graph)
