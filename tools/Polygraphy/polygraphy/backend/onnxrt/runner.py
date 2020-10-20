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

import numpy as np
from polygraphy.backend.base import BaseRunner
from polygraphy.common import TensorMetadata
from polygraphy.util import misc


class OnnxrtRunner(BaseRunner):
    """
    Runs inference using an ONNX-Runtime inference session.
    """
    def __init__(self, sess, name=None):
        """
        Args:
            sess (Callable() -> onnxruntime.InferenceSession):
                    A callable that can supply an ONNX-Runtime inferences session.
        """
        super().__init__(name=name, prefix="onnxrt-runner")
        self._sess = sess


    def activate_impl(self):
        self.sess, _ = misc.try_call(self._sess)


    def deactivate_impl(self):
        del self.sess


    def infer(self, feed_dict):
        start = time.time()
        inference_outputs = self.sess.run(None, feed_dict)
        end = time.time()

        out_dict = OrderedDict()
        for node, out in zip(self.sess.get_outputs(), inference_outputs):
            out_dict[node.name] = out
        self.inference_time = end - start
        return out_dict


    def get_input_metadata(self):
        ONNX_RT_TYPE_TO_NP = {
            "tensor(double)": np.float64,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int16)": np.int16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(int8)": np.int8,
            "tensor(uint16)": np.uint16,
            "tensor(uint32)": np.uint32,
            "tensor(uint64)": np.uint64,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool,
        }

        def process_shape(shape):
            # dims can be strings to indicate dynamic dimensions. Polygraphy uses None for the same purpose
            return [None if not isinstance(elem, int) else elem for elem in shape]

        meta = TensorMetadata()
        for node in self.sess.get_inputs():
            meta.add(node.name, dtype=ONNX_RT_TYPE_TO_NP[node.type], shape=process_shape(node.shape))
        return meta
