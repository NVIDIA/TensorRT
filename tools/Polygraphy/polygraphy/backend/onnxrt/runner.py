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
import time
from collections import OrderedDict

from polygraphy import mod, util
from polygraphy.backend.base import BaseRunner
from polygraphy.common import TensorMetadata

np = mod.lazy_import("numpy")


@mod.export()
class OnnxrtRunner(BaseRunner):
    """
    Runs inference using an ONNX-Runtime inference session.
    """

    def __init__(self, sess, name=None):
        """
        Args:
            sess (Union[onnxruntime.InferenceSession, Callable() -> onnxruntime.InferenceSession]):
                    An ONNX-Runtime inference session or a callable that returns one.
        """
        super().__init__(name=name, prefix="onnxrt-runner")
        self._sess = sess

    def activate_impl(self):
        self.sess, _ = util.invoke_if_callable(self._sess)

    def get_input_metadata_impl(self):
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
            "tensor(bool)": bool,
            "tensor(string)": np.unicode,
        }

        meta = TensorMetadata()
        for node in self.sess.get_inputs():
            dtype = ONNX_RT_TYPE_TO_NP[node.type] if node.type in ONNX_RT_TYPE_TO_NP else None
            meta.add(node.name, dtype=dtype, shape=node.shape)
        return meta

    def infer_impl(self, feed_dict):
        start = time.time()
        inference_outputs = self.sess.run(None, feed_dict)
        end = time.time()

        out_dict = OrderedDict()
        for node, out in zip(self.sess.get_outputs(), inference_outputs):
            out_dict[node.name] = out
        self.inference_time = end - start
        return out_dict

    def deactivate_impl(self):
        del self.sess
