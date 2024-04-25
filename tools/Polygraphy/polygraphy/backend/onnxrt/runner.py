#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from polygraphy.datatype import DataType


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

    @util.check_called_by("activate")
    def activate_impl(self):
        self.sess, _ = util.invoke_if_callable(self._sess)

    @util.check_called_by("get_input_metadata")
    def get_input_metadata_impl(self):
        meta = TensorMetadata()
        for node in self.sess.get_inputs():
            meta.add(
                node.name,
                dtype=DataType.from_dtype(node.type, "onnxruntime"),
                shape=node.shape,
            )
        return meta

    @util.check_called_by("infer")
    def infer_impl(self, feed_dict):
        """
        Implementation for running inference with ONNX-Runtime.
        Do not call this method directly - use ``infer()`` instead,
        which will forward unrecognized arguments to this method.

        Args:
            feed_dict (OrderedDict[str, Union[numpy.ndarray, torch.Tensor]]):
                    A mapping of input tensor names to corresponding input NumPy arrays or PyTorch tensors.
                    If PyTorch tensors are provided in the feed_dict, then this function
                    will return the outputs also as PyTorch tensors.

        Returns:
            OrderedDict[str, Union[numpy.ndarray, torch.Tensor]]:
                    A mapping of output tensor names to corresponding output NumPy arrays
                    or PyTorch tensors.
        """
        use_torch = any(util.array.is_torch(t) for t in feed_dict.values())
        # `to_numpy()`` and `to_torch()` should be zero-copy whenever possible.
        feed_dict = {name: util.array.to_numpy(t) for name, t in feed_dict.items()}

        start = time.time()
        inference_outputs = self.sess.run(None, feed_dict)
        end = time.time()

        out_dict = OrderedDict()
        for node, out in zip(self.sess.get_outputs(), inference_outputs):
            out_dict[node.name] = out if not use_torch else util.array.to_torch(out)
        self.inference_time = end - start
        return out_dict

    @util.check_called_by("deactivate")
    def deactivate_impl(self):
        del self.sess
