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
from polygraphy.backend.tf import GraphFromFrozen
from polygraphy.backend.common import BytesFromPath
from polygraphy.common import TensorMetadata
from polygraphy.util import misc

import numpy as np
import os


def model_path(name):
    MODELS_BASE_PATH = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(MODELS_BASE_PATH, name)


class Model(object):
    def __init__(self, path, LoaderType, check_runner, input_metadata=None):
        self.path = path
        self.loader = LoaderType(self.path)
        self.check_runner = check_runner
        self.input_metadata = input_metadata


def check_tf_identity(runner):
    feed_dict = {"Input:0": np.random.random_sample(size=(1, 15, 25, 30)).astype(np.float32)}
    outputs = runner.infer(feed_dict)
    assert np.all(outputs["Identity_2:0"] == feed_dict["Input:0"])


TF_MODELS = {
    "identity": Model(path=model_path("tf_identity.pb"), LoaderType=GraphFromFrozen, check_runner=check_tf_identity),
}


def check_identity(runner):
    feed_dict = {"x": np.random.random_sample(size=(1, 1, 2, 2)).astype(np.float32)}
    outputs = runner.infer(feed_dict)
    assert np.all(outputs["y"] == feed_dict["x"])


def check_identity_identity(runner):
    feed_dict = {"X": np.random.random_sample(size=(64, 64)).astype(np.float32)}
    outputs = runner.infer(feed_dict)
    assert np.all(outputs["identity_out_2"] == feed_dict["X"])


def check_dynamic_identity(runner, shapes):
    feed_dict = {"X": np.random.random_sample(size=shapes["X"]).astype(np.float32)}
    outputs = runner.infer(feed_dict)
    assert np.all(outputs["Y"] == feed_dict["X"])


def check_empty_tensor_expand(runner, shapes):
    shape = shapes["new_shape"]
    feed_dict = {"data": np.array((0, )), "new_shape": np.array(shape, dtype=np.int32)}
    outputs = runner.infer(feed_dict)
    # Empty tensor will still be empty after broadcast
    assert outputs["expanded"].shape == shape
    assert misc.volume(outputs["expanded"].shape) == 0


# scan
# "and": Model(path=model_path("and.onnx"), LoaderType=OnnxFileLoader),
# "pow_scalar": Model(path=model_path("pow_scalar.onnx"), LoaderType=OnnxFileLoader),
def no_check_implemented(runner):
    raise NotImplementedError("No check_runner implemented for this model")


ONNX_MODELS = {
    "identity": Model(path=model_path("identity.onnx"), LoaderType=BytesFromPath, check_runner=check_identity, input_metadata=TensorMetadata().add("x", dtype=np.float32, shape=(1, 1, 2, 2))),
    "identity_identity": Model(path=model_path("identity_identity.onnx"), LoaderType=BytesFromPath, check_runner=check_identity_identity),
    "dynamic_identity": Model(path=model_path("dynamic_identity.onnx"), LoaderType=BytesFromPath, check_runner=check_dynamic_identity, input_metadata=TensorMetadata().add("X", dtype=np.float32, shape=(1, 1, -1, -1))),
    "empty_tensor_expand": Model(path=model_path("empty_tensor_expand.onnx"), LoaderType=BytesFromPath, check_runner=check_empty_tensor_expand),

    "scan": Model(path=model_path("scan.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented),
    "dim_param": Model(path=model_path("dim_param.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented),
    "tensor_attr": Model(path=model_path("tensor_attr.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented),
    "identity_with_initializer": Model(path=model_path("identity_with_initializer.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented),
}
