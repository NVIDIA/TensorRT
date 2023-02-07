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
import os

import numpy as np
from polygraphy import util
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.onnx import OnnxFromPath
from polygraphy.backend.tf import GraphFromFrozen
from polygraphy.common import TensorMetadata


def model_path(name=None):
    path = os.path.abspath(os.path.dirname(__file__))
    if name is not None:
        path = os.path.join(path, name)
    return path


class Model:
    def __init__(self, path, LoaderType, check_runner, input_metadata=None, ext_data=None):
        self.path = path
        self.loader = LoaderType(self.path)
        self.check_runner = check_runner
        self.input_metadata = input_metadata
        self.ext_data = ext_data


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
    assert np.array_equal(outputs["Y"], feed_dict["X"])


def check_empty_tensor_expand(runner, shapes):
    shape = shapes["new_shape"]
    feed_dict = {"data": np.zeros(shape=(2, 0, 3, 0), dtype=np.float32), "new_shape": np.array(shape, dtype=np.int32)}
    outputs = runner.infer(feed_dict)
    # Empty tensor will still be empty after broadcast
    assert outputs["expanded"].shape == shape
    assert util.volume(outputs["expanded"].shape) == 0


def check_reshape(runner):
    feed_dict = {"data": np.random.random_sample(size=(1, 3, 5, 5)).astype(np.float32)}
    outputs = runner.infer(feed_dict)
    assert np.all(outputs["output"] == feed_dict["data"].ravel())


def no_check_implemented(runner):
    raise NotImplementedError("No check_runner implemented for this model")


ONNX_MODELS = {
    "identity": Model(
        path=model_path("identity.onnx"),
        LoaderType=BytesFromPath,
        check_runner=check_identity,
        input_metadata=TensorMetadata().add("x", dtype=np.float32, shape=(1, 1, 2, 2)),
    ),
    "identity_identity": Model(
        path=model_path("identity_identity.onnx"), LoaderType=BytesFromPath, check_runner=check_identity_identity
    ),
    "dynamic_identity": Model(
        path=model_path("dynamic_identity.onnx"),
        LoaderType=BytesFromPath,
        check_runner=check_dynamic_identity,
        input_metadata=TensorMetadata().add("X", dtype=np.float32, shape=(1, 1, -1, -1)),
    ),
    "empty_tensor_expand": Model(
        path=model_path("empty_tensor_expand.onnx"), LoaderType=BytesFromPath, check_runner=check_empty_tensor_expand
    ),
    "and": Model(path=model_path("and.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented),
    "scan": Model(path=model_path("scan.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented),
    "pow_scalar": Model(
        path=model_path("pow_scalar.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented
    ),
    "dim_param": Model(path=model_path("dim_param.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented),
    "tensor_attr": Model(
        path=model_path("tensor_attr.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented
    ),
    "identity_with_initializer": Model(
        path=model_path("identity_with_initializer.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented
    ),
    "const_foldable": Model(
        path=model_path("const_foldable.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented
    ),
    "reshape": Model(path=model_path("reshape.onnx"), LoaderType=BytesFromPath, check_runner=check_reshape),
    "reducable": Model(
        path=model_path("reducable.onnx"),
        LoaderType=BytesFromPath,
        check_runner=no_check_implemented,
        input_metadata=TensorMetadata().add("X0", shape=(1,), dtype=np.float32).add("Y0", shape=(1,), dtype=np.float32),
    ),
    "reducable_with_const": Model(
        path=model_path("reducable_with_const.onnx"),
        LoaderType=BytesFromPath,
        check_runner=no_check_implemented,
    ),
    "ext_weights": Model(
        path=model_path("ext_weights.onnx"),
        LoaderType=OnnxFromPath,
        check_runner=no_check_implemented,
        ext_data=model_path("data"),
    ),
    "ext_weights_same_dir": Model(
        path=model_path(os.path.join("ext_weights_same_dir", "ext_weights.onnx")),
        LoaderType=OnnxFromPath,
        check_runner=no_check_implemented,
        ext_data=model_path("ext_weights_same_dir"),
    ),
    "capability": Model(
        path=model_path("capability.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented
    ),
    "instancenorm": Model(
        path=model_path("instancenorm.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented
    ),
    "add_with_dup_inputs": Model(
        path=model_path("add_with_dup_inputs.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented
    ),
    "needs_constraints": Model(
        path=model_path("needs_constraints.onnx"),
        LoaderType=BytesFromPath,
        check_runner=no_check_implemented,
        input_metadata=TensorMetadata().add("x", dtype=np.float32, shape=(1, 1, 256, 256)),
    ),
    "constant_fold_bloater": Model(
        path=model_path("constant_fold_bloater.onnx"),
        LoaderType=BytesFromPath,
        check_runner=no_check_implemented,
    ),
    "nonzero": Model(path=model_path("nonzero.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented),
    "inp_dim_val_not_set": Model(path=model_path("inp_dim_val_not_set.onnx"), LoaderType=BytesFromPath, check_runner=no_check_implemented)
}
