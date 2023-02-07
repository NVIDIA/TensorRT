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
import onnx
import pytest
from polygraphy import util, mod
from tests.helper import TEST_DIR, poly_run, is_file_non_empty
from tests.models.meta import ONNX_MODELS_PATH

class TestModelEngine:
    def test_load_onnx_model(self):
        # Load an onnx model and run inference
        assert poly_run([ONNX_MODELS_PATH['identity'], "--trtexec"])

    def test_save_load_engine(self):
        # Save an engine and check if file exists. Run inference with saved engine
        with util.NamedTemporaryFile() as outpath:
            assert poly_run([ONNX_MODELS_PATH['identity'], "--trtexec", "--save-engine", outpath.name])
            assert is_file_non_empty(outpath.name)
            assert poly_run(["--trtexec", outpath.name, "--model-type=engine"])

class TestProfileShapes:
    def test_input_shape(self):
        assert poly_run([ONNX_MODELS_PATH['dynamic_identity'],
            "--trtexec",
            "--input-shapes", "X:[1,2,3,3]",
        ])

    def test_explicit_profile(self):
        assert poly_run([ ONNX_MODELS_PATH['dynamic_identity'],
            "--trtexec",
            "--input-shapes", "X:[1,2,1,1]",
            "--trt-min-shapes", "X:[1,2,1,1]",
            "--trt-opt-shapes", "X:[1,2,1,1]",
            "--trt-max-shapes", "X:[1,2,1,1]",
        ])

    def test_explicit_profile_opt_runtime_shapes_differ(self):
        assert poly_run([ONNX_MODELS_PATH['dynamic_identity'],
            "--trtexec",
            "--input-shapes", "X:[1,2,2,2]",
            "--trt-min-shapes", "X:[1,2,1,1]",
            "--trt-opt-shapes", "X:[1,2,3,3]",
            "--trt-max-shapes", "X:[1,2,4,4]",
        ])

class TestParameterMapping:
    def test_args(self):
        assert poly_run([ONNX_MODELS_PATH['identity'],
            "--trtexec",
            "--fp16", "--int8", "--best", "--tf32",
            "--use-cuda-graph", "--no-data-transfers",
        ])

    def test_kwargs(self):
        assert poly_run([ONNX_MODELS_PATH['identity'],
            "--trtexec",
            "--workspace=1M", "--streams=2", "--device=0",
            "--avg-runs=1", "--min-timing=1", "--avg-timing=3",
            "--trtexec-iterations=1", "--trtexec-warmup=1", "--duration=1",
        ])


class TestAccuracyComparision:
    def test_trtexec_trt(self):
        # Compare inference results between trtexec and trt
        assert poly_run([ONNX_MODELS_PATH['identity'], "--trtexec", "--trt"])

    def test_trtexec_onnxrt(self):
        # Compare inference results between trtexec and onnxrt
        assert poly_run([ONNX_MODELS_PATH['identity'], "--trtexec", "--onnxrt"])
