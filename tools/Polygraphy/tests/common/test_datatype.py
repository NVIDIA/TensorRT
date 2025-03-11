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

import numpy as np
import onnx
import pytest
import tensorrt as trt
import torch

from polygraphy import mod, util
from polygraphy.datatype import DataType, DataTypeEntry

DATATYPES = DataType.__members__.values()


class TestDataType:
    def compare_names(self, name, expected_name, replace_map):
        # Names may not match up exactly, so use the replace_map to make adjustments to the
        # foreign type before comparing against the Polygraphy type
        for old, new in replace_map.items():
            if name == old:
                name = new
        assert name == expected_name

    @pytest.mark.parametrize("dtype", DATATYPES, ids=str)
    def test_numpy(self, dtype):
        if dtype in [
            DataType.BFLOAT16,
            DataType.FLOAT8E4M3FN,
            DataType.FLOAT8E4M3FNUZ,
            DataType.FLOAT8E5M2,
            DataType.FLOAT8E5M2FNUZ,
            DataType.INT4,
            DataType.FLOAT4,
        ]:
            pytest.xfail("Type not supported by NumPy")

        np_type = dtype.numpy()
        assert DataType.to_dtype(dtype, "numpy") == np_type
        assert np_type.itemsize == dtype.itemsize
        self.compare_names(np_type.name, dtype.name, {"str": "string"})

        assert isinstance(np_type, np.dtype)

        assert DataType.from_dtype(np_type) == dtype

    @pytest.mark.parametrize("dtype", DATATYPES, ids=str)
    def test_onnxrt(self, dtype):
        if dtype in [
            DataType.INT4,
            DataType.FLOAT4,
        ]:
            pytest.skip("Type not supported by ONNX-RT")

        onnxrt_type = DataType.to_dtype(dtype, "onnxruntime")
        assert dtype.onnxruntime() == onnxrt_type

        assert isinstance(onnxrt_type, str)

        self.compare_names(
            onnxrt_type.replace("tensor(", "").replace(")", ""),
            dtype.name,
            {
                "double": "float64",
                "float": "float32",
            },
        )

        assert DataType.from_dtype(onnxrt_type) == dtype

    @pytest.mark.parametrize("dtype", DATATYPES, ids=str)
    def test_onnx(self, dtype):
        if dtype in [
            DataType.INT4,
            DataType.FLOAT4,
        ]:
            pytest.skip("Type not supported by ONNX")

        onnx_type = dtype.onnx()
        assert DataType.to_dtype(dtype, "onnx") == onnx_type

        assert isinstance(onnx_type, int)

        onnx_type_map = util.invert_dict(dict(onnx.TensorProto.DataType.items()))
        self.compare_names(
            onnx_type_map[onnx_type].lower(),
            dtype.name,
            {
                "double": "float64",
                "float": "float32",
            },
        )

        assert DataType.from_dtype(onnx_type) == dtype

    @pytest.mark.skipif(
        mod.version(trt.__version__) < mod.version("8.7"),
        reason="Unsupported before TRT 8.7",
    )
    @pytest.mark.parametrize("dtype", DATATYPES, ids=str)
    def test_tensorrt(self, dtype):
        unsupported_types = [
            DataType.FLOAT64,
            DataType.INT16,
            DataType.UINT16,
            DataType.UINT32,
            DataType.UINT64,
            DataType.STRING,
            DataType.INT64,
            DataType.FLOAT8E4M3FNUZ,
            DataType.FLOAT8E5M2,
            DataType.FLOAT8E5M2FNUZ,
        ]
        if  mod.version(trt.__version__) < mod.version("10.8"):
            unsupported_types.append(DataType.FLOAT4)
        if dtype in unsupported_types:
            pytest.xfail("Type not supported by TensorRT")

        tensorrt_dtype = dtype.tensorrt()
        assert DataType.to_dtype(dtype, "tensorrt") == tensorrt_dtype

        assert isinstance(tensorrt_dtype, trt.DataType)

        self.compare_names(
            tensorrt_dtype.name.lower(),
            dtype.name,
            {
                "double": "float64",
                "float": "float32",
                "half": "float16",
                "fp8": "float8e4m3fn",
                "fp4": "float4",
                "bf16": "bfloat16",
            },
        )

        assert DataType.from_dtype(tensorrt_dtype) == dtype

    @pytest.mark.parametrize("trt_dtype", trt.DataType.__members__.values())
    def test_all_tensorrt_types_supported(self, trt_dtype):
        dtype = DataType.from_dtype(trt_dtype, "tensorrt")
        assert isinstance(dtype, DataTypeEntry)

        assert dtype.tensorrt() == trt_dtype

    @pytest.mark.parametrize("dtype", DATATYPES, ids=str)
    def test_torch(self, dtype):
        if dtype in [
            DataType.FLOAT8E4M3FN,
            DataType.FLOAT8E4M3FNUZ,
            DataType.FLOAT8E5M2,
            DataType.FLOAT8E5M2FNUZ,
            DataType.UINT16,
            DataType.UINT32,
            DataType.UINT64,
            DataType.STRING,
            DataType.INT4,
            DataType.FLOAT4,
        ]:
            pytest.xfail("Type not supported by Torch")

        torch_type = dtype.torch()
        assert DataType.to_dtype(dtype, "torch") == torch_type

        assert isinstance(torch_type, torch.dtype)

        self.compare_names(str(torch_type).replace("torch.", ""), dtype.name, {})

        assert DataType.from_dtype(torch_type) == dtype
