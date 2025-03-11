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

from polygraphy import util
from polygraphy.datatype.datatype import (
    DataType,
    register_dtype_importer,
    register_dtype_exporter,
)

__DATATYPE_FROM_ONNXRT = {
    "tensor(double)": DataType.FLOAT64,
    "tensor(float)": DataType.FLOAT32,
    "tensor(float16)": DataType.FLOAT16,
    "tensor(int16)": DataType.INT16,
    "tensor(int32)": DataType.INT32,
    "tensor(int64)": DataType.INT64,
    "tensor(int8)": DataType.INT8,
    "tensor(int4)": DataType.INT4,
    "tensor(uint16)": DataType.UINT16,
    "tensor(uint32)": DataType.UINT32,
    "tensor(uint64)": DataType.UINT64,
    "tensor(uint8)": DataType.UINT8,
    "tensor(bool)": DataType.BOOL,
    "tensor(string)": DataType.STRING,
    "tensor(bfloat16)": DataType.BFLOAT16,
    "tensor(float8e4m3fn)": DataType.FLOAT8E4M3FN,
    "tensor(float8e4m3fnuz)": DataType.FLOAT8E4M3FNUZ,
    "tensor(float8e5m2)": DataType.FLOAT8E5M2,
    "tensor(float8e5m2fnuz)": DataType.FLOAT8E5M2FNUZ,
}

__ONNXRT_FROM_DATATYPE = util.invert_dict(__DATATYPE_FROM_ONNXRT)


@register_dtype_importer("onnxruntime")
def from_onnxrt(onnxrt_type):
    """
    Converts an ONNX-Runtime data type to a Polygraphy DataType.

    Args:
        onnxrt_type (str): The ONNX-Runtime data type.

    Returns:
        DataType: The Polygraphy data type.
    """
    return __DATATYPE_FROM_ONNXRT.get(onnxrt_type)


@register_dtype_exporter("onnxruntime")
def from_datatype(self):
    """
    Converts this Polygraphy DataType to an ONNX-Runtime data type.

    Returns:
        str: The ONNX-Runtime data type.
    """
    return __ONNXRT_FROM_DATATYPE.get(self)
