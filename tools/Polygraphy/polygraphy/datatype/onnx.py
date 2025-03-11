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

from polygraphy import mod, util
from polygraphy.datatype.datatype import (
    DataType,
    register_dtype_importer,
    register_dtype_exporter,
)

onnx = mod.lazy_import("onnx")


def _get_mapping():
    DATATYPE_FROM_ONNX = {
        "DOUBLE": DataType.FLOAT64,
        "FLOAT": DataType.FLOAT32,
        "FLOAT16": DataType.FLOAT16,
        "INT16": DataType.INT16,
        "INT32": DataType.INT32,
        "INT64": DataType.INT64,
        "INT8": DataType.INT8,
        "INT4": DataType.INT4,
        "UINT16": DataType.UINT16,
        "UINT32": DataType.UINT32,
        "UINT64": DataType.UINT64,
        "UINT8": DataType.UINT8,
        "BOOL": DataType.BOOL,
        "STRING": DataType.STRING,
        "BFLOAT16": DataType.BFLOAT16,
        "FLOAT8E4M3FN": DataType.FLOAT8E4M3FN,
        "FLOAT8E4M3FNUZ": DataType.FLOAT8E4M3FNUZ,
        "FLOAT8E5M2": DataType.FLOAT8E5M2,
        "FLOAT8E5M2FNUZ": DataType.FLOAT8E5M2FNUZ,
    }
    if None in DATATYPE_FROM_ONNX:
        del DATATYPE_FROM_ONNX[None]

    onnx_type_map = dict(onnx.TensorProto.DataType.items())
    return {
        onnx_type_map[key]: val
        for key, val in DATATYPE_FROM_ONNX.items()
        if key in onnx_type_map
    }


@register_dtype_importer("onnx")
def from_onnx(onnx_type):
    """
    Converts an ONNX data type to a Polygraphy DataType.

    Args:
        onnx_type (onnx.TensorProto.DataType): The ONNX data type.

    Returns:
        DataType: The Polygraphy data type.
    """
    if not onnx.is_installed() or not onnx.is_importable():
        return None

    return _get_mapping().get(onnx_type)


@register_dtype_exporter("onnx")
def from_datatype(self):
    """
    Converts this Polygraphy DataType to an ONNX data type.

    Returns:
        onnx.TensorProto.DataType: The ONNX data type.
    """
    return util.invert_dict(_get_mapping()).get(self)
