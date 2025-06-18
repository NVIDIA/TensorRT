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
from polygraphy.mod.trt_importer import lazy_import_trt

trt = lazy_import_trt()


def _get_mapping():
    DATATYPE_FROM_TENSORRT = {
        trt.float32: DataType.FLOAT32,
        trt.float16: DataType.FLOAT16,
        trt.int32: DataType.INT32,
        trt.int8: DataType.INT8,
        util.try_getattr(trt, "int64"): DataType.INT64,
        util.try_getattr(trt, "uint8"): DataType.UINT8,
        util.try_getattr(trt, "bool"): DataType.BOOL,
        util.try_getattr(trt, "bfloat16"): DataType.BFLOAT16,
        util.try_getattr(trt, "fp8"): DataType.FLOAT8E4M3FN,
        util.try_getattr(trt, "int4"): DataType.INT4,
        util.try_getattr(trt, "fp4"): DataType.FLOAT4,
    }
    if None in DATATYPE_FROM_TENSORRT:
        del DATATYPE_FROM_TENSORRT[None]

    return DATATYPE_FROM_TENSORRT


@register_dtype_importer("tensorrt")
def from_tensorrt(tensorrt_type):
    """
    Converts a TensorRT data type to a Polygraphy DataType.

    Args:
        tensorrt_type (tensorrt.DataType): The TensorRT data type.

    Returns:
        DataType: The Polygraphy data type.
    """
    if not trt.is_installed() or not trt.is_importable():
        return None

    return _get_mapping().get(tensorrt_type)


@register_dtype_exporter("tensorrt")
def from_datatype(self):
    """
    Converts this Polygraphy DataType to a TensorRT data type.

    Returns:
        tensorrt.DataType: The TensorRT data type.
    """
    return util.invert_dict(_get_mapping()).get(self)
