#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tensorrt as trt
import numpy as np
import typing

_numpy_to_plugin_field_type = {
    np.dtype('int32'): trt.PluginFieldType.INT32,
    np.dtype('int16'): trt.PluginFieldType.INT16,
    np.dtype('int8'): trt.PluginFieldType.INT8,
    np.dtype('bool'): trt.PluginFieldType.INT8,
    np.dtype('int64'): trt.PluginFieldType.INT64,
    np.dtype('float32'): trt.PluginFieldType.FLOAT32,
    np.dtype('float64'): trt.PluginFieldType.FLOAT64,
    np.dtype('float16'): trt.PluginFieldType.FLOAT16
}

_built_in_to_plugin_field_type = {
    int: trt.PluginFieldType.INT64,
    float: trt.PluginFieldType.FLOAT64,
    bool: trt.PluginFieldType.INT8,
    # str is handled separately, so not needed here
}

def _str_to_data_type(dtype: str) -> trt.DataType:
    if dtype == "FP32":
        return trt.DataType.FLOAT
    if dtype == "FP16":
        return trt.DataType.HALF
    try:
        return getattr(trt.DataType, dtype)
    except KeyError:
        raise ValueError(f"Unknown data type string '{dtype}'") from None


def _join_with(lst, middle = False, delim = ", "):
    if len(lst) == 0:
        return ""
    
    ret = ""
    if middle:
        ret += ", "

    ret += delim.join(lst)
    
    return ret

def _is_npt_ndarray(annotation):
    return (typing.get_origin(annotation) == np.ndarray) or (hasattr(annotation, "__origin__") and annotation.__origin__ == np.ndarray)

def _is_numpy_array(annotation):
    return (annotation == np.ndarray) or _is_npt_ndarray(annotation)

def _infer_numpy_type(annotation):
    assert _is_npt_ndarray(annotation)
    annot_args = typing.get_args(annotation) or annotation.__args__
    if len(annot_args) >= 2:
        np_type = typing.get_args(annot_args[1]) or annot_args[1].__args__
        if len(np_type) >= 1:
            return np_type[0]

    raise AttributeError("Improper annotation for numpy array. Annotate numpy array attributes using 'numpy.typing.NDArray[dtype]', where 'dtype' is the expected numpy dtype of the array.")
