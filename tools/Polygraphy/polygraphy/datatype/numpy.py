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

np = mod.lazy_import("numpy")


def _get_mapping():
    DATATYPE_FROM_NUMPY = {
        np.double: DataType.FLOAT64,
        np.float32: DataType.FLOAT32,
        np.float16: DataType.FLOAT16,
        np.int16: DataType.INT16,
        np.int32: DataType.INT32,
        np.int64: DataType.INT64,
        np.int8: DataType.INT8,
        np.uint16: DataType.UINT16,
        np.uint32: DataType.UINT32,
        np.uint64: DataType.UINT64,
        np.uint8: DataType.UINT8,
        np.bool_: DataType.BOOL,
        np.str_: DataType.STRING,
    }
    return {np.dtype(key): val for key, val in DATATYPE_FROM_NUMPY.items()}


@register_dtype_importer("numpy")
def from_numpy(numpy_type):
    """
    Converts a NumPy data type to a Polygraphy DataType.

    Args:
        numpy_type (np.dtype): The NumPy data type.

    Returns:
        DataType: The Polygraphy data type.
    """
    if not np.is_installed() or not np.is_importable():
        return None

    try:
        dtype = np.dtype(numpy_type)
    except TypeError:
        return None

    return _get_mapping().get(dtype)


@register_dtype_exporter("numpy")
def from_datatype(self):
    """
    Converts this Polygraphy DataType to a NumPy data type.

    Returns:
        np.dtype: The NumPy data type.
    """
    return util.invert_dict(_get_mapping()).get(self)
