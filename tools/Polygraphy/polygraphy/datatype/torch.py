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

torch = mod.lazy_import("torch>=1.13.0")


def _get_mapping():
    return {
        torch.float64: DataType.FLOAT64,
        torch.float32: DataType.FLOAT32,
        torch.float16: DataType.FLOAT16,
        torch.int16: DataType.INT16,
        torch.int32: DataType.INT32,
        torch.int64: DataType.INT64,
        torch.int8: DataType.INT8,
        torch.uint8: DataType.UINT8,
        torch.bool: DataType.BOOL,
        torch.bfloat16: DataType.BFLOAT16,
    }


@register_dtype_importer("torch")
def from_torch(torch_type):
    """
    Converts a PyTorch data type to a Polygraphy DataType.

    Args:
        torch_type (torch.dtype): The PyTorch data type.

    Returns:
        DataType: The Polygraphy data type.
    """
    if not torch.is_installed() or not torch.is_importable():
        return None

    return _get_mapping().get(torch_type)


@register_dtype_exporter("torch")
def from_datatype(self):
    """
    Converts this Polygraphy DataType to a PyTorch data type.

    Returns:
        torch.dtype: The PyTorch data type.
    """
    return util.invert_dict(_get_mapping()).get(self)
