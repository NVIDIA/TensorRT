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

"""
This file contains the Activation class which abstracts plan Region views.
"""


import numpy as np
from typing import Dict
import pandas as pd


# This dictionary compresses JSON's long format description strings.
_regionFormatDict = {
    "Four wide channel vectorized row major Int8 format" : "Int8 NC/4HW4",
    "Four wide channel vectorized row major FP32 format" : "FP32 NC/4HW4",
    "Thirty-two wide channel vectorized row major Int8 format": "Int8 NC/32HW32",
    "Thirty-two wide channel vectorized row major FP32 format": "FP32 NC/32HW32",
    "Thirty-two wide channel vectorized row major FP16 format": "FP16 NC/32HW32",
    "Thirty-two wide channel vectorized row major Int8 format with 3 spatial dimensions": "Int8 NC32DHW",
    "Thirty-two wide channel vectorized row major FP16 format with 3 spatial dimensions": "FP16 NC32DHW",
    "Sixteen wide channel vectorized row major FP16 format": "FP16 NC16HW",
    "Channel major FP16 format where channel % 4 == 0": "FP16 NHWC4",
    "Channel major FP32 format where channel % 4 == 0": "FP32 NHWC4",
    "Channel major Int8 format where channel % 4 == 0": "Int8 NHWC4",
    "Channel major FP16 format where channel % 8 == 0": "FP16 NHWC8",
    "Channel major FP16 format where channel % 16 == 0": "FP16 NHWC16",
    "Channel major FP16 format where channel == 4 and column stride % 32 == 0": "FP16 NHWC4",
    "Channel major INT8 format where channel == 4 and column stride % 32 == 0": "Int8 NHWC4",
    "Channel major FP16 format where channel % 2 == 0": "FP16 NHWC2",
    "Channel major INT8 format where column stride % 32 == 0": "Int8 NHWC1",
    "Channel major INT8 format where channel % 16 == 0": "Int8 NHWC16",
    "Row major INT8 format where column stride % 64 == 0": "Int8 NCHW",
    "Channel major FP16 format where channel % 8 == 0 with 3 spatial dimensions": "FP16 NDHWC8",
    "Channel major FP16 format where channel == 1 and column stride % 32 == 0": "FP16 NHWC1",
    "Row major FP16 format where column stride % 64 == 0": "FP16",
    "Two wide channel vectorized row major FP16 format": "FP16 NC/2HW2",
    "Row major linear FP32": "FP32 NCHW",
    "Row major linear Int32": "INT32 NCHW",
    "Row major linear FP16 format": "FP16 NCHW",
    "Row major Int8 format": "Int8 NCHW",
    "Channel major FP32 format":"FP32 NHWC",
    "Channel major FP16 format": "FP16 NHWC",
    "Channel major Int8 format": "Int8 NHWC",
    "Row major linear BOOL": "Bool",
    "Channel major FP32 format with 3 spatial dimensions": "FP32 NDHWC",
    "Channel major FP32 format with 3 spatial dimensions where channel % 4 == 0": "FP32 NDHWC4",
    "Channel major FP32 format where channel % 4 == 0 with 3 spatial dimensions": "FP32 NDHWC4",
    "Row major linear UInt8 format" : "UInt8 NCHW",
    "Channel major UInt8 format": "UInt8 NHWC",
    "Row major linear Int64 format": "Int64 NCHW",
    "Row major linear BFloat16 format": "BF16 NCHW",
    "Channel major BFloat16 format where channel % 8 == 0": "BF16 NHWC8",
    "Channel major BFloat16 format where channel % 4 == 0": "BF16 NHWC4",
    "Channel major BFloat16 format where channel % 8 == 0 with 3 spatial dimensions": "BF16 NDHWC8",
    "Channel major BFloat16 format where channel % 2 == 0": "BF16 NHWC2",
    "Two wide channel vectorized row major BFloat16 format": "BF16 NC2HW",
    "Row major linear FP8 format": "FP8 NCHW",
    "Unknown format": "Unknown format",
    # kgen formats
    "BFloat16": "BFloat16",
    "Bool": "Bool",
    "Double": "Double",
    "DoubleComplex": "DoubleComplex",
    "Float": "Float",
    "FloatComplex": "FloatComplex",
    "FP8": "FP8",
    "Half": "Half",
    "Int16": "Int16",
    "Int32": "Int32",
    "Int64": "Int64",
    "Int8": "Int8",
    "None": "None",
    "UInt16": "UInt16",
    "UInt32": "UInt32",
    "UInt64": "UInt64",
    "UInt8": "UInt8",
}

class Activation:
    """Convenience class wrapping activation regions."""
    def __init__(self, raw_dict: Dict):
        def parse_tensor_info(desc):
            try:
                data_type, layout = desc.split(' ')
            except ValueError:
                data_type = desc
            unknown_format = ('Unknown format', 0)
            precision, data_size = {
                'FP8':            ('FP8',    1),
                'FP16':           ('FP16',   2),
                'Half':           ('FP16',   2),
                'FP32':           ('FP32',   4),
                'Float':          ('FP32',   4),
                'Double':         ('FP64',   8),
                'BFloat16':       ('FP32',   2),
                'Int8':           ('INT8',   1),
                'Int16':          ('INT16',  2),
                'INT32':          ('INT32',  4),
                'Int32':          ('INT32',  4),
                'Int64':          ('INT64',  8),
                'UInt8':          ('UINT8',  1),
                'UInt16':         ('UINT16', 2),
                'UInt32':         ('UINT32', 4),
                'UInt64':         ('UINT64', 8),
                'Unknown format': unknown_format,
                'None':           unknown_format,
            }.get(data_type, (data_type, 0))

            return precision, data_size

        self.name = raw_dict['Name']
        self.shape = raw_dict['Dimensions']
        format = raw_dict['Format/Datatype'].replace('.', '')
        self.format = _regionFormatDict.get(format,"Unknown format")
        self.precision, self.data_size = parse_tensor_info(self.format)
        self.size_bytes = np.prod(self.shape) * self.data_size

    def tooltip(self):
        tip = "\\n".join((str(self.shape), self.format,))
        return tip

    def __repr__(self):
        return f"{self.name}: {str(self.shape)}x{self.format}"


def create_activations(layer: pd.Series):
    inputs = [Activation(tensor) for tensor in layer.Inputs]
    outputs = [Activation(tensor) for tensor in layer.Outputs]
    return inputs, outputs
