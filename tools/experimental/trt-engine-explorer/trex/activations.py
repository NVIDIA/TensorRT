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
    "Channel major INT8 format where column stride % 32 == 0": "Int8 NHWC1",
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
    "Unknown format": "Unknown format"
}

class Activation:
    """Convenience class wrapping activation regions."""
    def __init__(self, raw_dict: Dict):
        def parse_tensor_info(desc):
            if 'Int8' in desc:
                precision = 'INT8'
                data_size = 1
            elif 'FP32' in desc:
                precision = 'FP32'
                data_size = 4
            elif 'FP16' in desc:
                precision = 'FP16'
                data_size = 2
            elif 'INT32' in desc:
                precision = 'INT32'
                data_size = 4
            elif 'Bool' in desc:
                precision = 'BOOL'
                data_size = 4
            elif desc == "Unknown format":
                precision = 'Unknown'
                data_size = 0
            else:
                raise ValueError(f"Uknown precision {desc}")
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
