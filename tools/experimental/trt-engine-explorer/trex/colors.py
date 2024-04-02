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
This file contains color pallete defnitions.
"""


from collections import defaultdict


NVDA_GREEN = '#76b900'
UNKNOWN_KEY_COLOR = 'gray'
GRID_COLOR = 'rgba(114, 179, 24, 0.3)'


# pallete = px.colors.qualitative.G10
# https://medialab.github.io/iwanthue/
default_pallete = [
    "#a11350",
    "#008619",
    "#4064ec",
    "#ffb519",
    "#8f1a8e",
    "#b2b200",
    "#64b0ff",
    "#e46d00",
    "#02d2ba",
    "#ef393d",
    "#f1b0f7",
    "#7e4401",
    UNKNOWN_KEY_COLOR]


# Set a color for each precision datatype.
precision_colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR, {
    'INT8':  NVDA_GREEN,
    'FP32':  'red',
    'FP16':  'orange',
    'INT32': 'lightgray',
    'FP8':   'deepskyblue',
})


# Set a color for each layer type.
layer_colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR, {
    # https://htmlcolorcodes.com/
    "Convolution":    "#4682B4", # SteelBlue
    "Deconvolution":  "#7B68EE", # MediumSlateBlue
    "ConvActPool":    "#6495ED", # CornflowerBlue
    "MatrixMultiply": "#1E90FF", # DodgerBlue
    "gemm":           "#1E90FF", # DodgerBlue
    "Reformat":       "#00FFFF", # Cyan
    "Shuffle":        "#BC8F8F", # RosyBrown
    "Slice":          "#FFA500", # Orange
    "Scale":          "#8FBC8B", # DarkSeaGreen
    "Quantize":       "#6B8E23", # OliveDrab
    "Pooling":        "#3CB371", # MediumSeaGreen
    "PluginV2":       "#C71585", # MediumVioletRed
    "PointWise":      "#9ACD32", # YellowGreen
    "ElementWise":    "#9ACD32", # YellowGreen
    "Reduce":         "#90EE90", # LightGreen
    "SoftMax":        "#DA70D6", # Orchid
    "Myelin":         "#B39C4D", # Satic Sheen Gold
    "kgen":           "#B39C4D", # Satic Sheen Gold
    "NonZero":        "#98FB98", # PaleGreen
    "TrainStation":   "#FFA07A", # LightSalmon
})
