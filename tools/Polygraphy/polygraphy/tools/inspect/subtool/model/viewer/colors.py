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
"""Op-type color palette for the Cytoscape.js model viewer."""

# Each entry: (set of op-type strings, saturated hex, pastel hex).
# Saturated is used for legend swatches and node borders.
# Pastel is used for the HTML node header background.
# Order matters: the first match wins.
_OP_CATEGORY_COLORS: list = [
    (
        # ONNX activations
        {
            "Relu",
            "LeakyRelu",
            "Elu",
            "Selu",
            "Sigmoid",
            "Tanh",
            "HardSigmoid",
            "Softmax",
            "LogSoftmax",
            "Gelu",
            "Mish",
            "PRelu",
            "Shrink",
            "ThresholdedRelu",
            "HardSwish",
            "Swish",
            # TRT network (str(layer.type))
            "LayerType.ACTIVATION",
            "LayerType.SOFTMAX",
            "LayerType.RAGGED_SOFTMAX",
            # TRT engine (inspector JSON LayerType)
            "Activation",
            "RaggedSoftmax",
            # TRT legacy uppercase (older TRT str repr)
            "ACTIVATION",
            "SOFTMAX",
        },
        "#e8913a",
        "#fdd9b5",  # orange — activations
    ),
    (
        # ONNX conv / linear
        {
            "Conv",
            "ConvTranspose",
            "Gemm",
            "MatMul",
            "Linear",
            # TRT network
            "LayerType.CONVOLUTION",
            "LayerType.DECONVOLUTION",
            "LayerType.FULLY_CONNECTED",
            "LayerType.MATRIX_MULTIPLY",
            # TRT engine
            "Convolution",
            "Deconvolution",
            "FullyConnected",
            "MatrixMultiply",
            # TRT legacy uppercase
            "CONVOLUTION",
            "FULLY_CONNECTED",
            "DECONVOLUTION",
            "MATRIX_MULTIPLY",
        },
        "#3a7ca5",
        "#b8d4e3",  # steel blue — conv / linear
    ),
    (
        # ONNX normalisation
        {
            "BatchNormalization",
            "LayerNormalization",
            "GroupNormalization",
            "InstanceNormalization",
            "LpNormalization",
            # TRT network
            "LayerType.NORMALIZATION",
            # TRT engine
            "Normalization",
            "Norm",
            # TRT legacy uppercase
            "NORMALIZATION",
        },
        "#2a9d8f",
        "#b2dfdb",  # teal — normalisation
    ),
    (
        # ONNX pooling
        {
            "MaxPool",
            "AveragePool",
            "GlobalAveragePool",
            "GlobalMaxPool",
            "LpPool",
            # TRT network
            "LayerType.POOLING",
            # TRT engine
            "Pooling",
            # TRT legacy uppercase
            "POOLING",
        },
        "#7c5cad",
        "#d1c4e9",  # purple — pooling
    ),
    (
        # ONNX element-wise / arithmetic
        {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Pow",
            "Sqrt",
            "Exp",
            "Log",
            "Abs",
            "Neg",
            "Ceil",
            "Floor",
            "Round",
            "Clip",
            "Reduce",
            "ReduceSum",
            "ReduceMean",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            # TRT network
            "LayerType.ELEMENTWISE",
            "LayerType.REDUCE",
            "LayerType.SCALE",
            "LayerType.UNARY",
            # TRT engine
            "PointWise",
            "Scale",
            # TRT legacy uppercase
            "ELEMENTWISE",
            "SCALE",
        },
        "#d4a843",
        "#fff0c2",  # amber — element-wise
    ),
    (
        # ONNX shape / transform
        {
            "Reshape",
            "Transpose",
            "Flatten",
            "Squeeze",
            "Unsqueeze",
            "Concat",
            "Split",
            "Slice",
            "Gather",
            "GatherND",
            "Scatter",
            "ScatterND",
            "Tile",
            "Expand",
            "OneHot",
            # TRT network
            "LayerType.SHUFFLE",
            "LayerType.CONCATENATION",
            "LayerType.GATHER",
            "LayerType.SLICE",
            "LayerType.SCATTER",
            "LayerType.ONE_HOT",
            "LayerType.REVERSE_SEQUENCE",
            # TRT engine
            "Shuffle",
            "Concatenation",
            "ScatterElements",
            "ReverseSequence",
            # TRT legacy uppercase
            "SHUFFLE",
            "CONCATENATION",
        },
        "#d45a3e",
        "#ffccbc",  # coral — shape / transform
    ),
    (
        # ONNX resize / spatial
        {
            "Resize",
            "Upsample",
            "RoiAlign",
            "GridSample",
            # TRT network
            "LayerType.RESIZE",
            "LayerType.GRID_SAMPLE",
        },
        "#4ba3c7",
        "#b3e5fc",  # sky blue — resize / spatial
    ),
    (
        # ONNX recurrent
        {
            "LSTM",
            "GRU",
            "RNN",
            "Scan",
            # TRT network
            "LayerType.RNN_V2",
            "LayerType.LOOP",
            # TRT engine
            "RNNv2",
        },
        "#3f5fb0",
        "#c5cae9",  # indigo — recurrent
    ),
    (
        {"Attention", "MultiHeadAttention"},
        "#8e3da0",
        "#e1bee7",  # violet — attention
    ),
    (
        # TRT-specific: Myelin-compiled and format-conversion layers
        {
            "Reformat",
            "MyelinReformat",
            "Myelin",
            # TRT network
            "LayerType.REFORMAT",
        },
        "#5c7a8a",
        "#cfd8dc",  # slate — reformat / myelin
    ),
    (
        # TRT-specific: plugin and foreign-node layers
        {
            "PluginV2",
            "Plugin",
            "ForeignNode",
            # TRT network
            "LayerType.PLUGIN_V2",
            "LayerType.PLUGIN",
        },
        "#c97516",
        "#ffe0b2",  # amber-orange — plugin
    ),
]

_CATEGORY_NAMES = [
    "Activation",
    "Conv / Linear",
    "Normalisation",
    "Pooling",
    "Element-wise",
    "Shape / Transform",
    "Resize",
    "Recurrent",
    "Attention",
    "Reformat / Myelin",
    "Plugin",
]

_GRAPH_INPUT_COLOR = "#c8e6c9"
_GRAPH_OUTPUT_COLOR = "#f8bbd0"
_DEFAULT_OP_COLOR_SAT = "#b0b8c1"
_DEFAULT_OP_COLOR_PAS = "#e8ecf0"


def _op_color(op_type: str):
    """Return (saturated_color, pastel_color) for the given op type."""
    for ops, saturated, pastel in _OP_CATEGORY_COLORS:
        if op_type in ops:
            return saturated, pastel
    return _DEFAULT_OP_COLOR_SAT, _DEFAULT_OP_COLOR_PAS
