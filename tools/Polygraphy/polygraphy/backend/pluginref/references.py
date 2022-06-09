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

from polygraphy import mod
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")
gs = mod.lazy_import("onnx_graphsurgeon")

OP_REGISTRY = {}  # Dict[str, Callable]: Maps op names to reference implementations


def register(op):
    """
    Registers a function as the reference implementation for a given op.

    Args:
         op (str): The name of the op for which to register this function.
    """

    def register_impl(func):
        def wrapped_func(node, intermediate_tensors):
            inputs = []
            for inp in node.inputs:
                if inp.is_empty():  # Optional input
                    inputs.append(None)
                elif isinstance(inp, gs.Constant):
                    inputs.append(inp.values)
                elif inp.name in intermediate_tensors:
                    inputs.append(intermediate_tensors[inp.name])
                else:
                    G_LOGGER.internal_error(
                        f"Input: {inp.name} was not found in intermediate tensors and is not a constant.\nNote: Intermediate tensors include: {list(intermediate_tensors.keys())}"
                    )

            outputs = func(node.attrs, *inputs)
            if len(outputs) != len(node.outputs):
                G_LOGGER.internal_error(
                    f"{op} reference implementation returned the wrong number of outputs.\nNote: Expected {len(node.outputs)} but recevied {len(outputs)}"
                )

            return {out_tensor.name: out for out_tensor, out in zip(node.outputs, outputs)}

        OP_REGISTRY[op] = wrapped_func
        return wrapped_func

    return register_impl


@register("Identity")
def run_identity(attrs, x):
    return [x]


@register("InstanceNormalization")
def run_instancenorm(attrs, x, weights, bias):
    epsilon = attrs.get("epsilon", 1.0e-5)

    rank = len(x.shape)
    axis = tuple(range(2, rank))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)

    # Weights and bias needs to be broadcasted to shape of X. C dimension should be a wildcard.
    broadcast_shape = [-1] + [1] * (rank - 2)
    weights = weights.reshape(broadcast_shape)
    bias = bias.reshape(broadcast_shape)

    res = weights * (x - mean) / np.sqrt(var + epsilon) + bias
    return [res]


@register("MeanVarianceNormalization")
def run_meanvarnorm(attrs, x):
    epsilon = 1.0e-9
    axes = attrs.get("axes", [0, 2, 3])
    axes = tuple(axes)

    data_mean = np.mean(x, axis=axes, keepdims=True)
    data_mean_squared = np.power(data_mean, 2)
    data_squared = np.power(x, 2)
    data_squared_mean = np.mean(data_squared, axis=axes, keepdims=True)
    std = np.sqrt(data_squared_mean - data_mean_squared)
    res = (x - data_mean) / (std + epsilon)

    return [res]
