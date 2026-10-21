# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from polygraphy import mod

gs = mod.lazy_import("onnx_graphsurgeon>=0.5.0")
from typing import List, Dict


def get_plugin_pattern():
    """
    Conv+ReLU plugin pattern:
        Input
          |
        Conv
          |
        ReLU
          |
        Output
    """
    pattern = gs.GraphPattern()
    in_0 = pattern.variable()
    w_0 = pattern.variable()
    b_0 = pattern.variable()
    conv_out = pattern.add("ConvNode", "Conv", inputs=[in_0, w_0, b_0])
    relu_out = pattern.add("ReluNode", "Relu", inputs=[conv_out])
    pattern.set_output_tensors([relu_out])

    return pattern


def get_matching_subgraphs(graph) -> List[Dict[str, str]]:
    gp = get_plugin_pattern()
    matches = gp.match_all(graph)
    ans = []
    for m in matches:
        input_tensors = [ip_tensor.name for ip_tensor in m.inputs]
        output_tensors = [op_tensor.name for op_tensor in m.outputs]

        conv_node = m.get("ConvNode")
        attrs = {}
        if conv_node.attrs:
            attr_mapping = {
                "kernel_shape": "kernel_size",
                "strides": "stride",
                "pads": "padding",
                "group": "groups",
                "dilations": "dilation",
            }

            for attr_name, attr_value in conv_node.attrs.items():
                if attr_name in attr_mapping:
                    plugin_attr_name = attr_mapping[attr_name]
                    attrs[plugin_attr_name] = attr_value

        ioa = {"inputs": input_tensors, "outputs": output_tensors, "attributes": attrs}
        ans.append(ioa)
    return ans


def get_plugin_metadata() -> Dict[str, str]:
    return {
        "name": "convReluPlugin",
        "op": "ConvReluPlugin",
    }


def _get_const_values(graph, t):
    """Return numpy values if tensor is backed by an ONNX initializer or a Constant, else None."""
    # Direct constant
    if isinstance(t, gs.Constant):
        return t.values
    # Try resolve by name from graph tensors (initializers are Constants in tensor map)
    try:
        tensor_map = graph.tensors()
        maybe = tensor_map.get(t.name)
        if isinstance(maybe, gs.Constant):
            return maybe.values
    except Exception:
        pass
    return None


def replace_with_plugin(
    graph, input_tensors: list, output_tensors: list, attrs=None, op=None
):
    """
    Custom replacement method for Conv+ReLU plugin.

    Insert a Transpose (H<->W) on the weight input, followed by a Reshape that
    restores the original weight shape. When the weights are constants, TRT can
    constant-fold these transforms.
    """
    # Build Conv-style inputs [X, W, B] by traversing from output to Relu -> Conv
    act_input = None
    weight = None
    bias = None
    try:
        final_out = output_tensors[0] if output_tensors else None
        if final_out is not None and final_out.inputs:
            relu_node = final_out.inputs[0]
            if getattr(relu_node, "op", None) == "Relu" and relu_node.inputs:
                conv_out_tensor = relu_node.inputs[0]
                if conv_out_tensor.inputs:
                    conv_node = conv_out_tensor.inputs[0]
                    if getattr(conv_node, "op", None) == "Conv":
                        conv_inputs = list(conv_node.inputs)
                        if len(conv_inputs) >= 1:
                            act_input = conv_inputs[0]
                        if len(conv_inputs) >= 2:
                            weight = conv_inputs[1]
                        if len(conv_inputs) >= 3:
                            bias = conv_inputs[2]
    except Exception:
        pass

    # Fallback to provided inputs if traversal fails
    if act_input is None and input_tensors:
        act_input = input_tensors[0]
    if weight is None and len(input_tensors) >= 2:
        weight = input_tensors[1]
    if bias is None and len(input_tensors) >= 3:
        bias = input_tensors[2]

    # If weight is resolvable to a constant (initializer or Constant), apply Transpose(H<->W) + Reshape back
    replaced_weight = weight
    if weight is not None:
        values = _get_const_values(graph, weight)
        if values is not None:
            try:
                shape = list(values.shape)
            except Exception:
                shape = None
            if shape is not None and len(shape) == 4:
                import numpy as np

                # Create Transpose node
                perm = [0, 1, 3, 2]
                t_out = gs.Variable(
                    name="transpose_weight_output",
                    dtype=weight.dtype if hasattr(weight, "dtype") else None,
                )
                t_node = gs.Node(
                    op="Transpose",
                    inputs=[weight],
                    outputs=[t_out],
                    attrs={"perm": perm},
                )
                graph.nodes.append(t_node)

                # Create Reshape node
                shape_const = gs.Constant(
                    name="reshape_shape", values=np.array(shape, dtype=np.int64)
                )
                r_out = gs.Variable(
                    name="reshaped_weight_output",
                    dtype=weight.dtype if hasattr(weight, "dtype") else None,
                )
                r_node = gs.Node(
                    op="Reshape", inputs=[t_out, shape_const], outputs=[r_out]
                )
                graph.nodes.append(r_node)

                replaced_weight = r_out

    # Assemble plugin inputs and delegate disconnection/insertion to default helper
    plugin_inputs = []
    if act_input is not None:
        plugin_inputs.append(act_input)
    if replaced_weight is not None:
        plugin_inputs.append(replaced_weight)
    if bias is not None:
        plugin_inputs.append(bias)

    from polygraphy.tools.plugin.subtool.replace import default_replace_with_plugin

    return default_replace_with_plugin(graph, plugin_inputs, output_tensors, attrs, op)
