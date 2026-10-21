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
import onnx


def get_swish_pattern():
    """
    Swish sub-pattern:
        Input (temp)
          |
        Sigmoid (temp)
          |
        Mul (temp, sigmoid_out)
          |
        Output
    """
    pattern = gs.GraphPattern()
    temp = pattern.variable()  # Input tensor (result of GEMM)

    # Sigmoid: sigmoid_out = sigmoid(temp)
    sigmoid_out = pattern.add("SigmoidNode", "Sigmoid", inputs=[temp])

    # Multiplication: output = temp * sigmoid_out
    output = pattern.add("MulNode", "Mul", inputs=[temp, sigmoid_out])
    pattern.set_output_tensors([output])

    return pattern


def get_plugin_pattern():
    """
    SwiGLU plugin pattern with nested gemm-swish sub-pattern:
        Input (a)
          |
        Gemm1 (a, b1) -> c1
          |
        Gemm2 (a, b2) -> temp2
          |
        Swish (temp2) -> c2 (nested pattern)
          |
        Mul (c1, c2) -> d
          |
        Outputs: d, c2
    """
    pattern = gs.GraphPattern()

    # Inputs
    a = pattern.variable()  # Input tensor
    b1 = pattern.variable()  # First weight
    b2 = pattern.variable()  # Second weight

    # Check function to only match fp16 versions
    def check_fp16(node):
        """Check if the input tensor is fp16."""
        try:
            # Get the input tensor from the node
            if hasattr(node, "inputs") and len(node.inputs) > 0:
                input_tensor = node.inputs[0]
                # Check if the tensor has a dtype attribute and it's fp16
                if hasattr(input_tensor, "dtype"):
                    # For numpy dtypes, we need to check against numpy.float16
                    import numpy as np

                    return input_tensor.dtype == np.float16
            return True  # Default to matching if we can't determine
        except:
            return True  # Default to matching if there's an error

    # First GEMM: c1 = gemm(a, b1) with fp16 check
    c1 = pattern.add("Gemm1Node", "Gemm", inputs=[a, b1], check_func=check_fp16)

    # Second GEMM: temp2 = gemm(a, b2)
    temp2 = pattern.add("Gemm2Node", "Gemm", inputs=[a, b2])

    # Swish using nested pattern: c2 = swish(temp2)
    swish_pattern = get_swish_pattern()
    c2 = pattern.add("SwishSubPattern", swish_pattern, inputs=[temp2])

    # Final multiplication: d = mul(c1, c2)
    d = pattern.add("MulNode", "Mul", inputs=[c1, c2])

    # Set multiple outputs: d and c2
    pattern.set_output_tensors([d, c2])

    return pattern


def get_matching_subgraphs(graph) -> List[Dict[str, str]]:
    """Get matching subgraphs."""
    gp = get_plugin_pattern()
    matches = gp.match_all(graph)
    ans = []

    for m in matches:
        # Save the input and output tensor names of the matching subgraph(s)
        input_tensors = [ip_tensor.name for ip_tensor in m.inputs]
        output_tensors = [op_tensor.name for op_tensor in m.outputs]

        # Extract attributes from the Gemm nodes
        attrs = {}

        # Map Gemm attributes to plugin attributes
        attr_mapping = {
            "alpha": "alpha",
            "beta": "beta",
            "transA": "transA",
            "transB": "transB",
        }

        # Get attributes from first Gemm
        gemm1_node = m.get("Gemm1Node")
        if gemm1_node and gemm1_node.attrs:
            for attr_name, attr_value in gemm1_node.attrs.items():
                if attr_name in attr_mapping:
                    plugin_attr_name = attr_mapping[attr_name]
                    attrs[f"gemm1_{plugin_attr_name}"] = attr_value

        # Get attributes from second Gemm
        gemm2_node = m.get("Gemm2Node")
        if gemm2_node and gemm2_node.attrs:
            for attr_name, attr_value in gemm2_node.attrs.items():
                if attr_name in attr_mapping:
                    plugin_attr_name = attr_mapping[attr_name]
                    attrs[f"gemm2_{plugin_attr_name}"] = attr_value

        ioa = {"inputs": input_tensors, "outputs": output_tensors, "attributes": attrs}
        ans.append(ioa)

    return ans


def get_plugin_metadata() -> Dict[str, str]:
    return {
        "name": "swigluPlugin",
        "op": "SwiGLUPlugin",
    }


def replace_with_plugin(
    graph, input_tensors: list, output_tensors: list, attrs=None, op=None
):
    """
    Custom replacement method for SwiGLU plugin.
    This method can be used to implement custom replacement logic if needed.
    """
    # For this example, we'll use the default replacement method
    # But you could implement custom logic here if needed
    from polygraphy.tools.plugin.subtool.replace import default_replace_with_plugin

    return default_replace_with_plugin(graph, input_tensors, output_tensors, attrs, op)
