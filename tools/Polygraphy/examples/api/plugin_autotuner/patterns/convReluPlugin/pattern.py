from polygraphy import mod

gs = mod.lazy_import("onnx_graphsurgeon>=0.5.0")
from typing import List, Dict


def get_plugin_pattern():
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
                    attrs[attr_mapping[attr_name]] = attr_value

        ans.append(
            {"inputs": input_tensors, "outputs": output_tensors, "attributes": attrs}
        )
    return ans


def get_plugin_metadata() -> Dict[str, str]:
    return {"name": "convReluPlugin", "op": "ConvReluPlugin"}


def replace_with_plugin(
    graph, input_tensors: list, output_tensors: list, attrs=None, op=None
):
    from polygraphy.tools.plugin.subtool.replace import default_replace_with_plugin

    return default_replace_with_plugin(graph, input_tensors, output_tensors, attrs, op)
