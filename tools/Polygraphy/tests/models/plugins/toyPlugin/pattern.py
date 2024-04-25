from polygraphy import mod
from typing import List,Dict
gs = mod.lazy_import("onnx_graphsurgeon>=0.5.0")

def get_plugin_pattern():
    """
    Toy plugin pattern:
        A     B
        \   /
          C, attrs['x'] < 2.0
        /   \
        D     E
    """
    pattern = gs.GraphPattern()
    in_0 = pattern.variable()
    in_1 = pattern.variable()
    a_out = pattern.add("Anode", "A", inputs=[in_0])
    b_out = pattern.add("Bnode", "B", inputs=[in_1])
    check_function = lambda node : node.attrs["x"] < 2.0
    c_out = pattern.add("Cnode", "C", inputs=[a_out, b_out], check_func=check_function)
    d_out = pattern.add("Dnode", "D", inputs=[c_out])
    e_out = pattern.add("Enode", "E", inputs=[c_out])
    pattern.set_output_tensors([d_out, e_out])

    return pattern

def get_matching_subgraphs(graph) -> List[Dict[str,str]]:
    gp = get_plugin_pattern()
    matches = gp.match_all(graph)
    ans = []
    for m in matches:
        # save the input and output tensor names of the matching subgraph(s)
        input_tensors = list(set([ip_tensor.name for ip_tensor in m.inputs]))
        output_tensors = list(set([op_tensor.name for op_tensor in m.outputs]))

        attrs = {"ToyX": int(m.get("Cnode").attrs["x"]) * 2}
        ioa = {
            'inputs':input_tensors,
            'outputs':output_tensors,
            'attributes':attrs
        }
        ans.append(ioa)
    return ans

def get_plugin_metadata() -> Dict[str,str]:
    return {'name':'toyPlugin',
            'op':'CustomToyPlugin',
            }
