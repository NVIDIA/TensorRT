from polygraphy import mod
gs = mod.lazy_import("onnx_graphsurgeon>=0.5.0")

def get_plugin_pattern() -> gs.GraphPattern:
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

def get_plugin_attributes(sg) -> dict:
    """
    example plugin attribute mapping, where the plugin has attribute ToyX, which gets its value from C.x * 2
    """
    return {"ToyX": int(sg.get("Cnode").attrs["x"]) * 2}

