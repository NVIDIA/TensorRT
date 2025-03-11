# Inspecting TensorRT ONNX Support

## Introduction

The `inspect capability` subtool provides detailed information on TensorRT's ONNX operator support for a given ONNX graph.
It is also able to partition and save supported and unsupported subgraphs from the original model in order to report all the dynamically checked errors with a given model (see the example `08_inspecting_tensorrt_onnx_support`).

## Running The Example

1. Generate the capability report

    ```bash
    polygraphy inspect capability nested_local_function.onnx
    ```

2. This should display a summary table like:

    ```
    [I] ===== Summary =====
        Stack trace                                                                             | Operator  | Node               | Reason
        -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        onnx_graphsurgeon_node_1 (OuterFunction) -> onnx_graphsurgeon_node_1 (NestedLocalFake2) | Fake_2    | nested_node_fake_2 | In node 0 with name: nested_node_fake_2 and operator: Fake_2 (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"
        onnx_graphsurgeon_node_1 (OuterFunction)                                                | Fake_1    | nested_node_fake_1 | In node 0 with name: nested_node_fake_1 and operator: Fake_1 (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"
    ```

## Understanding The Output

In this example, `nested_local_function.onnx` contains `Fake_1` and `Fake_2` nodes that are not supported by TensorRT. `Fake_1` node is located inside a local function `OuterFunction` and `Fake_2` node is located inside a nested local function, `NestedLocalFake2`.
The summary table shows the current stack trace consisting of local functions, the operator in which the error occurred and the reason it's unsupported.

For more information and options, see `polygraphy inspect capability --help`.
