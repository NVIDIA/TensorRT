# Inspecting TensorRT ONNX Support

## Introduction

The `inspect capability` subtool provides detailed information on TensorRT's ONNX operator support for a given ONNX graph.
It is also able to partition and save supported and unsupported subgraphs from the original model in order to report all the dynamically checked errors with a given model.

## Running The Example

1. Generate the capability report

    ```bash
    polygraphy inspect capability --with-partitioning model.onnx
    ```

2. This should display a summary table like:

    ```
    [I] ===== Summary =====
        Operator | Count   | Reason                                                                                                                                                                    | Nodes
        -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Fake     |       1 | In node 0 with name:  and operator: Fake (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[2, 3]]
    ```

## Understanding The Output

In this example, `model.onnx` contains a `Fake` node that is not supported by TensorRT.
The summary table shows the unsupported operator, the reason it's unsupported, how many times it appears in the graph,
and the index range of these nodes in the graph in case there are multiple unsupported nodes in a row.
Note that this range uses an inclusive start index and an exclusive end index.

It is important to note that the graph partitioning logic (`--with-partitioning`) currently does not support surfacing issues with nodes inside local functions (`FunctionProto`s). See the description of the default flow (without `--with-partitioning` option, described in the example `09_inspecting_tensorrt_static_onnx_support`) for static error reporting that properly handles nodes inside local functions.

For more information and options, see `polygraphy inspect capability --help`.
