# Inspecting TensorRT ONNX Support

## Introduction

The `inspect capability` subtool provides detailed information on TensorRT's ONNX operator support for a given ONNX graph.
It also partitions and saves supported and unsupported subgraphs from the original model.


## Running The Example

1. Generate the capability report

    ```bash
    polygraphy inspect capability model.onnx
    ```

    This should display a summary table like:

    ```
    [I] ===== Summary =====
        Operator | Count   | Reason                                                                                                                                                            | Nodes
        -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Fake     |       1 | In node 0 (importFallbackPluginImporter): UNSUPPORTED_NODE: Assertion failed: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[2, 3]]
    ```

## Understanding The Output

In this example, `model.onnx` contains a `Fake` node that is not supported by TensorRT.
The summary table shows the unsupported operator, the reason it's unsupported, how many times it appears in the graph,
and the index range of these nodes in the graph in case there are multiple unsupported nodes in a row.
Note that this range uses an inclusive start index and an exclusive end index.

For more information and options, see `polygraphy inspect capability --help`.
