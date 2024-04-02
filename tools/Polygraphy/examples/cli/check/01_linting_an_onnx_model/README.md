# checking An ONNX Model


## Introduction

The `check lint` subtool validates ONNX Models and generates a JSON report detailing any bad/unused nodes or model errors.

## Running The Example

### Lint the ONNX model:

<!-- Polygraphy Test: XFAIL Start -->
```bash
polygraphy check lint bad_graph.onnx -o report.json
```
<!-- Polygraphy Test: XFAIL End -->
The output should look something like this:
```bash
[I] RUNNING | Command: polygraphy check lint bad_graph.onnx -o report.json
[I] Loading model: bad_graph.onnx
[E] LINT | Field 'name' of 'graph' is required to be non-empty.
[I] Will generate inference input data according to provided TensorMetadata: {E [dtype=float32, shape=(1, 4)],
     F [dtype=float32, shape=(4, 1)],
     G [dtype=int64, shape=(4, 4)],
     D [dtype=float32, shape=(4, 1)],
     C [dtype=float32, shape=(3, 4)],
     A [dtype=float32, shape=(1, 3)],
     B [dtype=float32, shape=(4, 4)]}
[E] LINT | Name: MatMul_3, Op: MatMul |  Incompatible dimensions for matrix multiplication
[E] LINT | Name: Add_0, Op: Add |  Incompatible dimensions
[E] LINT | Name: MatMul_0, Op: MatMul |  Incompatible dimensions for matrix multiplication
[W] LINT | Input: 'A' does not affect outputs, can be removed.
[W] LINT | Input: 'B' does not affect outputs, can be removed.
[W] LINT | Name: MatMul_0, Op: MatMul | Does not affect outputs, can be removed.
[I] Saving linting report to report.json
[E] FAILED | Runtime: 1.006s | Command: polygraphy check lint bad_graph.onnx -o report.json
```

- This will create a `report.json` that contains information about what's wrong with the model.
- The above example uses a faulty ONNX Model `bad_graph.onnx` that has multiple errors/warnings captured by the linter.
The errors are:
    1. Model has an empty name.
    2. Nodes `Add_0`, `MatMul_0` and `MatMul_3` have incompatible input shapes.
The warnings are:
    1. Inputs `A` and `B` are unused output.
    2. Node `MatMul_0` is unused by output.

### Example Report:

The generated report looks as follows:

<!-- Polygraphy Test: Ignore Start -->
```json
{
    "summary": {
        "passing": [
            "MatMul_1",
            "cast_to_int64",
            "NonZero"
        ],
        "failing": [
            "MatMul_0",
            "MatMul_3",
            "Add_0"
        ]
    },
    "lint_entries": [
        {
            "level": "exception",
            "source": "onnx_checker",
            "message": "Field 'name' of 'graph' is required to be non-empty."
        },
        {
            "level": "exception",
            "source": "onnxruntime",
            "message": " Incompatible dimensions for matrix multiplication",
            "nodes": [
                "MatMul_3"
            ]
        },
        {
            "level": "exception",
            "source": "onnxruntime",
            "message": " Incompatible dimensions",
            "nodes": [
                "Add_0"
            ]
        },
        {
            "level": "exception",
            "source": "onnxruntime",
            "message": " Incompatible dimensions for matrix multiplication",
            "nodes": [
                "MatMul_0"
            ]
        },
        {
            "level": "warning",
            "source": "onnx_graphsurgeon",
            "message": "Input: 'A' does not affect outputs, can be removed."
        },
        {
            "level": "warning",
            "source": "onnx_graphsurgeon",
            "message": "Input: 'B' does not affect outputs, can be removed."
        },
        {
            "level": "warning",
            "source": "onnx_graphsurgeon",
            "message": "Does not affect outputs, can be removed.",
            "nodes": [
                "MatMul_0"
            ]
        }
    ]
}
```
<!-- Polygraphy Test: Ignore End -->

### Notes
Since it runs ONNX Runtime under the hood, it is possible to specify execution providers using `--providers`. Defaults to CPU.

It is also possible to override the input shapes using `--input-shapes`, or provide custom input data. For more details, refer [how-to/use_custom_input_data](../../../../how-to/use_custom_input_data.md).

For more information on usage, use `polygraphy check lint --help`.
