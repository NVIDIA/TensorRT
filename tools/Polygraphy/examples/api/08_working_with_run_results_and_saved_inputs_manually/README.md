# Working With Run Results And Saved Inputs Manually

## Introduction

Inference inputs and outputs from `Comparator.run` can be serialized and saved to JSON
files so they can be reused. Inputs are stored as `List[Dict[str, np.ndarray]]` while outputs
are stored in a `RunResults` object, which can keep track of the outputs of multiple runners
from multiple inference iterations.

Command-line tools providing `--save-inputs` and `--save-outputs` options generally use these formats.

Usually, you'll only use saved inputs or `RunResults` with other Polygraphy APIs or
tools (as in [this example](../../cli//run/06_comparing_with_custom_output_data/)
or [this one](../../cli/inspect/05_inspecting_inference_outputs/)), but sometimes,
you may want to work with the underlying NumPy arrays manually.

Polygraphy includes convenience APIs that make it easy to load and manipulate these objects.

This example illustrates how you can load saved inputs and/or `RunResults` from a file
using the Python API and then access the NumPy arrays stored within.

## Running The Example

1. Generate some inference inputs and outputs:

    ```bash
    polygraphy run identity.onnx --trt --onnxrt \
        --save-inputs inputs.json --save-outputs outputs.json
    ```

2. **[Optional]** Use `inspect data` to view the inputs on the command-line:

    ```bash
    polygraphy inspect data inputs.json --show-values
    ```

3. **[Optional]** Use `inspect data` to view the outputs on the command-line:

    ```bash
    polygraphy inspect data outputs.json --show-values
    ```

4. Run the example:

    ```bash
    python3 example.py
    ```
