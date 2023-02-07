# Comparing With Custom Output Data

## Introduction

In some cases, it may be useful to compare against output values generated outside Polygraphy.
The simplest way to do so is to create a `RunResults` object and save it to a file.

This example illustrates how you can generate custom input and output data outside of Polygraphy
and seamlessly load it into Polygraphy for comparison.

## Running The Example

1. Generate the input and output data:

    ```bash
    python3 generate_data.py
    ```

2. **[Optional]** Inspect the data.
    For inputs:

    ```bash
    polygraphy inspect data custom_inputs.json
    ```

    For outputs:

    ```bash
    polygraphy inspect data custom_outputs.json
    ```

3. Run inference with the generated input data and then compare outputs against the custom outputs:

    ```bash
    polygraphy run identity.onnx --trt \
        --load-inputs custom_inputs.json \
        --load-outputs custom_outputs.json
    ```

## Further Reading

For details on how to access and work with the outputs stored in `RunResults` objects
using the Python API, refer to [API example 08](../../../api/08_working_with_run_results_and_saved_inputs_manually/).
