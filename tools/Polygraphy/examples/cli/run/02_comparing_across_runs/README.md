# Comparing Across Runs

## Prerequisites
For a general overview of how to use `polygraphy run` to compare the outputs of
different frameworks, see the example on [Comparing Frameworks](../../../../examples/cli/run/01_comparing_frameworks).

## Introduction

There are situations where you may need to compare results across different invocations
of the `polygraphy run` command.  Some examples of this include:

* Comparing results across different platforms
* Comparing results across different versions of TensorRT
* Comparing different model types with compatible input(s)/output(s)

In this example, we'll demonstrate how to accomplish this with Polygraphy.

## Running The Example

### Comparing Across Runs

1. Save the input and output values from the first run:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-inputs inputs.json --save-outputs run_0_outputs.json
    ```

2. Run the model again, this time loading the saved inputs and outputs from
    the first run.  The saved inputs will be used as inputs for the current run, and
    the saved outputs will be used to compare against the first run.

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-inputs inputs.json --load-outputs run_0_outputs.json
    ```

    The `--atol/--rtol/--check-error-stat` options all work the same as in the
    [Comparing Frameworks](../../../../examples/cli/run/01_comparing_frameworks) example:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-inputs inputs.json --load-outputs run_0_outputs.json \
        --atol 0.001 --rtol 0.001 --check-error-stat median
    ```

### Comparing Different Models

We can also use this technique to compare different models, like TensorRT engines
and ONNX modles (if they have matching outputs).

1. Convert the ONNX model to a TensorRT engine and save it to disk:

    ```bash
    polygraphy convert identity.onnx -o identity.engine
    ```

2. Run the saved engine in Polygraphy, using the saved inputs from the ONNX-Runtime run as
    inputs to the engine, and compare the engine's outputs to the saved ONNX-Runtime outputs:

    ```bash
    polygraphy run --trt identity.engine --model-type=engine \
        --load-inputs inputs.json --load-outputs run_0_outputs.json
    ```


## Further Reading

For details on how to access and work with the saved outputs
using the Python API, refer to [API example 08](../../../api/08_working_with_run_results_and_saved_inputs_manually/).

For information on comparing against custom outputs, refer to [`run` example 06](../06_comparing_with_custom_output_data/).
