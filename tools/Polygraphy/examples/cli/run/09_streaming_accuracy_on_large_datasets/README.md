# Streaming Accuracy Comparison On Large Datasets

## Prerequisites
For a general overview of how to use `polygraphy run` to compare the outputs of
different frameworks, see the example on [Comparing Frameworks](../../../../examples/cli/run/01_comparing_frameworks).

## Introduction

By default, `polygraphy run` processes the dataset **one iteration at a time**: for each input,
inference is run on every runner, the outputs are compared immediately, only the (tiny) accuracy
metrics are accumulated, and the output tensors are discarded. Memory usage therefore stays roughly
constant no matter how large the dataset is. (To instead run each runner to completion with the
whole dataset held in memory, pass `--sequential-runners`.)

Two things to keep in mind:

* All runners are activated at the same time, so their device/host memory is held concurrently.
* When saving, an *extensionless* `--save-inputs`/`--save-outputs` path is a *directory* holding one
    JSON file per iteration (read back one iteration at a time, for constant memory); it must be empty
    or not yet exist. A path *with a file extension* instead holds the whole run in a single file.
    Either is read back with `--load-inputs`/`--load-outputs`.

In this example, we use a lazy [data loader](./data_loader.py) (which yields inputs one at a time
without materializing the whole dataset) to compare two runs of a model over a large dataset.

## Running The Example

1. Stream the dataset through two runs of the model and compare them as you go:

    ```bash
    polygraphy run identity.onnx --onnxrt --onnxrt \
        --data-loader-script data_loader.py \
        --check-error-stat mean
    ```

2. You can also save each iteration's inputs and outputs to disk while streaming, then load
    them back later to compare against. First, save a "golden" run:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --data-loader-script data_loader.py \
        --save-inputs inputs --save-outputs golden
    ```

    This writes one JSON file per iteration into the `inputs/` and `golden/` directories.

3. Now run the model again, streaming the saved inputs and comparing against the saved golden
    outputs — still one iteration at a time:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-inputs inputs --load-outputs golden
    ```

## Further Reading

For more on saving and comparing across runs (without streaming), see
[`run` example 02](../02_comparing_across_runs/). For loading custom input data, see
[`run` example 05](../05_comparing_with_custom_input_data/).
