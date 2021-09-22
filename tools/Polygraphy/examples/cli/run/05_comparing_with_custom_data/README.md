# Comparing With Custom Input Data

## Introduction

In some cases, we may want to run comparisons using custom input data.
Polygraphy provides multiple ways to do so, which are detailed [here](../../../cli#using-custom-input-data).

In this example, we'll use a data loader script by defining a `load_data` function in a Python
script called `data_loader.py` and then use `polygraphy run` to compare TensorRT and ONNX-Runtime.

Since our model has dynamic shapes, we'll need to set up a TensorRT Optimization Profile.
For details on how we can do this via the command-line,
see [`convert` example 03](../..//convert/03_dynamic_shapes_in_tensorrt).
For simplicitly, we'll create a profile where `min` == `opt` == `max`.

*NOTE: It is important that our optimization profile works with the shapes provided by our*
    *custom data loader. In our very simple case, the data loader always generates inputs of*
    *shape (1, 2, 28, 28), so we just need to ensure this falls within [`min`, `max`].*

## Running The Example

1. Run the model with TensorRT and ONNX-Runtime using custom input data:

    ```bash
    polygraphy run dynamic_identity.onnx --trt --onnxrt \
        --data-loader-script data_loader.py \
        --trt-min-shapes X:[1,2,28,28] --trt-opt-shapes X:[1,2,28,28] --trt-max-shapes X:[1,2,28,28]
    ```
