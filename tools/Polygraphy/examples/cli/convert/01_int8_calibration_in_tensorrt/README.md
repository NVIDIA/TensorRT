# Int8 Calibration In TensorRT


## Introduction

In [API example 04](../../../api/04_int8_calibration_in_tensorrt/), we saw how we can leverage
Polygraphy's included calibrator to easily run int8 calibration with TensorRT.

But what if we wanted to do the same thing on the command-line?

To do this, we need a way to supply custom input data to our command-line tools.
Polygraphy provides multiple ways to do so, which are detailed [here](../../../cli#using-custom-input-data).

In this example, we'll use a data loader script by defining a `load_data` function in a Python
script called `data_loader.py` and then use `polygraphy convert` to build the TensorRT engine.

*TIP: We can use the same approach with `polygraphy run` to build and run the engine.*

## Running The Example

1. Convert the model, using the custom data loader script to supply calibration data:

    ```bash
    polygraphy convert identity.onnx --int8 \
        --data-loader-script ./data_loader.py \
        -o identity.engine
    ```

    In fact, this system is so flexible, we can even use the data loader we defined in the API example!
    We just need to set `--data-loader-func-name` since the example does not use `load_data` as the function name.


    ```bash
    polygraphy convert identity.onnx --int8 \
        --data-loader-script ../../../api/04_int8_calibration_in_tensorrt/example.py \
        --data-loader-func-name calib_data \
        -o identity.engine
    ```
