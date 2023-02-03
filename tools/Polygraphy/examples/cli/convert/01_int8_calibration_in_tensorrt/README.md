# Int8 Calibration In TensorRT


## Introduction

In [API example 04](../../../api/04_int8_calibration_in_tensorrt/), we saw how we can leverage
Polygraphy's included calibrator to easily run int8 calibration with TensorRT.

But what if we wanted to do the same thing on the command-line?

To do this, we need a way to supply custom input data to our command-line tools.
Polygraphy provides multiple ways to do so, which are detailed [here](../../../../how-to/use_custom_input_data.md).

In this example, we'll use a data loader script by defining a `load_data` function in a Python
script called `data_loader.py` and then use `polygraphy convert` to build the TensorRT engine.

*TIP: We can use a similar approach with `polygraphy run` to build and run the engine.*

## Running The Example

1. Convert the model, using the custom data loader script to supply calibration data,
    saving a calibration cache for future use:

    ```bash
    polygraphy convert identity.onnx --int8 \
        --data-loader-script ./data_loader.py \
        --calibration-cache identity_calib.cache \
        -o identity.engine
    ```

2. **[Optional]** Rebuild the engine using the cache to skip calibration:

    ```bash
    polygraphy convert identity.onnx --int8 \
        --calibration-cache identity_calib.cache \
        -o identity.engine
    ```

    Since the calibration cache is already populated, calibration will be skipped.
    Hence, we do *not* need to supply input data.


3. **[Optional]** Use the data loader directly from the API example.

    The method outlined here is so flexible that we can even use the data loader we defined in the API example!
    We just need to specify the function name since the example does not call it `load_data`:

    ```bash
    polygraphy convert identity.onnx --int8 \
        --data-loader-script ../../../api/04_int8_calibration_in_tensorrt/example.py:calib_data \
        -o identity.engine
    ```
