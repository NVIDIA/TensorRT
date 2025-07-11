# Int8 Calibration In TensorRT


## Introduction

Int8 calibration in TensorRT involves providing a representative set of input data
to TensorRT as part of the engine building process. The
[calibration API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/infer/Int8/Calibrator.html)
included in TensorRT requires the user to handle copying input data to the GPU and
manage the calibration cache generated by TensorRT.

While the TensorRT API provides a higher degree of control, we can greatly simplify the
process for many common use-cases. For that purpose, Polygraphy provides a calibrator, which
can be used either with Polygraphy or directly with TensorRT. In the latter
case, the Polygraphy calibrator behaves exactly like a normal TensorRT int8 calibrator.

In this example, we'll look at how you can use Polygraphy's calibrator to calibrate a network
with (fake) calibration data, and how you can manage the calibration cache with just a single
parameter.


## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`

2. Run the example:

    ```bash
    python3 example.py
    ```

3. The first time you run the example, it will create a calibration cache
    called `identity-calib.cache`. If you run the example again, you should see that
    it now uses the cache instead of running calibration again:

    ```bash
    python3 example.py
    ```


## Further Reading

For more information on how int8 calibration works in TensorRT, see the
[developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#optimizing_int8_c)
