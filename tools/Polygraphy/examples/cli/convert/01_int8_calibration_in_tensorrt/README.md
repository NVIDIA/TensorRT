# Int8 Calibration In TensorRT

In [API example 04](../../../api/04_int8_calibration_in_tensorrt/), we saw how we can leverage
Polygraphy's included calibrator to easily run int8 calibration with TensorRT.

But what if we wanted to do the same thing on the command-line?

To do this, we need a way to supply custom input data to our command-line tools.
Polygraphy currently provides 2 ways to do so:

1. Using `--load-input-data`, which takes a path to a JSON file containing a `List[Dict[str, np.ndarray]]`.
    This will cause Polygraphy to load the entire object into memory, which would
    be impractical or even impossible for a large calibration set.

2. Using `--data-loader-script` which takes a path to a Python script that defines a `load_data` function
    that returns a data loader. The data loader can be any iterable or generator that yields
    `Dict[str, np.ndarray]`. By using a generator, we can avoid loading the entire calibration set
    at once, and instead limit it to just a single batch at a time.


We'll define a `load_data` function in a Python script called `data_loader.py` and
then use `polygraphy convert` to build the TensorRT engine.
Note that we could have just as easily used `polygraphy run` to build *and run* the engine.

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
