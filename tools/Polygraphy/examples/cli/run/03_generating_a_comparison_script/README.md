
# Generating A Script For Advanced Comparisons


## Introduction

For more advanced requirements, you may want to use the [API](../../../../polygraphy).
Instead of writing a script from scratch, you can use `run`'s `--gen-script` option
to create a Python script that you can use as a starting point.


## Running The Example

1. Generate a comparison script:

    ```bash
    polygraphy run identity.onnx --trt --onnxrt \
        --gen-script=compare_trt_onnxrt.py
    ```

    The generated script will do exactly what the `run` command would otherwise do.

2. Run the comparison script, optionally after modifying it:

    ```bash
    python3 compare_trt_onnxrt.py
    ```
