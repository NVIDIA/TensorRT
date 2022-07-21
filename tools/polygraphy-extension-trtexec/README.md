# Extending `polygraphy run` to support `trtexec`

## Introduction

`polygraphy run` allows you to run inference with multiple backends, including TensorRT and ONNX-Runtime, and compare outputs.
This extension adds support to run inference with `trtexec`.

## Installation

Follow the steps below to install the extension module. After installation, you should see the `trtexec` options in the help
output of `polygraphy run`:

1. Build using `setup.py`:

    ```bash
    python3 setup.py bdist_wheel
    ```

2. Install the wheel: The wheel is installed in the `dist` directory. Install the wheel by running the following command    
    ```
    python3 -m pip install dist/polygraphy_trtexec-*.whl \
        --extra-index-url https://pypi.ngc.nvidia.com
    ```
    *NOTE: You may have to update the above command to install the appropriate version of the wheel*

3. After the installation, you can run it on the trtexec backend by using the `--trtexec` flag as follows:

    ```bash
    polygraphy run sample.onnx --trtexec
    ```

