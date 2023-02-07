# Extending `polygraphy run`

## Introduction

`polygraphy run` allows you to run inference with multiple backends, including TensorRT and ONNX-Runtime, and compare outputs.
While it does provide mechanisms to load and compare against custom outputs from unsupported backends,
adding support for the backend via an extension module allows it to be integrated more seamlessly,
providing a better user experience.

In this example, we'll create an extension module for `polygraphy run` called `polygraphy_reshape_destroyer`,
which will include the following:

- A special loader that will replace no-op `Reshape` nodes in an ONNX model with `Identity` nodes.

- A custom runner that supports ONNX models containing only `Identity` nodes.

- Command-line options to:
    - Enable or disable renaming nodes when a transformation is applied by the loader.
    - Run the model in `slow`, `medium`, or `fast` mode.
        In `slow` and `medium` modes, we'll inject a `time.sleep()` during inference
        (this will result in massive performance gains in `fast` mode!).

## Background

Although this example is self-contained and concepts will be explained as you encounter them, it is still
recommended that you first familiarize yourself with
[Polygraphy's `Loader` and `Runner` APIs](../../../polygraphy/README.md),
the [`Argument Group` Interface](../../../polygraphy/tools/args/README.md),
as well as the [`Script` interface](../../../polygraphy/tools/script.py).

After that, creating an extension module for `polygraphy run` is a simple matter of defining your
custom `Loader`s/`Runner`s and `Argument Group`s and making them visible to Polygraphy via
`setuptools`'s `entry_points` API.

*NOTE: Defining a custom `Loader` is not strictly required, but will be covered in this example for the sake of completeness.*

As a matter of convention, Polygraphy extension module names are prefixed with `polygraphy_`.

## Reading The Example Code

We've structured our example extension module such that it somewhat mirrors the structure of the Polygraphy repository.
This should make it easier to see the parallels between functionality in the extension module and that provided by Polygraphy natively.
The structure is:
<!-- Polygraphy Test: Ignore Start -->
```bash
- extension_module/
    - polygraphy_reshape_destroyer/
        - backend/
            - __init__.py   # Controls submodule-level exports
            - loader.py     # Defines our custom loader.
            - runner.py     # Defines our custom runner.
        - args/
            - __init__.py   # Controls submodule-level exports
            - loader.py     # Defines command-line argument group for our custom loader.
            - runner.py     # Defines command-line argument group for our custom runner.
        - __init__.py       # Controls module-level exports
        - export.py         # Defines the entry-point for `polygraphy run`.
    - setup.py              # Builds our module
```
<!-- Polygraphy Test: Ignore End -->

It is recommended that you read these files in the following order:

1. [backend/loader.py](./extension_module/polygraphy_reshape_destroyer/backend/loader.py)
2. [backend/runner.py](./extension_module/polygraphy_reshape_destroyer/backend/runner.py)
3. [backend/\_\_init\_\_.py](./extension_module/polygraphy_reshape_destroyer/backend/__init__.py)
4. [args/loader.py](./extension_module/polygraphy_reshape_destroyer/args/loader.py)
5. [args/runner.py](./extension_module/polygraphy_reshape_destroyer/args/runner.py)
6. [args/\_\_init\_\_.py](./extension_module/polygraphy_reshape_destroyer/args/__init__.py)
7. [\_\_init\_\_.py](./extension_module/polygraphy_reshape_destroyer/__init__.py)
8. [export.py](./extension_module/polygraphy_reshape_destroyer/export.py)
9. [setup.py](./extension_module/setup.py)


## Running The Example

1. Build and install the extension module:

    Build using `setup.py`:

    ```bash
    python3 extension_module/setup.py bdist_wheel
    ```

    Install the wheel:

    <!-- Polygraphy Test
        *NOTE: For tests, 'protobuf==3.19.4 onnx==1.10.0' is required to work around compatibility*
            *breakages in more recent versions of protobuf*
        ```bash
        python3 -m pip install protobuf==3.19.4 onnx==1.10.0
        ```
     Polygraphy Test -->

    ```bash
    python3 -m pip install extension_module/dist/polygraphy_reshape_destroyer-0.0.1-py3-none-any.whl \
        --extra-index-url https://pypi.ngc.nvidia.com
    ```

    *TIP: If you make changes to the example extension module, you can update your installed version by*
    *rebuilding (by following step 1) and then running:*

    ```bash
    python3 -m pip install extension_module/dist/polygraphy_reshape_destroyer-0.0.1-py3-none-any.whl \
        --force-reinstall --no-deps
    ```

2. Once the extension module is installed, you should see the options you added appear in the help output
    of `polygraphy run`:

    ```bash
    polygraphy run -h
    ```

3. Next, we can try out our custom runner with an ONNX model containing a no-op Reshape:

    ```bash
    polygraphy run no_op_reshape.onnx --res-des
    ```

4. We can also try some of the other command-line options we added:

    - Renaming replaced nodes:

        ```bash
        polygraphy run no_op_reshape.onnx --res-des --res-des-rename-nodes
        ```

    - Different inference speeds:

        ```bash
        polygraphy run no_op_reshape.onnx --res-des --res-des-speed=slow
        ```

        ```bash
        polygraphy run no_op_reshape.onnx --res-des --res-des-speed=medium
        ```

        ```bash
        polygraphy run no_op_reshape.onnx --res-des --res-des-speed=fast
        ```

5. Lastly, let's compare our implementation against ONNX-Runtime to make sure it is functionally correct:

    ```bash
    polygraphy run no_op_reshape.onnx --res-des --onnxrt
    ```
