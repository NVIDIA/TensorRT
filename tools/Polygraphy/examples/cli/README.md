# Polygraphy CLI Examples

This directory includes examples that use the Polygraphy CLI.
For examples of the Python API, see the [api](../api/) directory instead.

## Common Topics

This section covers some concepts that are common to multiple tools.

### Using Custom Input Data

For tools that require input data, such as `run`, Polygraphy currently
provides 2 ways to use custom input data:

1. `--load-input-data`, which takes a path to a JSON file containing a `List[Dict[str, np.ndarray]]`.
    This will cause Polygraphy to load the entire object into memory.
    *NOTE: This may be impractical or even impossible if the data is very large.*

2. `--data-loader-script`, which takes a path to a Python script that defines a `load_data` function
    that returns a data loader. The data loader can be any iterable or generator that yields
    `Dict[str, np.ndarray]`. By using a generator, we can avoid loading all the data
    at once, and instead limit it to just a single input at a time.

    *TIP: If you have an existing script that already defines such a function, you do **not** need to create*
        *a separate script just for the sake of `--data-loader-script`. You can simply use the existing script*
        *and optionally use the `--data-loader-func-name` argument to specify the name of the function (if it's not `load_data`)*

See [`run` example 05](run/05_comparing_with_custom_data) for details.

### Defining A Custom TensorRT Network

Many of the command-line tools involve creating TensorRT networks. In most cases, these
are created by parsing a model from a framework (generally in ONNX format). However, it
is also possible to define the TensorRT network manually using a Python script.

1. To get started, generate a template script with:

    ```bash
    polygraphy template trt-network -o define_network.py
    ```

    If you want to start from a model and modify the resulting TensorRT network instead
    of creating one from scratch, simply provide the model as an argument to `template trt-network`.
    For example, for an ONNX model:

    ```bash
    polygraphy template trt-network </path/to/model.onnx> -o define_network.py
    ```

2. Once you've filled out the body of the `load_network` function in `define_network.py`,
    you can use it in the tools by providing the script in place of a model argument.
    For example:

    ```bash
    polygraphy run define_network.py --model-type=trt-network-script --trt
    ```

See [`run` example 04](run/04_defining_a_tensorrt_network_manually) for details.


### Defining a Custom TensorRT Builder Configuration

Similar to defining custom TensorRT networks, it is possible to provide custom
TensorRT builder configurations on the command-line using a Python script.

See [`run` example 04](run/04_defining_a_tensorrt_network_manually) for details.
