# Defining A TensorRT Network Or Config Manually


## Introduction

In some cases, it can be useful to define a TensorRT network from scratch using the Python API,
or modify a network created by other means (e.g. a parser). Normally, this would restrict you
from using CLI tools, at least until you build an engine, since the network cannot be serialized
to disk and loaded on the command-line.

Polygraphy CLI tools provide a work-around for this - if your Python script defines a function
named `load_network`, which takes no parameters and returns a TensorRT builder, network,
and optionally parser, then you can provide your Python script in place of a model argument.

Similarly, we can create a custom TensorRT builder configuration using a script that defines
a function called `load_config` which accepts a builder and network and returns a builder configuration.

In this example, the included `define_network.py` script parses an ONNX model and appends an identity
layer to it. Since it returns the builder, network, and parser in a function called `load_network`,
we can build and run a TensorRT engine from it using just a single command. The `create_config.py`
script creates a new TensorRT builder configuration and enables FP16 mode.


### TIP: Generating Script Templates Automatically

Instead of writing the network script from scratch, you can use
`polygraphy template trt-network` to give you a starting point:

```bash
polygraphy template trt-network -o my_define_network.py
```

If you want to start from a model and modify the resulting TensorRT network instead
of creating one from scratch, simply provide the model as an argument to `template trt-network`:

```bash
polygraphy template trt-network identity.onnx -o my_define_network.py
```

Similarly, you can generate a template script for the config using `polygraphy template trt-config`:

```bash
polygraphy template trt-config -o my_create_config.py
```

You can also specify builder configuration options to pre-populate the script.
For example, to enable FP16 mode:

```bash
polygraphy template trt-config --fp16 -o my_create_config.py
```


## Running The Example

1. Run the network defined in `define_network.py`:

    ```bash
    polygraphy run --trt define_network.py --model-type=trt-network-script
    ```

2. Run the network from step (1) using the builder configuration defined in `create_config.py`:

    ```bash
    polygraphy run --trt define_network.py --model-type=trt-network-script --trt-config-script=create_config.py
    ```

    Note that we could have defined both `load_network` and `load_config` in the same script.
    In fact, we could have retrieved these functions from arbitrary scripts, or even modules.

*TIP: We can use the same approach with `polygraphy convert` to build, but not run, the engine.*
