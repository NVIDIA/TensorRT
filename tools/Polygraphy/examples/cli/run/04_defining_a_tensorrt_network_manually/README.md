# Defining A TensorRT Network Manually

In some cases, it can be useful to define a TensorRT network from scratch using the Python API,
or modify a network created by other means (e.g. a parser). Normally, this would restrict you
from using CLI tools, at least until you build an engine, since the network cannot be serialized
to disk and loaded on the command-line.

However, Polygraphy CLI tools provide a work-around for this - if your Python script defines a function
named `load_network`, which takes no parameters and returns a TensorRT builder, network,
and optionally parser, then you can provide your Python script in place of a model argument.

*TIP: Instead of writing the network script from scratch, you can use `polygraphy template trt-network`*
    *to give you a starting point.*

In this example, the included `define_network.py` script parses an ONNX model and appends an identity
layer to it. Since it returns the builder, network, and parser in a function called `load_network`,
we can build and run a TensorRT engine from it using just a single command:

```bash
polygraphy run --trt define_network.py --model-type=trt-network-script
```

Similarly, we can define a TensorRT builder configuration using a script, provided that we define
a function called `load_config` that accepts a builder and network and returns a builder configuration:

```bash
polygraphy run --trt define_network.py --model-type=trt-network-script --trt-config-script=create_config.py
```

Note that we could have defined both `load_network` and `load_config` in the same script.
In fact, we could have retrieved these functions from arbitrary scripts, or even modules.
