# Using Extract To Isolate A Subgraph

The `surgeon extract` subtool can be used to extract a subgraph from a model with a single command.

In this example, we'll extract a subgraph from a simple 2-layer identity model.
We will assume that we have already used `inspect model` with `--mode` to determine that
the input tensor of the subgraph we want is `identity_out_0`, and the output tensor is `identity_out_2`.

Additionally, we'll assume that we don't know what the shapes or types of these tensors are,
and furthermore, that ONNX shape inference was not able to determine these either.

When shapes and types are unknown, you can use `auto` to indicate that `extract` should
attempt to automatically determine these.
For inputs, we must specify both shape and data type, whereas outputs only require the data
type (hence `--inputs` requires 2 `auto`s and `--outputs` requires only 1).

In our case, we can run:

```bash
polygraphy surgeon extract ../../../models/2_layer_identity.onnx \
    --inputs identity_out_0,auto,auto \
    --outputs identity_out_2,auto \
    -o subgraph.onnx
```

**NOTE:** When even ONNX shape inference cannot determine shapes, `extract` will run inference on the model
using input data (you can control the shape of this input data using the `--model-inputs` argument).
This will cause the inputs of the resulting subgraph to have fixed shapes. You can change
these back to dynamic by using the extract command again on the subgraph, and specifying the same inputs,
but using shapes with dynamic dimensions, e.g. `--inputs identity_out_0,-1x-1,auto`


If we knew the shapes and/or data types, we could instead write:

```bash
polygraphy surgeon extract ../../../models/2_layer_identity.onnx \
    --inputs identity_out_0,64x64,float32 \
    --outputs identity_out_2,float32 \
    -o subgraph.onnx
```

`subgraph.onnx` will be an ONNX model which contains only the subgraph whose input is
`identity_out_0` and whose output is `identity_out_2` - in this case, just a single node.


At this point, the model is ready for use. You can use `inspect` to confirm
whether it looks correct:

```bash
polygraphy inspect model subgraph.onnx --mode=basic
```
