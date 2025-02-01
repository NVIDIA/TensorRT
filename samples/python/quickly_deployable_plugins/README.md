# Quickly Deployable TensorRT Python Plugins [Experimental]

This is a sample to showcase quickly deployable Python-based plugin definitions (QDPs) in TRT. QDPs are able to support a large majority of use cases for adding custom operators to TRT, and will be the recommended option when it becomes a stable feature.

# Introduction

While the regular TRT plugin interfaces are powerful in the flexibility and tunability they provide, for the vast majority of use cases, users will benefit from the simplicity offered by the QDP workflow.
 - The `tensorrt.plugin` module provides many intuitive APIs that drastically reduces the amount of boilerplate required to implement a plugin
 - The concept of plugin registration, plugin creators and the plugin registry is abstracted away
 - The stateless nature of QDPs all but eliminates the complications of having to comply with a predefined plugin lifecycle

This sample contains several mini-samples that demonstrate a few common use cases.

# Setting Up The Environment

To build and install the bindings, follow the instructions in `$TRT_OSSPATH/python/README.md`.

Then install the requisite packages
```bash
cd $TRT_OSSPATH/samples/python/quickly_deployable_plugins
pip3 install -r requirements.txt
```

# Implementing a quickly deployable Python plugin

QDP definitions consist of a set of decorated functions that define properties and behaviors of the plugin.
 - `@tensorrt.plugin.register`: Returns shape and type characteristics of output tensors, and any attributes the plugin needs to function.
 - `@tensorrt.plugin.impl`: Performs the plugin computation
 - (Optional) `@tensorrt.plugin.autotune`: Defines the different data types and formats (tensor layouts) supported by the plugin's IO and any tactics supported by the plugin. Defining this function allows TensorRT to "tune" the plugin during the engine build to find the most performant type/format and tactic combination on the target system.

The specifics of these functions will become clear through the following mini-samples.

# A Simple Plugin: Elementwise-Add

This mini-sample contains an elementwise addition plugin, where the computation is being performed with an OpenAI Triton kernel. Let's first take a look at the `tensorrt.plugin.register` function.

```python
import tensorrt.plugin as trtp

@trtp.register("sample::elemwise_add_plugin")
def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> trtp.TensorDesc:
    return inp0.like()
```

The argument "sample::elemwise_add_plugin" defines the namespace ("sample") and name ("elemwise_add_plugin") of the plugin. Input arguments to the decorated function (`plugin_desc`) annotated with `trt.plugin.TensorDesc` denote the input tensors; all others are interpreted as plugin attributes (see the [TRT API Reference](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/tensorrt.plugin/trt_plugin_register.html) for a full list of allowed attribute types). The output signature is a `trt.plugin.TensorDesc` describing the output. `inp0.like()` returns a tensor descriptor with identical shape and type characteristics to `inp0`.

The computation function, decorated with `trt.plugin.impl`, receives `trt.plugin.Tensor`s for each input and output. In contrast to `TensorDesc`s, a `Tensor` references an underlying data buffer, directly accessible through `Tensor.data_ptr`. When working with Torch and OpenAI Triton kernels, it is easier to use `torch.as_tensor()` to zero-copy construct a `torch.Tensor` corresponding to the `trt.plugin.Tensor`.

This sample also showcases the effect of omitting/defining a `trt.plugin.autotune` function, which must return a list of `trt.plugin.AutoTuneCombination`s. In this case, we define a single combination `AutoTuneCombination("FP32|FP16, FP32|FP16")`; this indicates that the input and output must be either both FP32 or both FP16. See the TRT API Reference for a detailed description of the grammar underlying `AutoTuneCombination`s.

## Running the sample

```bash
python3 qdp_runner.py add [--autotune] [-v]
```

`--autotune` simulates having defined a `trt.plugin.autotune` function. Enabling verbose logging (`-v`) is recommended to see the effect of autotuning. It can be observed that the `trt.plugin.impl` function is invoked several times during the engine build process when autotune is enabled. With autotuning turned off, `trt.plugin.impl` is invoked only once (when inference is run after building the engine).

```bash
$ python3 qdp_runner.py add --autotune -v
...
Executing for inp0.dtype=DataType.FLOAT and output[0].dtype=DataType.FLOAT
Executing for inp0.dtype=DataType.FLOAT and output[0].dtype=DataType.FLOAT
Executing for inp0.dtype=DataType.FLOAT and output[0].dtype=DataType.FLOAT
Executing for inp0.dtype=DataType.FLOAT and output[0].dtype=DataType.FLOAT
Executing for inp0.dtype=DataType.HALF and output[0].dtype=DataType.HALF
Executing for inp0.dtype=DataType.HALF and output[0].dtype=DataType.HALF
Executing for inp0.dtype=DataType.HALF and output[0].dtype=DataType.HALF
Executing for inp0.dtype=DataType.HALF and output[0].dtype=DataType.HALF
[I] Finished engine building in 1.073 seconds
Executing for inp0.dtype=DataType.HALF and output[0].dtype=DataType.HALF
```

# Implementing in-place custom ops with I/O aliasing

In-place computations can be accomplished with TRT plugins via aliased I/O. i.e. An input that needs to be modified in-place can be represented by an input-output pair, where the output is aliased to the input. For example, if in-place addition is needed (instead of the out-of-place addition of the above sample), that can be achieved as below:
```python
import tensorrt.plugin as trtp

@trtp.register("sample::elemwise_add_plugin_")
def add_plugin_desc_(inp0: trtp.TensorDesc) -> trtp.TensorDesc:
    return inp0.aliased()
```

Note the use of `trt.plugin.TensorDesc.aliased()` to produce an output `TensorDesc` that is aliased to `inp0`.

To appreciate the effect of aliasing better, this sample adds two in-place add plugins chained together.

## Running the sample

Enabling verbose logging (`-v`) is recommended to see the effect of autotuning, which is always enabled.

```bash
python3 qdp_runner.py inplace_add [--autotune] [-v]
```

# An op with data-dependent output shapes: Non-zero

Non-zero is an operation where the indices of the non-zero elements of the input tensor is found -- it has data-dependent output shapes (DDS). As such, typical shape calculations cannot be done with input shapes.

To handle DDS, the extent of each data-dependent output dimension must be expressed in terms of a *_size tensor_*, which is a scalar that communicates to TRT an upper-bound and an autotune value for that dimension, in terms of the input shapes. The TRT engine build may be optimized for the autotune value, but the extent of that dimension may stretch up to the upper-bound at runtime.

In this sample, we consider a 2D input tensor `inp0`; the output will be an $N x 2$ tensor (a set of $N$ 2D indices), where $N$ is the number of non-zero indices. At maximum, all elements could be non-zero, and so the upper-bound could be expressed as `upper_bound = inp0.shape_expr[0] * inp0.shape_expr[1]`. Note that `trt.plugin.TensorDesc.shape_expr` returns symbolic shape expressions for that tensor. Arithmetic operations on shape expressions are supported through standard Python binary operators (see [TRT Python API reference](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/tensorrt.plugin/Shape/ShapeExpr.html) for full list of supported operations).

On average, we can expect half of the input to be filled with zero, so a size tensor can be constructed with that as the autotune value:
```python
st = trtp.size_tensor(opt = upper_bound // 2, upper_bound = upper_bound)
```

Now we're ready to construct the output shape. `st.expr()` returns a shape expression for the size tensor, so a tensor descriptor for the output shape can be constructed as `trt.plugin.from_shape_expr((st.expr(), 2), dtype=trt.int32)`. TRT requires that any size tensors also be made outputs of the plugin. Putting things together, we arrive at the following:

```python
import tensorrt.plugin as trtp

@trtp.register("sample::non_zero_plugin")
def non_zero_plugin_reg(
    inp0: trtp.TensorDesc,
) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    upper_bound = inp0.shape_expr[0] * inp0.shape_expr[1]
    st = trtp.size_tensor(upper_bound // 2, upper_bound)
    return trtp.from_shape_expr((st.expr(), 2), dtype=trt.int32), st
```

## Running the sample

Enabling verbose logging (`-v`) is recommended to see the effect of autotuning, which is always enabled.

```bash
python3 qdp_runner.py non_zero [-v]
```

# Using multiple tactics and ONNX: Cirular padding

This sample contains a circular padding plugin, which is useful for ops like circular convolution. It is equivalent to PyTorch's [torch.nn.CircularPad2d](https://pytorch.org/docs/stable/generated/torch.nn.CircularPad2d.html#torch.nn.CircularPad2d).

Refer [this section about circular padding plugin](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/pluginGuide.html#example-circular-padding-plugin) in the Python plugin guide for more info.

## ONNX model with a plugin

It is often useful to run an ONNX node with a custom op through a TRT plugin that you have written. To allow the TRT ONNX parser to correctly recognize your plugin as being mapped to an ONNX node, ensure that
 - The `op` property of the node is exactly the same as your plugin name.
 - The node contains a string attribute called "plugin_namespace" with the namespace of your plugin.

In this sample, we define a plugin with the ID "sample::circ_pad_plugin", so if using ONNX Graphsurgeon, the custom op node can be constructed as follows:

```python
import onnx_graphsurgeon as gs

var_x = gs.Variable(name="x", shape=inp_shape, dtype=np.float32)
var_y = gs.Variable(name="y", dtype=np.float32)

circ_pad_node = gs.Node(
    name="circ_pad_plugin",
    op="circ_pad_plugin",
    inputs=[var_x],
    outputs=[var_y],
    attrs={"pads": pads, "plugin_namespace": "sample"},
)
```

## Multiple tactics

Sometimes, you may have multiple kernels (or backends) that can be used to perform the computation of the plugin -- these are typically called *_tactics_*. If it cannot be predetermined which of these tactics may perform the fastest, it is possible to let TRT time the plugin for each tactic and determine which one is fastest.

Communicating the availability of multiple tactics can simply be done through the `trt.plugin.autotune` function.
```python
import tensorrt.plugin as trtp
from enum import IntEnum

class Tactic(IntEnum):
    TORCH = 1
    TRITON = 2

@trt.plugin.autotune("sample::circ_pad_plugin")
def circ_pad_plugin_autotune(inp0: trtp.TensorDesc, pads: npt.NDArray[np.int32], outputs: Tuple[trtp.TensorDesc]) -> List[trtp.AutoTuneCombination]:
    c = trtp.AutoTuneCombination()
    c.pos([0, 1], "FP32|FP16")
    c.tactics([int(Tactic.TORCH), int(Tactic.TRITON)])
    return [c]
```

Note that we're using another way of constructing a `trt.plugin.AutoTuneCombination` here -- namely, through `pos(...)` to populate the type/format information and `tactics(...)` to specify the tactics. In this sample, we use an OpenAI Triton kernel and `torch.nn.functional.pad` as two methods to compute the circular padding.

## Loading and running a TRT engine containing a plugin

If you have a TRT engine built with a plugin, executing that engine only requires the plugin definitions for `trt.plugin.register` and `trt.plugin.impl` to be available in the module where the engine is being deserialized (note: the `trt.plugin.autotune` definition is not required to be present).

To simulate the loading of an engine, first run this sample with the `--save_engine` flag, followed by `--artifacts_dir [dir]` with a directory in which you wish the engine to be saved. Then run the sample again with `--load engine` and `--artifacts_dir` set to the same directory.

## Running the sample

```bash
python3 qdp_runner.py circ_pad [--multi_tactic] [--save_engine] [--load_engine] --mode {onnx,inetdef} [--artifacts_dir ARTIFACTS_DIR]  [-v]

options:
  --multi_tactic        Enable multiple tactics.
  --save_engine         Save engine to the artifacts_dir.
  --load_engine         Load engine from the artifacts_dir. Ignores all other options.
  --artifacts_dir ARTIFACTS_DIR
                        Whether to store (or retrieve) artifacts.
  --mode {onnx,inetdef} Whether to use ONNX parser or INetworkDefinition APIs to construct the network.
  -v, --verbose         Enable verbose log output.
```

# Additional resources


**Python Plugin Guide**
- [pluginGuide.md](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/pluginGuide.html)

**`tensorrt.plugin` API reference**
- [`tensorrt.plugin` module API reference](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/tensorrt.plugin/index.html)

**Guide to TensorRT plugins**
- [Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

October 2024: Initial release of this sample

# Known issues

There are no known issues in this sample
