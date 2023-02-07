# ONNX GraphSurgeon


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Examples](#examples)
- [Understanding The Basics](#understanding-the-basics)
    - [Importers](#importers)
    - [IR](#ir)
        - [Tensor](#tensor)
        - [Node](#node)
        - [A Note On Modifying Inputs And Outputs](#a-note-on-modifying-inputs-and-outputs)
        - [Graph](#graph)
    - [Exporters](#exporters)
- [Advanced](#advanced)
    - [Working With Models With External Data](#working-with-models-with-external-data)

## Introduction

ONNX GraphSurgeon is a tool that allows you to easily generate new ONNX graphs, or modify existing ones.


## Installation

### Using Prebuilt Wheels
```bash
python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
```

### Building From Source

#### Using Make Targets
```
make install
```

#### Building Manually

1. Build a wheel:
```
make build
```

2. Install the wheel manually from **outside** the repository:
```
python3 -m pip install onnx_graphsurgeon/dist/onnx_graphsurgeon-*-py2.py3-none-any.whl
```


## Examples

The [examples](./examples) directory contains several examples of common use-cases of ONNX GraphSurgeon.

The visualizations provided were generated using [Netron](https://github.com/lutzroeder/netron).


## Understanding The Basics

ONNX GraphSurgeon is composed of three major components: Importers, the IR, and Exporters.

### Importers

Importers are used to import a graph into the ONNX GraphSurgeon IR.
The importer interface is defined in [base_importer.py](./onnx_graphsurgeon/importers/base_importer.py).

ONNX GraphSurgeon also provides [high-level importer APIs](./onnx_graphsurgeon/api/api.py) for ease of use:
```python
graph = gs.import_onnx(onnx.load("model.onnx"))
```

### IR

The Intermediate Representation (IR) is where all modifications to the graph are made. It can also be used to
create new graphs from scratch. The IR involves three components: [Tensor](./onnx_graphsurgeon/ir/tensor.py)s,
[Node](./onnx_graphsurgeon/ir/node.py)s, and [Graph](./onnx_graphsurgeon/ir/graph.py)s.

Nearly all of the member variables of each component can be freely modified. For details on the various
attributes of these classes, you can view the help output using `help(<class_or_instance>)` in an
interactive shell, or using `print(help(<class_or_instance>))` in a script, where `<class_or_instance>`
is an ONNX GraphSurgeon type, or an instance of that type.

#### Tensor

Tensors are divided into two subclasses: `Variable` and `Constant`.

- A `Constant` is a tensor whose values are known upfront, and can be retrieved as a NumPy array and modified.
    *Note: The `values` property of a `Constant` is loaded on-demand. If the property is not accessed, the values will*
    *not be loaded as a NumPy array*.
- A `Variable` is a tensor whose values are unknown until inference-time, but may contain information about data type and shape.

The inputs and outputs of Tensors are always Nodes.

**An example constant tensor from ResNet50:**
```
>>> print(tensor)
Constant (gpu_0/res_conv1_bn_s_0)
[0.85369843 1.1515082  0.9152944  0.9577646  1.0663182  0.55629414
 1.2009839  1.1912311  2.2619808  0.62263143 1.1149117  1.4921428
 0.89566356 1.0358194  1.431092   1.5360111  1.25086    0.8706703
 1.2564877  0.8524589  0.9436758  0.7507614  0.8945271  0.93587166
 1.8422242  3.0609846  1.3124607  1.2158023  1.3937513  0.7857263
 0.8928106  1.3042281  1.0153942  0.89356416 1.0052011  1.2964457
 1.1117343  1.0669073  0.91343874 0.92906713 1.0465593  1.1261675
 1.4551278  1.8252873  1.9678202  1.1031747  2.3236883  0.8831993
 1.1133649  1.1654979  1.2705412  2.5578163  0.9504889  1.0441847
 1.0620039  0.92997414 1.2119316  1.3101407  0.7091761  0.99814713
 1.3404484  0.96389204 1.3435135  0.9236031 ]
```

**An example variable tensor from ResNet50:**
```
>>> print(tensor)
Variable (gpu_0/data_0): (shape=[1, 3, 224, 224], dtype=float32)
```


#### Node

A `Node` defines an operation in the graph. A node may specify attributes; attribute values can be any
Python primitive types, as well as ONNX GraphSurgeon `Graph`s or `Tensor`s

The inputs and outputs of Nodes are always Tensors

**An example ReLU node from ResNet50:**
```
>>> print(node)
 (Relu)
    Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

In this case, the node has no attributes. Otherwise, attributes are displayed as an `OrderedDict`.


#### A Note On Modifying Inputs And Outputs

The `inputs`/`outputs` members of nodes and tensors have special logic that will update the inputs/outputs of all
affected nodes/tensors when you make a change. This means, for example, that you do **not** need to update the `inputs`
of a Node when you make a change to the `outputs` of its input tensor.

Consider the following node:
```
>>> print(node)
 (Relu).
    Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

The input tensor can be accessed like so:
```
>>> tensor = node.inputs[0]
>>> print(tensor)
Tensor (gpu_0/res_conv1_bn_1)
>>> print(tensor.outputs)
[ (Relu).
	Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
	Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```

If we remove the node from the outputs of the tensor, this is reflected in the node inputs as well:
```
>>> del tensor.outputs[0]
>>> print(tensor.outputs)
[]
>>> print(node)
 (Relu).
    Inputs: []
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
```


#### Graph

A `Graph` contains zero or more `Node`s and input/output `Tensor`s.

Intermediate tensors are not explicitly tracked, but are instead retrieved from the nodes contained within the graph.

The `Graph` class exposes several functions. A small subset is listed here:

- `cleanup()`: Removes unused nodes and tensors in the graph
- `toposort()`: Topologically sorts the graph.
- `tensors()`: Returns a `Dict[str, Tensor]` mapping tensor names to tensors, by walking over all the tensors in the graph.
    This is an `O(N)` operation, and so may be slow for large graphs.

To see the full Graph API, you can see `help(onnx_graphsurgeon.Graph)` in an interactive Python shell.

### Exporters

Exporters are used to export the ONNX GraphSurgeon IR to ONNX or other types of graphs.
The exporter interface is defined in [base_exporter.py](./onnx_graphsurgeon/exporters/base_exporter.py).

ONNX GraphSurgeon also provides [high-level exporter APIs](./onnx_graphsurgeon/api/api.py) for ease of use:
```python
onnx.save(gs.export_onnx(graph), "model.onnx")
```


## Advanced

### Working With Models With External Data

Using models with externally stored data with ONNX-GraphSurgeon is almost the same as working with
ONNX models without external data. Refer to the
[official ONNX documentation](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#loading-an-onnx-model-with-external-data)
for details on how to load such models. To import the model into ONNX-GraphSurgeon, you can use the
`import_onnx` function as normal.

During export, you just need to take one additional step:

1. Export the model from ONNX-GraphSurgeon as normal:
    ```python
    model = gs.export_onnx(graph)
    ```

2. Update the model so that it writes its data to the external location. If the location is not
    specified, it defaults to the same directory as the ONNX model:
    ```python
    from onnx.external_data_helper import convert_model_to_external_data

    convert_model_to_external_data(model, location="model.data")
    ```

3. Then you can save the model as usual:
    ```python
    onnx.save(model, "model.onnx")
    ```
