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
python3 -m pip install onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
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

ONNX GraphSurgeon also provides [high-level importer APIs](./onnx_graphsurgeon/__init__.py) for ease of use:
```python
import onnx_graphsurgeon as gs
graph = gs.import_onnx(onnx.load("model.onnx"))
```

The `model.onnx` file used in the following examples is [resnet50-v2-7.onnx](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx).

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
>>> tensor = graph.tensors()['resnetv24_batchnorm1_gamma']
>>> print(tensor)
Constant (resnetv24_batchnorm1_gamma): (shape=(64,), dtype=float32)
>>> print(tensor.values)
[-9.83304853e-05  2.17902549e-02  8.74318480e-02  3.26565914e-02
  8.23329296e-03  5.41263507e-05 -9.77454185e-02  2.29816716e-02
  7.52642791e-06  6.15647587e-05  3.15795541e-02  2.24303094e-05
 -9.81112898e-05 -3.51903400e-05  2.34726686e-02 -3.48845992e-06
  1.07414778e-02 -2.35510282e-02 -6.32902316e-04  2.36321557e-02
 -2.30935775e-02  9.88963172e-02  2.41898187e-02  2.11347304e-02
  2.35060174e-02  5.13273444e-05  2.67624146e-05  2.45444812e-02
  6.36491532e-05  4.07683291e-02 -4.90635410e-02  1.20312367e-02
  2.06732173e-02 -1.19354352e-01 -5.92932338e-05 -4.35315929e-02
  3.90425622e-02  6.16753958e-02  1.35400733e-02  2.10027705e-04
 -2.40152876e-05  2.48841383e-02  1.38983105e-05  2.23469138e-02
 -3.32205333e-02  2.01729666e-02  2.43023913e-02  2.44748250e-01
  2.33223271e-02  5.36156949e-05  4.04572971e-02  1.73668638e-02
 -3.28809301e-06  2.53515430e-02 -3.43644933e-06  2.19323078e-06
  1.24725382e-04  1.08645864e-01 -3.93772598e-06  1.88900251e-02
  2.35187691e-02  1.16659294e-05 -7.32624685e-05  2.96757370e-02]
```

**An example variable tensor from ResNet50:**
```
>>> tensor = graph.inputs[0]
>>> print(tensor)
Variable (data): (shape=['N', 3, 224, 224], dtype=float32)
```


#### Node

A `Node` defines an operation in the graph. A node may specify attributes; attribute values can be any
Python primitive types, as well as ONNX GraphSurgeon `Graph`s or `Tensor`s

The inputs and outputs of Nodes are always Tensors

**An example ReLU node from ResNet50:**
```
>>> node = next(node for node in graph.nodes if node.name == "resnetv24_relu0_fwd")
>>> print(node)
resnetv24_relu0_fwd (Relu)
	Inputs: [
		Variable (resnetv24_batchnorm1_fwd): (shape=None, dtype=None)
	]
	Outputs: [
		Variable (resnetv24_relu0_fwd): (shape=None, dtype=None)
	]
```

In this case, the node has no attributes. Otherwise, attributes are displayed as an `OrderedDict`.


#### A Note On Modifying Inputs And Outputs

The `inputs`/`outputs` members of nodes and tensors have special logic that will update the inputs/outputs of all
affected nodes/tensors when you make a change. This means, for example, that you do **not** need to update the `inputs`
of a Node when you make a change to the `outputs` of its input tensor.

Consider the following node:
```
>>> print(node)
resnetv24_relu0_fwd (Relu)
	Inputs: [
		Variable (resnetv24_batchnorm1_fwd): (shape=None, dtype=None)
	]
	Outputs: [
		Variable (resnetv24_relu0_fwd): (shape=None, dtype=None)
	]
```

The input tensor can be accessed like so:
```
>>> tensor = node.inputs[0]
>>> print(tensor)
Variable (resnetv24_batchnorm1_fwd): (shape=None, dtype=None)
>>> print(tensor.outputs)
[resnetv24_relu0_fwd (Relu)
	Inputs: [
		Variable (resnetv24_batchnorm1_fwd): (shape=None, dtype=None)
	]
	Outputs: [
		Variable (resnetv24_relu0_fwd): (shape=None, dtype=None)
	]]
```

If we remove the node from the outputs of the tensor, this is reflected in the node inputs as well:
```
>>> del tensor.outputs[0]
>>> print(tensor.outputs)
[]
>>> print(node)
resnetv24_relu0_fwd (Relu)
	Inputs: [
	]
	Outputs: [
		Variable (resnetv24_relu0_fwd): (shape=None, dtype=None)
	]
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

ONNX GraphSurgeon also provides [high-level exporter APIs](./onnx_graphsurgeon/__init__.py) for ease of use:
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
