# Polygraphy Python API

## Table of Contents

- [Introduction](#introduction)
- [Backends](#backends)
    - [Loaders](#loaders)
    - [Runners](#runners)
        - [Writing A Custom Runner](#writing-a-custom-runner)
- [Comparator](#comparator)
    - [Data Loaders](#data-loaders)
- [Putting It All Together](#putting-it-all-together)
- [Examples](#examples)
- [Enabling PyTorch](#enabling-pytorch)


## Introduction

**IMPORTANT:** The Python API is still not completely stable, and minor but breaking changes
may be made in future versions.

The Polygraphy API consists broadly of two major components:
[`Backend`s](#backends) and the [`Comparator`](#comparator).

**NOTE:** To help you get started with the API, you can use the [`run`](./tools/run/) tool to auto-generate
template scripts that use the Polygraphy API.


## Backends

A Polygraphy backend provides an interface for a deep learning framework.
Backends are comprised of two components: Loaders and Runners.


### Loaders

A `Loader` is used to load models for runners (see [`BaseLoadModel`](./backend/base/runner.py)).

A `Loader` can be any `Callable` that takes no arguments and returns a model.
"Model" here is a generic term, and the specifics depend on the runner for which the loader has been implemented.

Moreover, existing `Loader`s can be composed for more advanced behaviors.
For example, we can implement a conversion like `TensorFlow Frozen Model -> ONNX -> TensorRT Network -> TensorRT Engine`:
```python
from polygraphy.backend.tf import GraphFromFrozen
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxBytes
from polygraphy.backend.onnx import OnnxFromTfGraph, BytesFromOnnx

build_engine = EngineFromNetwork(NetworkFromOnnxBytes(
                BytesFromOnnx(OnnxFromTfGraph(GraphFromFrozen("/path/to/model.pb")))))
```
We can now provide `build_engine` directly to a `TrtRunner`.


### Runners

A `Runner` uses a loader to load a model and can then run inference (see [`BaseRunner`](./backend/base/runner.py)).

**IMPORTANT:** Runners may reuse their output buffers. Thus, if you need to save outputs from multiple inferences, you should
make a copy of the outputs with `copy.deepcopy(outputs)`.

To use a runner, you just need to activate it, and then call `infer()`.
Note that activating a runner can be very expensive, so you should minimize the
number of times you activate a runner - ideally do not do this more than once.

It is recommended to use a context manager to activate and deactivate the
runner rather than calling the functions manually:
```python
from polygraphy.backend.trt import TrtRunner

with TrtRunner(build_engine) as runner:
    outputs = runner.infer(feed_dict={"input0": input_data})
```

#### Writing A Custom Runner

Generally, you do not need to write custom runners unless you want to support a new backend.

In case you do, in the simplest case, you only need to implement two functions:
- `infer`: Accepts a dictionary of numpy buffers, runs inference, and finally returns a dictionary containing the outputs.
- `get_input_metadata`: Returns a [`TensorMetadata`](./common/struct.py) mapping input names to their shapes and data types.
    You may use `None` to indicate dynamic dimensions.

For more advanced runners, where some setup is required, you may also need to implement the `activate_impl()` and `deactivate_impl()` functions.

For example, in the `TrtRunner`, engines are created in `activate_impl()` and destroyed in `deactivate_impl()`.
Importantly, the GPU is *not used at all* until these functions have been called (notice, for example,
    that in the `TrtRunner`, the CUDA runtime library is only loaded in the `activate_impl()` function).
This allows the `Comparator` to optionally provide each runner with exclusive access to the GPU, to prevent any interference between runners.


## Comparator

The `Comparator` is used to run inference for runners, and then compare accuracy (see [Comparator.py](./comparator/Comparator.py)).
This process is divided into two phases:

1. Running inference:
    ```python
    run_results = Comparator.run(runners)
    ```
    This function accepts a list of runners, and returns a `RunResults` object (see [Comparator.py](./comparator/Comparator.py))
    containing information about the outputs of each run.
    It also accepts an optional `data_loader` argument to control the input data. If not provided, it will use the
    default data loader. `Comparator.run()` continues until inputs from the data loader are exhausted.

2. Comparing results:
    ```python
    Comparator.compare_accuracy(run_results)
    ```
    This function accepts the results returned by `Comparator.run` and compares them between runners.


### Data Loaders

A data loader is used by the `Comparator` to load input data to feed to each runner
(see [DataLoader.py](./comparator/DataLoader.py)).

A data loader can be any generator or iterable that yields
a dictionary of input buffers. In the simplest case, this can just be a `list` of `dict`s.

In case you don't know details about the inputs ahead of time, you can access the `input_metadata`
property in your data loader, which will be set to an `TensorMetadata` instance by the Comparator.

**NOTE:** Polygraphy provides a default `DataLoader` class that uses numpy to generate random input buffers.
The input data can be bounded via parameters to the constructor.

**IMPORTANT:** Data loaders are designed for scenarios where you need to compare a small number
of inputs across multiple runners. It is **not** a good idea to use a custom data loader
to validate a model with an entire dataset! Instead, runners should be used directly for such
cases (see the [example](../examples/api/02_using_real_data)).


## Putting It All Together

Now that you know the basic components of Polygraphy, let's take a look at how they fit together.

In this example, we will write a script that:
1. Loads a TensorFlow frozen model
2. Converts it to ONNX
3. Builds a TensorRT engine from the ONNX model
4. Bounds input values in the range `[0, 2]`
5. Runs inference using TensorFlow, ONNX-Runtime, and TensorRT
6. Compares the results and checks that they match

```python
from polygraphy.backend.onnx import OnnxFromTfGraph, BytesFromOnnx
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnxBytes
from polygraphy.backend.tf import TfRunner, GraphFromFrozen, SessionFromGraph
from polygraphy.backend.trt import TrtRunner, EngineFromNetwork, NetworkFromOnnxBytes
from polygraphy.comparator import Comparator, DataLoader

# Convert the model into the various formats we care about.
load_frozen = GraphFromFrozen("/path/to/frozen/model.pb")
build_tf_session = SessionFromGraph(load_frozen)
export_serialized_onnx = BytesFromOnnx(OnnxFromTfGraph(load_frozen))
build_onnxrt_session = SessionFromOnnxBytes(export_serialized_onnx)
build_engine = EngineFromNetwork(NetworkFromOnnxBytes(export_serialized_onnx))

# We want to run the model with TensorFlow, ONNX Runtime, and TensorRT.
runners = [
    TfRunner(build_tf_session),
    OnnxrtRunner(build_onnxrt_session),
    TrtRunner(build_engine),
]

# For this model, assume inputs need to be bounded.
data_loader = DataLoader(int_range=(0, 2), float_range=(0.0, 2.0))

# Finally, run and check accuracy.
run_results = Comparator.run(runners, data_loader=data_loader)
assert bool(Comparator.compare_accuracy(run_results))
```


## Examples

You can find complete code examples that use the Polygraphy Python API [here](../examples/api).


## Enabling PyTorch

In order to enable PyTorch, you need to provide three things to the `PytRunner`:
1. A `BaseLoadPyt`: In the simplest case, this can be a callable that returns a `torch.nn.Module`.

2. `input_metadata`: A `TensorMetadata` describing the inputs of the model. This maps input names to their shapes and data types. As with other runners, `None` may be used to indicate dynamic dimensions.

    **NOTE:** Other runners are able to automatically determine input metadata by inspecting the model definition, but because of the way PyTorch is implemented, it is difficult to write a generic function to determine model inputs from a `torch.nn.Module`.

3. `output_names`: A list of output names. This is used by the `Comparator` to match `PytRunner` outputs to those of other runners.
