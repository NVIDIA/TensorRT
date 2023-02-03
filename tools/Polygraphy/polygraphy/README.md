# Polygraphy Python API

## Table of Contents

- [Introduction](#introduction)
- [Backends](#backends)
    - [Loaders](#loaders)
    - [Runners](#runners)
        - [Writing A Custom Runner](#writing-a-custom-runner)
- [Comparator](#comparator)
    - [Data Loaders](#data-loaders)
- [Logger](#logger)
- [Putting It All Together](#putting-it-all-together)
- [Enabling PyTorch](#enabling-pytorch)
- [Examples](#examples)
- [Python API Reference Documentation](#python-api-reference-documentation)
    - [Building Python API Documentation Locally](#building-python-api-documentation-locally)
- [Deprecation Policy](#deprecation-policy)

## Introduction

The Polygraphy API consists broadly of two major components:
[`Backend`s](#backends) and the [`Comparator`](#comparator).

**NOTE:** To help you get started with the API, you can use the [`run`](./tools/run/) tool
with the `--gen-script` option to auto-generate template scripts that use the Polygraphy API.

> :warning: Any APIs not documented in the [API reference documentation](#python-api-reference-documentation)
    should be considered internal only and do not adhere to the [deprecation policy](#deprecation-policy)
    as the public APIs do. Thus, they may be modified or removed at any time without warning.
    Avoid using these internal APIs outside of Polygraphy!


## Backends

A Polygraphy backend provides an interface for a deep learning framework.
Backends are comprised of two components: Loaders and Runners.


### Loaders

A `Loader` is a functor or callable that loads or operates on models in some way.

Existing `Loader`s can be composed for more advanced behaviors.
For example, we can implement a conversion like `ONNX -> TensorRT Network -> TensorRT Engine`:

```python
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath

build_engine = EngineFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))
```

`build_engine` is a callable that will build a TensorRT engine.

Polygraphy also provides immediately evaluated functional variants of each Loader.
These use the same names, except `snake_case` instead of `PascalCase`, and expose the same APIs.
Using the functional loaders, the conversion above would be:

```python
from polygraphy.backend.trt import engine_from_network, network_from_onnx_path

engine = engine_from_network(network_from_onnx_path("/path/to/model.onnx"))
```

`engine` is a TensorRT engine as opposed to a callable.


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
- `infer_impl`: Accepts a dictionary of numpy buffers, runs inference, and finally returns a dictionary containing the outputs.
- `get_input_metadata_impl`: Returns a [`TensorMetadata`](./common/struct.py) mapping input names to their shapes and data types.
    You may use `None`, negative numbers, or strings to indicate dynamic dimensions.

For more advanced runners, where some setup is required, you may also need to implement the `activate_impl()` and `deactivate_impl()` functions.

For example, in the `TrtRunner`, engines are created in `activate_impl()` and destroyed in `deactivate_impl()`.
Importantly, the GPU is *not used at all* until these functions have been called (notice, for example,
    that in the `TrtRunner`, the CUDA runtime library is only loaded in the `activate_impl()` function).
This allows the `Comparator` to optionally provide each runner with exclusive access to the GPU, to prevent any interference between runners.


## Comparator

The `Comparator` is used to run inference for runners, and then compare accuracy (see [Comparator.py](./comparator/comparator.py)).
This process is divided into two phases:

1. Running inference:

    ```python
    run_results = Comparator.run(runners)
    ```

    This function accepts a list of runners and returns a `RunResults` object (see [Comparator.py](./comparator/comparator.py))
    containing the inference outputs of each run.
    It also accepts an optional `data_loader` argument to control the input data. If not provided, it will use the
    default data loader. `Comparator.run()` continues until inputs from the data loader are exhausted.

2. Comparing results:

    ```python
    Comparator.compare_accuracy(run_results)
    ```

    This function accepts the results returned by `Comparator.run` and compares them between runners.

**IMPORTANT:** The Comparator is designed for scenarios where you need to compare a small number
of inputs across multiple runners. It is **not** a good idea to use it
to validate a model with an entire dataset! Instead, runners should be used directly for such
cases (see the [example](../examples/api/02_validating_on_a_dataset)).


### Data Loaders

A data loader is used by the `Comparator` to load input data to feed to each runner
(for example, see Polygraphy's [default data loader](./comparator/data_loader.py)).

A data loader can be any generator or iterable that yields
a dictionary of input buffers. In the simplest case, this can just be a `list` of `dict`s.

In case you don't know details about the inputs ahead of time, you can access the `input_metadata`
property in your data loader, which will be set to an `TensorMetadata` instance by the Comparator.

**NOTE:** Polygraphy provides a default `DataLoader` class that uses numpy to generate random input buffers.
The input data can be bounded via parameters to the constructor.


## Logger

Polygraphy also includes a global logger which can control the verbosity not only of messages emitted by Polygraphy,
but also of those emitted by underlying frameworks, like TensorRT. For example, the `EXTRA_VERBOSE` verbosity corresponds
to TensorRT's `kVERBOSE` logging mode.

To set the verbosity of the logger, use:
```py
G_LOGGER.module_severity = severity
```

For example:
```py
G_LOGGER.module_severity = G_LOGGER.EXTRA_VERBOSE
```


## Putting It All Together

Now that you know the basic components of Polygraphy, let's take a look at how they fit together.

In this example, we will write a script that:
1. Builds a TensorRT engine from an ONNX model
2. Bounds input values in the range `[0, 2]`
3. Runs inference using ONNX-Runtime and TensorRT
4. Compares the results and checks that they match

```python
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import TrtRunner, EngineFromNetwork, NetworkFromOnnxPath
from polygraphy.comparator import Comparator, DataLoader

model_path = "/path/to/model.onnx"

build_onnxrt_session = SessionFromOnnx(model_path)
build_engine = EngineFromNetwork(NetworkFromOnnxPath(model_path))

runners = [
    OnnxrtRunner(build_onnxrt_session),
    TrtRunner(build_engine),
]

data_loader = DataLoader(val_range=(0, 2))

run_results = Comparator.run(runners, data_loader=data_loader)
assert bool(Comparator.compare_accuracy(run_results))
```

## Enabling PyTorch

In order to enable PyTorch, you need to provide three things to the `PytRunner`:
1. A model loader: In the simplest case, this can be a callable that returns a `torch.nn.Module`.

2. `input_metadata`: A `TensorMetadata` describing the inputs of the model. This maps input names to their shapes and data types. As with other runners, `None` may be used to indicate dynamic dimensions.

    **NOTE:** Other runners are able to automatically determine input metadata by inspecting the model definition, but because of the way PyTorch is implemented, it is difficult to write a generic function to determine model inputs from a `torch.nn.Module`.

3. `output_names`: A list of output names. This is used by the `Comparator` to match `PytRunner` outputs to those of other runners.


## Examples

You can find complete code examples that use the Polygraphy Python API [here](../examples/api).


## Python API Reference Documentation

For more details, see the [Polygraphy Python API reference documentation](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html).

### Building Python API Documentation Locally

To build the API documentation, first install required packages:

```bash
python -m pip install -r docs/requirements.txt
```

and then use the `make` target to build docs:

```bash
make docs
```

The HTML documentation will be generated under `build/docs`
To view the docs, open `build/docs/index.html` in a browser or HTML viewer.
