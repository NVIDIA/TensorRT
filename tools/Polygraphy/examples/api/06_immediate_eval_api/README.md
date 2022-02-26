# Immediately Evaluated Functional API

## Introduction

<!-- Polygraphy Test: Ignore Start -->
Most of the time, the lazy loaders included with Polygraphy have several advantages:

- They allow us to defer the work until we actually need to do it, which can potentially save
    time.
- Since constructed loaders are extremely light-weight, runners using lazily evaluated loaders can be
    easily copied into other processes or threads, where they can then be launched.
    If runners instead referenced entire models/inference sessions, it would be non-trivial to copy them in this way.
- They allow us to define a sequence of operations in advance by chaining loaders together, which
    provides an easy way to build reusable functions.
    For example, we could create a loader that imports a model from ONNX and generates a serialized TensorRT Engine:

    ```python
    build_engine = EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))
    ```

- They allow for special semantics where if a callable is provided to a loader, it takes ownership
    of the return value, whereas otherwise it does not. These special semantics are useful for
    sharing objects between multiple loaders.

However, this can sometimes lead to code that is less readable, or even downright confusing.
For example, consider the following:
```python
# Each line in this example looks almost the same, but has significantly
# different behavior. Some of these lines even cause memory leaks!
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")) # This is a loader instance, not an engine!
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))() # This is an engine.
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")()) # And it's a loader instance again...
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx")())() # Back to an engine!
EngineBytesFromNetwork(NetworkFromOnnxPath("/path/to/model.onnx"))()() # This throws - can you see why?
```

For that reason, Polygraphy provides immediately-evaluated functional
equivalents of each loader. Each functional variant uses the same name as the loader, but
`snake_case` instead of `PascalCase`. Using the functional variants, loader code like:

```python
parse_network = NetworkFromOnnxPath("/path/to/model.onnx")
create_config = CreateConfig(fp16=True, tf32=True)
build_engine = EngineFromNetwork(parse_network, create_config)
engine = build_engine()
```

becomes:

```python
builder, network, parser = network_from_onnx_path("/path/to/model.onnx")
config = create_config(builder, network, fp16=True, tf32=True)
engine = engine_from_network((builder, network, parser), config)
```
<!-- Polygraphy Test: Ignore End -->


In this example, we'll look at how you can leverage the functional API to convert an ONNX
model to a TensorRT network, modify the network, build a TensorRT engine with FP16 precision
enabled, and run inference.
We'll also save the engine to a file to see how you can load it again and run inference.


## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`

2. **[Optional]** Inspect the model before running the example:

    ```bash
    polygraphy inspect model identity.onnx
    ```

3. Run the script that builds and runs the engine:

    ```bash
    python3 build_and_run.py
    ```

4. **[Optional]** Inspect the TensorRT engine built by the example:

    ```bash
    polygraphy inspect model identity.engine
    ```

5. Run the script that loads the previously built engine, then runs it:

    ```bash
    python3 load_and_run.py
    ```
