# Using Dynamic Shapes With TensorRT

## Introduction

*NOTE: This example is intended for use with TensorRT 8.0 or newer.*
    *Older versions may require slight modifications to the example code.*

In order to use dynamic input shapes with TensorRT, we have to specify a range
(or multiple ranges) of possible shapes when we build the engine.
TensorRT optimization profiles provide the means of doing so.

Using the TensorRT API, the process involves two steps:

1. During engine building, specify one or more optimization profiles.
    An optimization profile includes 3 shapes for each input:
    - `min`: The minimum shape for which the profile should work.
    - `opt`: The shape which TensorRT should optimize for.
        Generally, you'd want this to correspond to the most commonly used shape.
    - `max`: The maximum shape for which the profile should work.

2. During inference, set the input shape(s) in the execution context, then
    query the execution context (*not* the engine) to determine the shape(s) of the output(s).
    Based on the output shape(s), the device buffers can be resized to accomodate
    the entire output(s).

    For a single-input, single-output model, this would look roughly as follows:

    <!-- Polygraphy Test: Ignore Start -->
    ```python
    context.set_binding_shape(0, inp.shape)

    out_shape = context.get_binding_shape(1)
    out_buf.resize(out_shape)

    # Rest of inference code...
    ```
    <!-- Polygraphy Test: Ignore End -->

Polygraphy can simplify both steps and help you avoid common pitfalls:

1. It provides a `Profile` abstraction, which is an `OrderedDict` that
    can be converted to a TensorRT `IOptimizationProfile` and includes some utility functions:
    - `fill_defaults`: Fills the profile with default shapes based on the network.
    - `to_trt`: Creates a TensorRT `IOptimizationProfile` using the shapes in this `Profile`.

    What's more, `Profile` will automatically handle complexities like the
    distinction between shape-tensor vs. non-shape-tensor inputs - you do not
    need to worry about this distinction yourself.

2. The `TrtRunner` will automatically handle dynamic shapes in the model.
    As in `Profile`, distinctions between shape-tensor and non-shape-tensor inputs
    are handled automatically.

    Additionally, the runner will only update the context binding shapes when required,
    as changing the shapes has a small overhead. The output device buffers will only
    be resized if their current size is smaller that the context outputs, thus avoiding
    unnecessary reallocation.


### Setting The Stage

For the sake of this example, we'll imagine a hypothetical scenario:

We're running an inference workload using an image classification model.

Normally, we use this model in an online scenario - i.e. we want the lowest possible
latency, so we'll process one image at a time.
For this case, assume `batch_size` is `[1]`.

However, if we have too many users, then we need to employ dynamic batching so that
our throughput doesn't suffer. Our range of batch sizes is still small to
keep the latency acceptable. Our most frequently used batch size is 4.
For this case, assume `batch_size` is in the range `[1, 32]`.

In even rarer cases, we need to process large amounts of data offline. In this case,
we use a very large batch size to improve our throughput.
For this case, assume `batch_size` is `[128]`.

### Performance Considerations

In implementing our inference pipeline, we need to consider a few tradeoffs:

- A profile with a large range will not perform as well as for the entire range as
    multiple profiles each with smaller ranges.
- Switching shapes within a profile has a small but non-zero cost.
- Switching profiles within a context has a larger cost than switching shapes within a profile.
    - We can avoid the cost of switching profiles by creating a separate execution context
        for each profile and selecting the appropriate context at runtime.
        However, keep in mind that each context will require some additional memory.


### A Possible Solution

Assuming the image size is `(3, 28, 28)`, we'll create three separate
optimization profiles, and a separate context for each:

1. For the low latency case:
    `min=(1, 3, 28, 28), opt=(1, 3, 28, 28), max=(1, 3, 28, 28)`

2. For the dynamic batching case:
    `min=(1, 3, 28, 28), opt=(4, 3, 28, 28), max=(32, 3, 28, 28)`

    Note that we use a batch size of `4` for `opt` since that's the most common case.

3. For the offline case:
    `min=(128, 3, 28, 28), opt=(128, 3, 28, 28), max=(128, 3, 28, 28)`

For each context, we'll create a corresponding `TrtRunner`. If we make sure that
we own the engine and the context (by not providing them via lazy loaders), then
the cost of activating a runner should be small - it just needs to allocate
input and output buffers. Hence, we'll be able to activate runners on-demand quickly.


## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`

2. Run the example:

    ```bash
    python3 example.py
    ```

3. **[Optional]** Inspect the generated engine:

    ```bash
    polygraphy inspect model dynamic_identity.engine
    ```

## Further Reading

For more information on using dynamic shapes with TensorRT, see the
[developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)
