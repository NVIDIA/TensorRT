# Using `debug reduce` Effectively


## Table Of Contents

- [Introduction](#introduction)
- [A Note On Models With Dynamic Input Shapes](#a-note-on-models-with-dynamic-input-shapes)
- [Debugging Accuracy Errors](#debugging-accuracy-errors)
    - [Tolerances](#tolerances)
    - [Generating Golden Values](#generating-golden-values)
- [Tips And Tricks](#tips-and-tricks)
    - [Saving Intermediate Models](#saving-intermediate-models)
    - [Insights From Minimum Good Models](#insights-from-minimum-good-models)
    - [Reduction Modes](#reduction-modes)
- [Further Reading](#further-reading)


## Introduction

The `debug reduce` subtool allows you to iteratively reduce a failing ONNX model
to find a minimal failing case, which might be easier to debug than the original model.
The fundamental steps undertaken by `debug reduce` are as follows:

1. Remove some nodes from the original graph and write a new model to `polygraphy_debug.onnx`
    (this path can be changed using the `--iter-artifact` option).

2. Evaluate the model either interactively or, if a `--check` command is provided, automatically.

3. Remove more nodes if the model still fails, otherwise add nodes back; Then, repeat the process.

This guide provides some general information as well as tips and tricks on
how to use `debug reduce` effectively.

Also see the [general how-to guide for `debug` subtools](./use_debug_subtools_effectively.md),
which includes information applicable to all the `debug` subtools.


## A Note On Models With Dynamic Input Shapes

For models with dynamic input shapes, you may not always know the shapes of all intermediate
tensors in the model. Thus, when you check subgraphs, you may end up using
incorrect tensor shapes.

There are two ways to get around this:

1. Use `polygraphy surgeon sanitize --override-input-shapes <shapes>` to freeze the input shapes in the model
2. Supply `--model-input-shapes` to `debug reduce`, which will use shape inference to infer shapes
    of intermediate tensors.

If your model uses shape operations, it is generally best to use option (1) and fold the shape
operations away with `--fold-constants`.

In either case, if there's a problem with shape inference, you can use
`--force-fallback-shape-inference` to infer shapes by running inference instead.

Alternatively, you can use `--no-reduce-inputs` so that the model inputs are not modified.
The `polygraphy_debug.onnx` subgraph generated during each iteration will always use the inputs
of the original model; only layers from the end will be removed.


## Debugging Accuracy Errors

Accuracy errors are especially complex to debug since errors introduced by early layers
in the graph might be amplified by subsequent layers, making it difficult to determine
which layer is the true root cause of the error. This section outlines some things to
keep in mind when using `debug reduce` to debug accuracy errors.

### Tolerances

In some model architectures, intermediate layers may have large errors without necessarily
causing accuracy issues in the final model output. Thus, make sure the tolerance you use for
comparison is high enough to ignore these kinds of false positives.

At the same time, tolerance must be low enough to catch real errors.

A good starting point is to set tolerances close to the error you observe in the full model.


### Generating Golden Values

There are two different approaches you can take when generating golden values for comparison,
each with their own advantages and disadvantages:

1. **Generating golden values for all layers ahead of time.**

    When generating golden values ahead of time, you need to make sure that the input values
    to each subgraph come from the golden values. Otherwise, comparing the outputs of
    the subgraph against the golden values will be meaningless.
    See [the example](../examples/cli/debug/02_reducing_failing_onnx_models/) for
    details on this approach.

2. **Generating golden values for each subgraph.**

    Regenerating golden values for each subgraph may require less manual effort, but
    has the disadvantage that it doesn't necessarily accurately replicate the behavior
    of the subgraph in the context of the larger graph.
    For example, if the error in your model was caused by an overflow in an intermediate
    layer of the original model, generating fresh input values for each subgraph may not
    reproduce it.


## Tips And Tricks


### Saving Intermediate Models

In some cases, it's useful to have access to every model generated during the reduction process.
This way, if reduction exits early or fails to generate a minimal model, you still have something
to work with. Additionally, you can manually compare the various passing and failing subgraphs
to identify patterns, which may help you determine the root cause of the error.

You can specify `--artifacts polygraphy_debug.onnx` to `debug reduce` to automatically sort models
from each iteration into `good` and `bad` directories. The file name will include the iteration
number so you can easily correlate it with the logging output during reduction.


### Insights From Minimum Good Models

In addition to minimum failing models, `debug reduce` can also generate minimum passing models.
Generally, this is the passing model that is closest in size to the minimal failing model.
Comparing this against the minimum failing model can yield additional insights on the root
cause of a failure.

To make `debug reduce` save minimum passing models, use the `--min-good <path>` option.


### Reduction Modes

`debug reduce` offers multiple strategies to reduce the model, which you can specify with the `--mode` option:
`bisect` operates in `O(log(N))` time, while `linear` operates in `O(N)` time but may lead to smaller models.
A good compromise is to use `bisect` on the original model, then further reduce the result using `linear`.


## Further Reading

- The [how-to guide for `debug` subtools](./use_debug_subtools_effectively.md),
    which includes information applicable to all the `debug` subtools.

- The [`debug reduce` example](../examples/cli/debug/02_reducing_failing_onnx_models/), which
    demonstrates some of the features outlined here.
