# Debugging TensorRT Accuracy Issues

Accuracy issues in TensorRT, especially with large networks, can be challenging to debug.
One way to make them manageable is to reduce the problem size or pinpoint the source of failure.

This guide aims to provide a general approach to doing so; it is structured as a flattened flowchart -
at each branch, two links are provided so you can choose the one that best matches your situation.

If you're using an ONNX model, try [sanitizing it](../examples/cli/surgeon/02_folding_constants/) before
proceeding, as this may solve the problem in some cases.


## Does Real Input Data Make A Difference?

Some models may be sensitive to input data. For example, real inputs may result in better accuracy
than randomly generated ones. Polygraphy offers multiple ways to supply real input
data, outlined in [`run` example 05](../examples/cli/run/05_comparing_with_custom_input_data/).

Does using real input data improve the accuracy?

- Yes, accuracy is acceptable when using real input data.

    This likely means there is no bug; rather, your model is sensitive to input data.

- No, I still see accuracy issues even with real input data.

    Go To: [Intermittent Or Not?](#intermittent-or-not)


## Intermittent Or Not?

Is the issue intermittent between engine builds?

- Yes, sometimes the accuracy issue disappears when I rebuild the engine.

    Go To: [Debugging Intermittent Accuracy Issues](#debugging-intermittent-accuracy-issues)

- No, I see accuracy issues every time I build an engine.

    Go To: [Is Layerwise An Option?](#is-layerwise-an-option)


## Debugging Intermittent Accuracy Issues

Since the engine building process is non-deterministic, different tactics (i.e. layer implementations) may
be selected each time the engine is built. When one of the tactics is faulty, this may manifest as an intermittent
failure. Polygraphy includes a `debug build` subtool to help you find such tactics.

For more information, refer to [`debug` example 01](../examples/cli/debug/01_debugging_flaky_trt_tactics/).

Were you able to find the failing tactic?

- Yes, I know which tactic is faulty.

    Go To: [You Have A Minimal Failing Case!](#you-have-a-minimal-failing-case)

- No, the failure may not be intermittent.

    Go To: [Is Layerwise An Option?](#is-layerwise-an-option)



## Is Layerwise An Option?

If the accuracy issue is consistently reproducible, the best next step is to figure out which
layer is causing the failure. Polygraphy includes a mechanism to mark all tensors in the network
as outputs so that they can be compared; however, this can potentially affect TensorRT's optimization
process. Hence, we need to determine if we still observe the accuracy issue when all output tensors are marked.

Refer to [this example](../examples/cli/run/01_comparing_frameworks/README.md#comparing-per-layer-outputs-between-onnx-runtime-and-tensorrt) for details on how to compare
per-layer outputs before proceeding.

Were you able to reproduce the accuracy failure when comparing layer-wise outputs?

- Yes, the failure reprodces even if I mark other outputs in the network.

    Go To: [Extracting A Failing Subgraph](#extracting-a-failing-subgraph)

- No, marking other outputs causes the accuracy to improve OR I am not able to run the model at all when I mark other outputs.

    Go To: [Reducing A Failing Onnx Model](#reducing-a-failing-onnx-model)


## Extracting A Failing Subgraph

Since we're able to compare layerwise outputs, we should be able to determine which layer
first introduces the error by looking at the output comparison logs. Once we know which layer
is problematic, we can extract it from the model.

In order to figure out the input and output tensors for the layer in question, we can use
`polygraphy inspect model`. Refer to one of these examples for details:

- [TensorRT Networks](../examples/cli/inspect/01_inspecting_a_tensorrt_network/)
- [ONNX models](../examples/cli/inspect/03_inspecting_an_onnx_model/).

Next, we can extract a subgraph including just the problematic layer.
For more information, refer to [`surgeon` example 01](../examples/cli/surgeon/01_isolating_subgraphs/).

Does this isolated subgraph reproduce the problem?

- Yes, the subgraph fails too.

    Go To: [You Have A Minimal Failing Case!](#you-have-a-minimal-failing-case)

- No, the subgraph works fine.

    Go To: [Reducing A Failing Onnx Model](#reducing-a-failing-onnx-model)


## Reducing A Failing ONNX Model

When we're unable to pinpoint the source of failure using a layerwise comparison, we can
use a brute force method of reducing the ONNX model - iteratively generate smaller and smaller
subgraphs to find the smallest possible one that still fails. The `debug reduce` tools helps automate this process.

For more information, refer to [`debug` example 02](../examples/cli/debug/02_reducing_failing_onnx_models/).

Does the reduced model fail?

- Yes, the reduced model fails.

    Go To: [You Have A Minimal Failing Case!](#you-have-a-minimal-failing-case)

- No, the reduced model doesn't fail, or fails in a different way.

    Go To: [Double Check Your Reduce Options](#double-check-your-reduce-options)


## Double Check Your Reduce Options

If the reduced model no longer fails, or fails in a different way, ensure that your `--check` command
is correct. You may also want to use `--fail-regex` to ensure that you're only considering the accuracy
failure (and not other, unrelated failures) when reducing the model.

- Try reducing again.

    Go To: [Reducing A Failing Onnx Model](#reducing-a-failing-onnx-model)

## You Have A Minimal Failing Case!

If you've made it to this point, you now have a minimal failing case! Further debugging should
be significantly easier.

If you are a TensorRT developer, you'll need to dive into the code at this point.
If not, please report your bug!
