# Debugging Flaky TensorRT Tactics


## Introduction

Sometimes, a tactic in TensorRT may produce incorrect results, or have
otherwise buggy behavior. Since the TensorRT builder relies on timing
tactics, engine builds are non-deterministic, which can make tactic bugs
manifest as flaky/intermittent failures.

One approach to tackling the problem is to run the builder several times,
saving tactic replay files from each run. Once we have a set of known-good and
known-bad tactics, we can compare them to determine which tactic
is likely to be the source of error.

The `debug build` subtool allows you to automate this process.

For more details on how the `debug` tools work, see the help output:
`polygraphy debug -h` and `polygraphy debug build -h`.


## Running The Example

1. Generate golden outputs from ONNX-Runtime:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-outputs golden.json
    ```

2. Use `debug build` to repeatedly build TensorRT engines and compare results against the golden outputs,
    saving a tactic replay file each time:

    ```bash
    polygraphy debug build identity.onnx --fp16 --save-tactics replay.json \
        --artifacts-dir replays --artifacts replay.json --until=10 \
        --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json
    ```

    Let's break this down:

    - Like other `debug` subtools, `debug build` generates an intermediate artifact each iteration
        (`./polygraphy_debug.engine` by default). This artifact in this case is a TensorRT engine.

        *TIP: `debug build` supports all the TensorRT builder configuration options supported*
            *by other tools, like `convert` or `run`.*

    - In order for `debug build` to determine whether each engine fails or passes,
        we provide a `--check` command. Since we're looking at a (fake) accuracy issue,
        we can use `polygraphy run` to compare the outputs of the engine to our golden values.

        *TIP: Like other `debug` subtools, an interactive mode is also supported, which you can*
            *use simply by omitting the `--check` argument.*

    - Unlike other `debug` subtools, `debug build` has no automatic terminating condition, so we need
        to provide the `--until` option so that the tool knows when to stop. This can either be a number
        of iterations, or `"good"` or `"bad"`. In the latter case, the tool will stop after finding the
        first passing or failing iteration respectively.

    - Since we eventually want to compare the good and bad tactic replays, we specify `--save-tactics`
        to save tactic replay files from each iteration, then use `--artifacts` to tell `debug build`
        to manage them, which involves sorting them into `good` and `bad` subdirectories under the
        main artifacts directory, specified with `--artifacts-dir`.


3. Use `inspect diff-tactics` to determine which tactics could be bad:

    ```bash
    polygraphy inspect diff-tactics --dir replays
    ```

    *NOTE: This last step should report that it could not determine potentially bad tactics since*
        *our `bad` directory should be empty at this point (please file a TensorRT issue otherwise!):*

    <!-- Polygraphy Test: Ignore Start -->
    ```
    [I] Loaded 2 good tactic replays.
    [I] Loaded 0 bad tactic replays.
    [I] Could not determine potentially bad tactics. Try generating more tactic replay files?
    ```
    <!-- Polygraphy Test: Ignore End -->


## Further Reading

For more information on the `debug` tool, as well as tips and tricks applicable
to all `debug` subtools, see the
[how-to guide for `debug` subtools](../../../../how-to/use_debug_subtools_effectively.md).
