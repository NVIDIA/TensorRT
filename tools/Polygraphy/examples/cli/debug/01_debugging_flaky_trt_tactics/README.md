# Debugging Flaky TensorRT Tactics


## Introduction

Sometimes, a tactic in TensorRT may produce incorrect results, or have
otherwise buggy behavior. Since the TensorRT builder relies on timing
tactics, engine builds are non-deterministic, which can make tactic bugs
manifest as flaky failures.

One approach to tackling the problem is to run the builder several times,
saving tactic replay files from each run. Once we have a set of known-good and
known-bad tactics, we can compare them to determine which tactic
is likely to be the source of error.

The `debug build` subtool allows us to automate this process.

For more details on how the `debug` tools work, see [here](polygraphy/tools/debug/).

## Running The Example

For this example, we'll break the process down into 3 steps:

1. Generate golden outputs from ONNX-Runtime:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-outputs golden.json
    ```

2. Use `debug build` to repeatedly build TensorRT engines (in this case, for 2 iterations, specified in `--until`)
    and compare results against the golden outputs, saving a tactic replay file each time:

    ```bash
    polygraphy debug build identity.onnx --fp16 --save-tactics replay.json \
        --artifacts-dir replays --artifacts replay.json --until=2 \
        --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json
    ```

    `debug build` will build the engine, in this case with FP16 mode enabled,
    and write it to a file called `polygraphy_debug.engine` in the current directory.
    During each iteration, the engine saved during the previous iteration will be overwritten.

    *TIP: `debug build` supports all the TensorRT builder configuration options supported by other tools,*
        *like `convert` or `run`. See `polygraphy debug build -h` for details.*

    The `--save-tactics replay.json` option will write out a tactic replay file to `replay.json` for each iteration.

    Since we want to sort these into `good` and `bad` replays, we let `debug build` manage
    them by specifying them as `--artifacts`. If the `--check` command succeeds,
    the run is considered `good` and the tactic replay will be moved to `replays/good`.
    Otherwise, it will be considered `bad` and the tactic replay will be moved to `replays/bad`.

    In our `--check` command, we compare our TensorRT results to the previously generated
    golden outputs. If the outputs don't match, the command will fail.

    *TIP: For finer control over what qualifies as a `--check` success/failure, you can use the*
        *`--fail-regex`, `--fail-code`, and `--ignore-fail-code` options. See `polygraphy debug build -h` for details.*
        *By default, only the status code is taken into consideration.*

    *NOTE: In this case, all the replay files should be copied into the `good` directory - it's*
        *very unlikely that a simple identity model will fail.*

3. Use `debug diff-tactics` to determine which tactics could be bad:

    ```bash
    polygraphy debug diff-tactics --dir replays
    ```

    *NOTE: This last step should report that it could not determine potentially bad tactics since*
        *our `bad` directory is empty at this point:*

    <!-- Polygraphy Test: Ignore Start -->
    ```
    [I] Loaded 2 good tactic replays.
    [I] Loaded 0 bad tactic replays.
    [I] Could not determine potentially bad tactics. Try generating more tactic replay files?
    ```
    <!-- Polygraphy Test: Ignore End -->
