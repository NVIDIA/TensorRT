# Deterministic Engine Building In TensorRT


## Introduction

During engine building, TensorRT runs and times several kernels in order to select
the most optimal ones. Since kernel timings may vary slightly from run to run, this
process is inherently non-deterministic.

In many cases, deterministic engine builds may be desirable. One way of achieving this
is to use the `IAlgorithmSelector` API to ensure the same kernels are picked each time.

To make this process easier, Polygraphy provides two built-in algorithm selectors:
`TacticRecorder` and `TacticReplayer`. The former can be used to record tactics selected
during an engine build, and the latter to play them back during a subsequent build.
The CLI tools include `--save-tactics` and `--load-tactics` options correspnding to these.

## Running The Example

1. Build an engine and save a replay file:

    ```bash
    polygraphy convert identity.onnx \
        --save-tactics replay.json \
        -o 0.engine
    ```

    The resulting `replay.json` file is human-readable. Optionally, we can
    use `inspect tactics` to view it in a friendly format:

    ```bash
    polygraphy inspect tactics replay.json
    ```

2. Use the replay file for another engine build:

    ```bash
    polygraphy convert identity.onnx \
        --load-tactics replay.json \
        -o 1.engine
    ```

3. Verify that the engines are exactly the same:

    ```bash
    diff -sa 0.engine 1.engine
    ```
