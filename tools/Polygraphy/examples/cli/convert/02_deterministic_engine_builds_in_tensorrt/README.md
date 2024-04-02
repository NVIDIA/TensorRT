# Deterministic Engine Building In TensorRT

**NOTE: This example requires TensorRT 8.7 or newer.**

## Introduction

During engine building, TensorRT runs and times several kernels in order to select
the most optimal ones. Since kernel timings may vary slightly from run to run, this
process is inherently non-deterministic.

In many cases, deterministic engine builds may be desirable. One way of achieving this
is to use a timing cache to ensure the same kernels are picked each time.

## Running The Example

1. Build an engine and save a timing cache:

    ```bash
    polygraphy convert identity.onnx \
        --save-timing-cache timing.cache \
        -o 0.engine
    ```

2. Use the timing cache for another engine build:

    ```bash
    polygraphy convert identity.onnx \
        --load-timing-cache timing.cache --error-on-timing-cache-miss \
        -o 1.engine
    ```

    We specify `--error-on-timing-cache-miss` so that we can be sure that the new engine
    used the entries from the timing cache for each layer.

3. Verify that the engines are exactly the same:

    <!-- Polygraphy Test: Ignore Start -->
    ```bash
    diff <(polygraphy inspect model 0.engine --show layers attrs) <(polygraphy inspect model 1.engine --show layers attrs)
    ```
    <!-- Polygraphy Test: Ignore End -->
