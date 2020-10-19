# [EXPERIMENTAL] Precision

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Modes](#modes)
- [Examples](#examples)
    - [Debugging An Inaccurate Network](#debugging-an-inaccurate-network)


## Introduction

The `precision` tool can be used to determine which layers of a TensorRT network need to be
run in a higher precision in order to maintain the desired accuracy.

The tool works by iteratively marking a subset of the layers in the network in the specified
higher precision (`float32` by default). Then, it checks the accuracy against golden outputs
and attempts to find the minimum number of layers that should be run in a higher precision.

The golden outputs required can be generated separately by other Polygraphy tools, such as `polygraphy run`

**NOTE:** You can greatly increase the speed of this tool by allowing it to create a calibration
cache. See step 2 in the [example](#debugging-an-inaccurate-network) for details.


## Usage

See `polygraphy precision -h` for usage information.


## Modes

This tool includes several different subtools, or modes, which control how it decides which layers
to mark in a higher precision.

- `bisect`: Performs a binary search by marking groups of adjacent layers in the network. For example, it
    may mark the first N layers in FP32. If the network passes, then it will attempt to mark only N/2 layers.
    If that passes, it will mark N/4 layers, otherwise, 3N/4, and so on.
- `linear`: Performs a linear search by marking layers one at a time. For example, it may mark only the 1st layer
    in FP32. If that fails, then it will mark the 2nd, then the 3rd, 4th, and so on.
- `worst-first`: Attempts to find the layers that introduce the most error (i.e. the layers that see the greatest
    increase in required tolerance compared to the previous layer) and marks the top N worst layers first.

    **NOTE:** This mode depends on being able to mark every layer of the network as an output. Additionally, it
    assumes that subsequent outputs come from subsequent layers, and therefore may not correctly find the worst layers
    if layers produce multiple outputs.


## Examples

### Debugging An Inaccurate Network

In this example, we assume that `model.onnx` is a model that is not working when all layers
are marked to run in `int8` precision.

1. Generate the golden outputs using ONNX Runtime:

    ```bash
    polygraphy run model.onnx --onnxrt --save-results=golden.pkl
    ```

2. Use `precision`'s `bisect` mode to perform a binary search over the model.

    ```bash
    polygraphy precision bisect model.onnx --golden=golden.pkl --mode=forward --int8 --precision=float32 --calibration-cache=model_calibration.cache
    ```

    `--mode=forward` indicates that the algorithm will mark layers to run in a
    higher precision starting from the inputs of the network. You can use `reverse` to
    mark layers starting from the outputs of the networks.

    `--precision=float32` indicates that the higher precision to mark layers with should be
    `trt.float32`. You can try using lower precisions like `float16` for better performance.

    `--calibration-cache` allows you to save calibration data to the disk, which eliminates the
    need to generate it each time. It is **strongly recommended** to use this option, as it can greatly
    speed up the process.

3. After `precision` completes, if acceptable accuracy was achieved, you should see a message like:
    ```
    [I] To achieve acceptable accuracy, try running the first 36 layers in DataType.FLOAT precision
    ```

    `precision` will also display information about tolerances
    required for each iteration, and information about which layers it ran in a higher precision.
