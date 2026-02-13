# Comparing INT8 and FP16 Accuracy

## Introduction

This example demonstrates how to debug accuracy issues when an INT8 quantized model shows significantly lower accuracy compared to an FP16 model. This is a common issue in TensorRT when using INT8 quantization for inference optimization.

The example covers:
1. Comparing layer-wise outputs between INT8 and FP16 engines
2. Parsing Polygraphy JSON output files to analyze errors
3. Using the `debug precision` tool to automatically identify problematic layers
4. Setting specific layers to FP32 precision to recover accuracy

## Prerequisites

- TensorRT 8.0 or later
- An ONNX model
- Calibration dataset (for INT8 quantization)

## Running The Example

### Step 1: Prepare Calibration Data

Create a data loader script for your calibration dataset:

```python
# calibration_data_loader.py
import numpy as np

def load_data():
    """
    Generator that yields calibration data.
    Modify this to load your actual calibration dataset.
    """
    for i in range(100):  # Use 500-1000 samples in practice
        # Generate or load your calibration data
        data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        yield {"input": data}
```

### Step 2: Build FP16 and INT8 Engines

Build an FP16 engine as baseline:

```bash
polygraphy convert model.onnx --fp16 -o model_fp16.engine
```

Build an INT8 engine with calibration:

```bash
polygraphy convert model.onnx \
    --int8 \
    --data-loader-script calibration_data_loader.py \
    --calibration-cache calibration.cache \
    -o model_int8.engine
```

### Step 3: Compare Overall Accuracy

First, let's see if there's an accuracy difference:

```bash
# Generate test inputs
polygraphy run model.onnx --onnxrt --save-inputs test_inputs.json

# Run FP16 engine
polygraphy run model_fp16.engine --trt \
    --load-inputs test_inputs.json \
    --save-outputs fp16_outputs.json

# Run INT8 engine and compare
polygraphy run model_int8.engine --trt \
    --load-inputs test_inputs.json \
    --load-outputs fp16_outputs.json
```

If the comparison fails, proceed to identify problematic layers.

### Step 4: Compare Layer-wise Outputs

Compare all intermediate layer outputs:

```bash
# Save all FP16 layer outputs
polygraphy run model.onnx --trt --fp16 \
    --trt-outputs mark all \
    --load-inputs test_inputs.json \
    --save-outputs fp16_all_layers.json

# Compare INT8 layer outputs against FP16
polygraphy run model.onnx --trt --int8 \
    --calibration-cache calibration.cache \
    --trt-outputs mark all \
    --load-inputs test_inputs.json \
    --load-outputs fp16_all_layers.json \
    --fail-fast
```

The `--fail-fast` option will stop at the first layer with significant error.

### Step 5: Parse and Analyze JSON Outputs

Use the provided script to analyze the layer-wise errors:

```bash
python3 parse_layer_errors.py \
    --fp16-outputs fp16_all_layers.json \
    --int8-outputs int8_all_layers.json \
    --threshold 0.1
```

This will generate a report showing which layers have the highest errors.

### Step 6: Use Debug Precision Tool (Automated)

The `debug precision` tool automates the process of finding which layers need higher precision:

```bash
polygraphy debug precision model.onnx \
    --int8 \
    --calibration-cache calibration.cache \
    --precision float32 \
    --mode bisect \
    --check polygraphy run model.onnx --fp16 --onnxrt \
        --load-inputs test_inputs.json \
        --save-outputs golden.json && \
        polygraphy run polygraphy_debug.engine --trt \
        --load-inputs test_inputs.json \
        --load-outputs golden.json
```

This will use binary search to efficiently identify which layers need to run in FP32.

### Step 7: Apply Precision Constraints

Once you've identified problematic layers, create a network postprocessing script:

```python
# fix_precision.py
import tensorrt as trt

def postprocess(network):
    # Replace with actual layer names from previous steps
    fp32_layers = ["Conv_0", "Conv_5", "Add_10"]
    
    for layer in network:
        if layer.name in fp32_layers:
            layer.precision = trt.float32
            for i in range(layer.num_outputs):
                layer.set_output_type(i, trt.float32)
```

Build the fixed engine:

```bash
polygraphy convert model.onnx \
    --int8 \
    --calibration-cache calibration.cache \
    --trt-network-postprocess-script fix_precision.py \
    --precision-constraints obey \
    -o model_int8_fixed.engine
```

### Step 8: Verify the Fix

```bash
polygraphy run model_int8_fixed.engine --trt \
    --load-inputs test_inputs.json \
    --load-outputs golden.json
```

## Alternative: Using Layer Precisions Option

Instead of a postprocessing script, you can use the `--layer-precisions` option:

```bash
polygraphy convert model.onnx \
    --int8 \
    --calibration-cache calibration.cache \
    --layer-precisions Conv_0:float32 Conv_5:float32 Add_10:float32 \
    --precision-constraints obey \
    -o model_int8_fixed.engine
```

## Understanding the Output

When comparing outputs, Polygraphy will show:

```
[I] Comparing Output: 'output' (dtype=float32, shape=(1, 1000)) with 'output' (dtype=float32, shape=(1, 1000))
[I]     Tolerance: [abs=0.001, rel=0.001] | Checking elemwise error
[I]     fp16-runner-N0-01/01/24-12:00:00: output | Stats: mean=0.001, std-dev=0.0005, var=2.5e-07, median=0.0009, min=0 at (0, 0), max=0.005 at (0, 500)
[I]     int8-runner-N0-01/01/24-12:00:01: output | Stats: mean=0.001, std-dev=0.002, var=4e-06, median=0.0008, min=0 at (0, 0), max=0.05 at (0, 500)
[E]     FAILED | Difference exceeds tolerance (rel=0.001, abs=0.001)
```

## Tips

1. **Start with a small number of calibration samples** for faster iteration during debugging
2. **Use `--fail-fast`** to quickly identify the first problematic layer
3. **Try different calibration algorithms** if accuracy is poor:
   - `--calibration-algo=entropy` (default)
   - `--calibration-algo=minmax`
   - `--calibration-algo=percentile`
4. **Monitor performance impact** of FP32 layers using `trtexec --loadEngine=model.engine`
5. **Consider Quantization-Aware Training (QAT)** if too many layers need FP32

## See Also

- [INT8 Calibration in TensorRT](../../../cli/convert/01_int8_calibration_in_tensorrt/)
- [Adding Precision Constraints](../../../cli/run/08_adding_precision_constraints/)
- [Working with Reduced Precision](../../../../how-to/work_with_reduced_precision.md)
- [Debugging Accuracy Issues](../../../../how-to/debug_accuracy.md)
