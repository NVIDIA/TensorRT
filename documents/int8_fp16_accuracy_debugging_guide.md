# INT8 vs FP16 Accuracy Debugging Guide for TensorRT

## Overview

This guide addresses the common issue of INT8 quantized models showing significantly lower accuracy compared to FP16 models in TensorRT. It provides comprehensive solutions for:

1. **Comparing layer-wise errors** between INT8 and FP16 engines
2. **Identifying problematic layers** with large errors
3. **Setting specific layers to FP32 precision** to recover accuracy
4. **Parsing Polygraphy JSON output** for analysis
5. **Using trtexec for calibration** with custom data

## Table of Contents

- [Prerequisites](#prerequisites)
- [Question 1: Layer-wise Error Comparison Tools](#question-1-layer-wise-error-comparison-tools)
- [Question 2: Parsing Polygraphy JSON Output](#question-2-parsing-polygraphy-json-output)
- [Question 3: Using trtexec for Calibration](#question-3-using-trtexec-for-calibration)
- [Complete Workflow Example](#complete-workflow-example)
- [Best Practices](#best-practices)

## Prerequisites

- TensorRT 8.0 or later
- Polygraphy (included with TensorRT or install via `pip install polygraphy`)
- Python 3.6+
- ONNX model and calibration dataset

## Question 1: Layer-wise Error Comparison Tools

### Method 1: Using Polygraphy `debug precision` (Recommended)

The `polygraphy debug precision` tool automatically identifies which layers need higher precision:

```bash
# Compare INT8 engine against FP16 reference
polygraphy debug precision model.onnx \
    --int8 \
    --fp16 \
    --calibration-cache calibration.cache \
    --check polygraphy run model.onnx --fp16 --onnxrt \
        --save-outputs fp16_golden.json && \
        polygraphy run polygraphy_debug.engine --trt \
        --load-outputs fp16_golden.json
```

This tool will:
- Iteratively mark layers to run in FP32
- Use binary search (bisect mode) to efficiently find problematic layers
- Output which layers need higher precision

### Method 2: Layer-wise Output Comparison

Compare all intermediate layer outputs between INT8 and FP16:

```bash
# Step 1: Run FP16 engine and save all layer outputs
polygraphy run model.onnx \
    --trt --fp16 \
    --trt-outputs mark all \
    --save-outputs fp16_all_outputs.json

# Step 2: Run INT8 engine and save all layer outputs
polygraphy run model.onnx \
    --trt --int8 \
    --calibration-cache calibration.cache \
    --trt-outputs mark all \
    --save-outputs int8_all_outputs.json

# Step 3: Compare the outputs
polygraphy run model.onnx \
    --trt --int8 \
    --calibration-cache calibration.cache \
    --trt-outputs mark all \
    --load-outputs fp16_all_outputs.json \
    --fail-fast
```

The `--fail-fast` option will stop at the first layer with significant error, helping you identify the problematic layer.

### Method 3: Using Python API for Custom Analysis

See the companion script `compare_layer_outputs.py` for a detailed implementation.

## Question 2: Parsing Polygraphy JSON Output

Polygraphy saves outputs in a structured JSON format. Here's how to parse and analyze them:

### Understanding the JSON Structure

The JSON file contains a `RunResults` object with this structure:

```json
{
  "runner_name": [
    {
      "output_name_1": {
        "__type__": "ndarray",
        "dtype": "float32",
        "shape": [1, 3, 224, 224],
        "values": [...]
      },
      "output_name_2": {...}
    }
  ]
}
```

### Parsing with Python

```python
from polygraphy.comparator import RunResults
import numpy as np

# Load the JSON files
fp16_results = RunResults.load("fp16_all_outputs.json")
int8_results = RunResults.load("int8_all_outputs.json")

# Extract outputs for comparison
for runner_name in fp16_results.keys():
    fp16_outputs = fp16_results[runner_name][0]  # First iteration
    int8_outputs = int8_results[runner_name][0]
    
    # Compare each layer
    for layer_name in fp16_outputs.keys():
        if layer_name in int8_outputs:
            fp16_array = fp16_outputs[layer_name]
            int8_array = int8_outputs[layer_name]
            
            # Calculate error metrics
            abs_diff = np.abs(fp16_array - int8_array)
            rel_diff = abs_diff / (np.abs(fp16_array) + 1e-8)
            
            print(f"Layer: {layer_name}")
            print(f"  Max absolute error: {np.max(abs_diff)}")
            print(f"  Mean absolute error: {np.mean(abs_diff)}")
            print(f"  Max relative error: {np.max(rel_diff)}")
            print(f"  Mean relative error: {np.mean(rel_diff)}")
```

See the companion script `parse_polygraphy_outputs.py` for a complete implementation.

### Using Polygraphy CLI to Inspect Outputs

```bash
# View summary statistics
polygraphy inspect data fp16_all_outputs.json

# View actual values
polygraphy inspect data fp16_all_outputs.json --show-values

# Compare two output files
polygraphy run --load-outputs fp16_all_outputs.json \
    --load-outputs int8_all_outputs.json
```

## Question 3: Using trtexec for Calibration

While `trtexec` doesn't directly support custom calibration data, you can use it with a pre-generated calibration cache.

### Step 1: Generate Calibration Cache with Polygraphy

Create a data loader script (`calibration_data_loader.py`):

```python
import numpy as np

def load_data():
    """
    Generator function that yields calibration data.
    Replace this with your actual data loading logic.
    """
    # Load your 1000+ calibration images
    for i in range(1000):
        # Load and preprocess your data
        data = load_your_image(i)  # Implement this
        data = preprocess(data)     # Implement this
        
        # Yield as a dictionary mapping input names to numpy arrays
        yield {"input": data}

# Alternative: Load from a dataset
def load_data_from_dataset():
    import glob
    from PIL import Image
    
    image_files = glob.glob("/path/to/calibration/images/*.jpg")
    
    for img_path in image_files[:1000]:
        img = Image.open(img_path)
        # Preprocess image
        img = img.resize((224, 224))
        img_array = np.array(img).astype(np.float32)
        img_array = img_array / 255.0  # Normalize
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        yield {"input": img_array}
```

Generate the calibration cache:

```bash
polygraphy convert model.onnx \
    --int8 \
    --data-loader-script calibration_data_loader.py \
    --calibration-cache model_calibration.cache \
    -o model_int8.engine
```

### Step 2: Use the Cache with trtexec

```bash
trtexec --onnx=model.onnx \
    --int8 \
    --calib=model_calibration.cache \
    --saveEngine=model_int8.engine
```

### Alternative: Direct Calibration with Polygraphy (Recommended)

Instead of using trtexec, use Polygraphy directly for better control:

```bash
# Build INT8 engine with calibration
polygraphy convert model.onnx \
    --int8 \
    --data-loader-script calibration_data_loader.py \
    --calibration-cache model_calibration.cache \
    -o model_int8.engine

# Reuse the cache for subsequent builds
polygraphy convert model.onnx \
    --int8 \
    --calibration-cache model_calibration.cache \
    -o model_int8.engine
```

## Complete Workflow Example

Here's a complete workflow to debug INT8 accuracy issues:

### Step 1: Build FP16 Baseline

```bash
polygraphy convert model.onnx \
    --fp16 \
    -o model_fp16.engine
```

### Step 2: Build INT8 Engine with Calibration

```bash
polygraphy convert model.onnx \
    --int8 \
    --data-loader-script calibration_data_loader.py \
    --calibration-cache model_calibration.cache \
    -o model_int8.engine
```

### Step 3: Compare Outputs

```bash
# Generate test inputs
polygraphy run model.onnx --onnxrt \
    --save-inputs test_inputs.json

# Run both engines and compare
polygraphy run model_fp16.engine --trt \
    --load-inputs test_inputs.json \
    --save-outputs fp16_outputs.json

polygraphy run model_int8.engine --trt \
    --load-inputs test_inputs.json \
    --save-outputs int8_outputs.json \
    --load-outputs fp16_outputs.json
```

### Step 4: Identify Problematic Layers

```bash
# Compare layer-wise outputs
polygraphy run model.onnx \
    --trt --fp16 \
    --trt-outputs mark all \
    --load-inputs test_inputs.json \
    --save-outputs fp16_layer_outputs.json

polygraphy run model.onnx \
    --trt --int8 \
    --calibration-cache model_calibration.cache \
    --trt-outputs mark all \
    --load-inputs test_inputs.json \
    --load-outputs fp16_layer_outputs.json \
    --fail-fast
```

### Step 5: Use Debug Precision Tool

```bash
polygraphy debug precision model.onnx \
    --int8 \
    --calibration-cache model_calibration.cache \
    --precision float32 \
    --check polygraphy run model.onnx --fp16 --onnxrt \
        --load-inputs test_inputs.json \
        --save-outputs golden.json && \
        polygraphy run polygraphy_debug.engine --trt \
        --load-inputs test_inputs.json \
        --load-outputs golden.json
```

### Step 6: Apply Precision Constraints

Create a network postprocessing script (`fix_precision.py`):

```python
import tensorrt as trt

def postprocess(network):
    """
    Set specific layers to FP32 precision.
    """
    # List of layer names that need FP32 precision
    # (identified from previous steps)
    fp32_layers = ["layer_name_1", "layer_name_2", "layer_name_3"]
    
    for layer in network:
        if layer.name in fp32_layers:
            print(f"Setting {layer.name} to FP32")
            layer.precision = trt.float32
            # Also set output type to prevent FP16 storage
            for i in range(layer.num_outputs):
                layer.set_output_type(i, trt.float32)
```

Build engine with constraints:

```bash
polygraphy convert model.onnx \
    --int8 \
    --calibration-cache model_calibration.cache \
    --trt-network-postprocess-script fix_precision.py \
    --precision-constraints obey \
    -o model_int8_fixed.engine
```

### Step 7: Verify Accuracy

```bash
polygraphy run model_int8_fixed.engine --trt \
    --load-inputs test_inputs.json \
    --load-outputs golden.json
```

## Best Practices

### 1. Calibration Data Quality

- Use **representative data** from your actual use case
- Include **edge cases** and **diverse samples**
- Use at least **500-1000 samples** for calibration
- Ensure data preprocessing matches inference preprocessing

### 2. Layer Precision Selection

- Start with the **first failing layer** identified by layer-wise comparison
- Use **binary search** (debug precision bisect mode) for efficiency
- Consider setting **entire subgraphs** to FP32 if layers are tightly coupled
- Monitor **performance impact** of FP32 layers

### 3. Calibration Cache Management

- **Save and version** calibration caches for reproducibility
- **Regenerate caches** when model architecture changes
- Test with **different calibration algorithms** (Entropy, MinMax, Percentile)

### 4. Validation

- Always validate on a **separate test set**
- Compare against **FP32 or FP16 baseline**
- Use appropriate **tolerance thresholds** for your application
- Monitor **per-class metrics** for segmentation/classification tasks

### 5. Iterative Refinement

1. Identify problematic layers
2. Set them to FP32
3. Measure accuracy improvement
4. Measure performance impact
5. Iterate if needed

## Troubleshooting

### Issue: All layers show high error

- **Check calibration data quality**
- Verify preprocessing pipeline
- Try different calibration algorithms
- Consider if INT8 is appropriate for your model

### Issue: Debug precision doesn't converge

- Increase tolerance thresholds
- Check if model is inherently sensitive to quantization
- Consider Quantization-Aware Training (QAT)

### Issue: Performance degradation with FP32 layers

- Minimize number of FP32 layers
- Group FP32 layers to reduce format conversions
- Consider FP16 as intermediate precision
- Profile with `trtexec` or `nsys`

## Additional Resources

- [TensorRT Developer Guide - INT8](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
- [Polygraphy Documentation](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy)
- [Quantization-Aware Training](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#best-practices)

## Summary

To address the original issue:

1. **Layer-wise comparison**: Use `polygraphy debug precision` or layer-wise output comparison with `--trt-outputs mark all`

2. **Parse JSON outputs**: Use `RunResults.load()` API or the provided `parse_polygraphy_outputs.py` script

3. **trtexec calibration**: Generate calibration cache with Polygraphy using `--data-loader-script`, then use with trtexec via `--calib` flag

The recommended approach is to use Polygraphy's `debug precision` tool, which automates the entire process of identifying and fixing problematic layers.
