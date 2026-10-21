# SwiGLU Demo - Multi-Input/Output and Nested Patterns

This demo showcases advanced AutoTuner features:

- **Multi-input/Multi-output support**: Plugins with multiple inputs and outputs
- **Nested subgraph patterns**: Swish as a sub-pattern within SwiGLU
- **Check function (check_func)**: Selective pattern matching based on data types

## SwiGLU Pattern

```
Input (a)
  |
Gemm1 (a, b1) -> c1
  |
Gemm2 (a, b2) -> temp
  |
Swish (temp) -> c2      # Nested pattern: x * sigmoid(x)
  |
Mul (c1, c2) -> d
  |
Outputs: d, c2          # Multiple outputs
```

**Key Features:**
- 3 inputs (input, weight1, weight2), 2 outputs (d, c2)
- Swish defined as reusable nested sub-pattern
- `check_func` filters by data type (only replaces fp16 version)

## Usage

```bash
# Combined model (fp16 + fp32) - demonstrates check_func
python3 swiglu_demo.py

# Simple fp16 model only
python3 swiglu_demo.py --simple

# Verbose logging
python3 swiglu_demo.py -v

# Custom model
python3 swiglu_demo.py -i model.onnx -o optimized.onnx
```

### Command Line Options
- `--input, -i`: Input ONNX model path (creates one if not provided)
- `--output, -o`: Output ONNX model path (default: `optimized_swiglu_model.onnx`)
- `--simple, -s`: Use simple fp16 model instead of combined model
- `--verbose, -v`: Enable verbose logging

## Model Types

### Combined Model (default)
```
Input (fp16) -> SwiGLU_fp16 -> final_fp16_output
Input (fp32) -> SwiGLU_fp32 -> final_fp32_output
```

The `check_func` only replaces the fp16 SwiGLU layer, leaving fp32 as native ops.

### Simple Model (--simple)
```
Input (fp16) -> SwiGLU -> Outputs (d, c2)
```

Single SwiGLU layer replaced with plugin.

## Performance

- **Plugin fusion**: Combines 2x Gemm + Sigmoid + 2x Mul into single plugin
- **Memory efficiency**: Reduces intermediate tensor allocations
- **Expected speedup**: From fused operations

## Files

- `swiglu_demo.py`: Main demo script
- `plugins/trt_swiglu_plugin.py`: SwiGLU plugin implementation
- `patterns/swigluPlugin/pattern.py`: Pattern definition with nested sub-patterns
