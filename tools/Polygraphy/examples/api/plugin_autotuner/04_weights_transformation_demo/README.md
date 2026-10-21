# Weights Transformation Demo

This demo demonstrates **pre-processing transformations** (Transpose, Reshape) injected on constant weight inputs before they are passed to a plugin. These transformations are **constant-folded by TensorRT** at compile time, resulting in zero runtime overhead.

## Overview

The demo customizes the plugin pattern's `replace_with_plugin` function to inject:
1. **Transpose** operation (swap H/W dimensions on Conv weights)
2. **Reshape** operation (restore original shape)

When weights are constants (initializers), TensorRT constant-folds these operations during engine compilation.

## Features Demonstrated

### 1. Custom `replace_with_plugin` Implementation
- Located in local `patterns/convReluPlugin/pattern.py`
- Checks if weight input is a constant
- Injects Transpose + Reshape nodes on constant weights
- TensorRT constant-folds at compile time → **zero runtime overhead**

### 2. Visual Model Structure Display
- Shows injected transformation nodes marked with ⚠️
- Shows plugin nodes marked with 🔌
- Displays node attributes and connections

### 3. Local Patterns Directory
- This demo uses its **own `patterns/` directory** with custom `replace_with_plugin` logic
- Avoids modifying shared patterns used by other demos
- Demonstrates pattern-specific customizations

## Usage

```bash
# Run with default settings (1 Conv+ReLU group)
python3 weights_transformation_demo.py

# Custom number of groups
python3 weights_transformation_demo.py -g 3

# Show original and optimized model structures
python3 weights_transformation_demo.py -s

# Verbose logging
python3 weights_transformation_demo.py -v
```

### Command Line Options
- `--input, -i`: Input ONNX model path (creates one if not provided)
- `--output, -o`: Output ONNX model path (default: `optimized_conv_relu_model.onnx`)
- `--num-groups, -g`: Number of Conv+ReLU groups (default: 1)
- `--show-structure, -s`: Display ONNX model structure before/after
- `--verbose, -v`: Enable verbose logging

## How It Works

### Weight Transformation Flow

```
Original Weight (Constant: [N, C, H, W])
         ↓
  Transpose (perm=[0,1,3,2])  ⚠️ Injected at pattern replacement
         ↓
Transformed Weight ([N, C, W, H])
         ↓
  Reshape (back to [N, C, H, W])  ⚠️ Injected at pattern replacement
         ↓
    Plugin Input
         ↓
[TensorRT Constant Folding at Compile Time]
         ↓
Pre-transformed Constant → Plugin  (Zero runtime overhead)
```

### Implementation Snippet

```python
# patterns/convReluPlugin/pattern.py
def replace_with_plugin(graph, input_tensors, output_tensors, attrs, op):
    weight = input_tensors[1]
    
    # Check if weight is constant
    if _get_const_values(graph, weight) is not None:
        # Insert Transpose: swap H and W
        t_out = graph.layer(op="Transpose", inputs=[weight], 
                           attrs={"perm": [0, 1, 3, 2]})
        
        # Insert Reshape: restore shape
        r_out = graph.layer(op="Reshape", 
                           inputs=[t_out, shape_const])
        
        # Use transformed weight as plugin input
        plugin_inputs = [input_tensors[0], r_out, input_tensors[2]]
    
    # Call default replacement with transformed weight
    return default_replace_with_plugin(graph, plugin_inputs, ...)
```

## Why This Works

**Advantages:**
- **Zero Runtime Overhead**: Transformations constant-folded at compile time
- **Flexibility**: Different patterns can inject different transformations
- **Optimization**: Leverages TensorRT's constant folding engine

**Requirements:**
- Input must be a constant (initializer or Constant node)
- Transformation must be deterministic
- TensorRT must support the transformation operations

## Directory Structure

```
04_weights_transformation_demo/
├── weights_transformation_demo.py
├── README.md
└── patterns/                          # ⚠️ LOCAL patterns (not shared)
    └── convReluPlugin/
        └── pattern.py                 # Custom with weight transformation
```

**vs. Other Demos:**
```
01_conv_relu_demo/
├── conv_relu_demo.py
└── (uses ../patterns/)                # ✓ SHARED patterns from parent
    └── convReluPlugin/
        └── pattern.py                 # Standard replacement
```

## Extensibility

You can extend this pattern to support other transformations:

- `Transpose`: Permute dimensions
- `Reshape`: Change tensor shape
- `Cast`: Type conversion
- `Squeeze/Unsqueeze`: Add/remove dimensions
- `Slice`: Extract sub-tensors
- `Concat`: Combine multiple constants

## Files

- `weights_transformation_demo.py`: Main demo script
- `patterns/convReluPlugin/pattern.py`: Custom pattern with transformation injection
- `README.md`: This file

## Use Cases

This technique is useful when:
- Plugin expects weights in a different layout than model provides
- Need to apply preprocessing that can be constant-folded
- Want to optimize memory layout without runtime overhead
- Migrating between different framework weight formats
