# Polygraphy Plugin Autotuner CLI Demo

A simple demo showing how to use the **Polygraphy Plugin Autotuner** CLI to automatically optimize ONNX models by replacing subgraphs with TensorRT plugins.

## 🚀 Quick Start

### Basic Demo (Recommended)

```bash
python cli_demo.py
```

This will create a test model, run optimization, and generate a report.

### Save/Load Demo (Advanced)

```bash
python save_load_demo.py
```

Two-step workflow: (1) Match patterns → (2) Optimize with saved config.

## 📁 Files

- `cli_demo.py` - Complete end-to-end optimization workflow (cross-platform)
- `save_load_demo.py` - Two-step workflow (match → save config → optimize, cross-platform)
- `create_test_model.py` - Test ONNX model generator
- `README.md` - This documentation

## 📖 Usage Options

### Customize Model Size
```bash
python cli_demo.py --groups 5
# or
NUM_GROUPS=5 python cli_demo.py
```

### Skip Cleanup Prompt
```bash
python cli_demo.py --no-cleanup
```

### Help
```bash
python cli_demo.py --help
python save_load_demo.py --help
```

### Save/Load Demo Options
```bash
# Run both steps (save config → load and optimize)
python save_load_demo.py

# Run only pattern matching step
python save_load_demo.py --step save

# Run only optimization step (requires existing config)
python save_load_demo.py --step load

# Skip cleanup prompt
python save_load_demo.py --no-cleanup
```

### Manual CLI Usage
The demo script prints the actual Polygraphy command it executes. You can also run it manually:

```bash
# Basic autotune
polygraphy plugin autotune conv_relu_test_model.onnx --plugin-dir ../patterns \
  --python-plugins ../plugins/trt_conv_relu_plugin.py -o optimized.onnx

# With all options
polygraphy plugin autotune conv_relu_test_model.onnx \
  --plugin-dir ../patterns \
  --python-plugins ../plugins/trt_conv_relu_plugin.py \
  --autotune-mode greedy \
  --max-combinations 1000 \
  --threshold 0.1 \
  -o optimized.onnx \
  --report report.txt

# Match patterns only
polygraphy plugin match conv_relu_test_model.onnx --plugin-dir ../patterns -o matches.yaml

# Use saved config
polygraphy plugin autotune conv_relu_test_model.onnx \
  --plugin-dir ../patterns \
  --python-plugins ../plugins/trt_conv_relu_plugin.py \
  --load-config-yaml matches.yaml \
  -o optimized.onnx
```

### Key Options

- `--plugin-dir`: Directory containing plugin pattern definitions (`pattern.py` files) **[Required]**
- `--python-plugins`: Python file(s) to import that register TensorRT plugins (optional, list one or more)
- `--plugin-libraries`: Shared library (`.so`) plugin file(s) to load (optional, list one or more)
- `--autotune-mode`: Plugin selection strategy: `greedy` (fast) or `exhaustive` (thorough) (default: `greedy`)
- `--max-combinations`: Maximum plugin combinations to test (default: 1000)
- `--threshold`: Performance improvement threshold for replacement (default: 0.1)
- `--verbose`: Enable detailed logging