# Conv+ReLU Demo - Python API Usage

Demonstrates using the **Plugin AutoTuner Python API** directly to replace Conv+ReLU patterns with TensorRT plugins.

## 🚀 Quick Start

```bash
# Default: 2 Conv+ReLU groups
python conv_relu_demo.py

# Custom model size
python conv_relu_demo.py -g 5

# Verbose logging
python conv_relu_demo.py -v
```

## 📋 What This Demo Shows

This demo showcases the **Python API workflow** for plugin replacement:

1. **Pattern Matching** - Find Conv+ReLU subgraphs in ONNX model
2. **AutoTuner Optimization** - Test replacements and select optimal strategy
3. **Model Generation** - Save optimized ONNX model with plugins
4. **Report Generation** - Create performance and replacement report

### Key Difference from `01_cli_demo`
- **01_cli_demo** (`examples/cli/plugin_autotuner/01_cli_demo`): Uses `polygraphy plugin autotune` CLI command
- **01_conv_relu_demo**: Uses Python API (`ReplacementEngine`, `AutoTuner`) directly

## 📖 Usage Options

```bash
# Basic usage (creates test model automatically)
python conv_relu_demo.py

# Custom number of Conv+ReLU groups
python conv_relu_demo.py -g 5

# Use existing model
python conv_relu_demo.py --input my_model.onnx --output optimized.onnx

# Enable verbose logging
python conv_relu_demo.py -v
```

### Command Line Options
- `--input, -i`: Input ONNX model path (creates test model if not provided)
- `--output, -o`: Output path (default: `optimized_conv_relu_model.onnx`)
- `--num-groups, -g`: Number of Conv+ReLU groups in test model (default: 2)
- `--verbose, -v`: Enable detailed logging

## 🔧 Python API Workflow

```python
# 1. Initialize replacement engine with patterns
repl = ReplacementEngine()
repl.load_plugins(patterns_dir)

# 2. Match Conv+ReLU patterns and save to config
repl.match_subgraphs(model_path, config_path)

# 3. Initialize AutoTuner
autotuner = AutoTuner(
    builder=builder,
    config=config,
    replacement_engine=repl
)

# 4. Run optimization with config
result = autotuner.optimize_network(
    input_source=model_path,
    load_config_yaml=config_path,
    use_greedy=True
)

# 5. Save optimized model
temp_path = autotuner.get_optimized_model_path(result)
model = onnx.load(temp_path)
onnx.save(model, output_path)
```

## 📊 Generated Files

After running the demo, you'll find:

- `native_conv_relu_model.onnx` - Original model with native Conv+ReLU
- `optimized_conv_relu_model.onnx` - Optimized model with plugins
- `config.yaml` - Plugin configuration for matched patterns

## 📈 Expected Results

The demo will:
1. **Find patterns**: Identifies Conv+ReLU subgraphs (typically 2 matches)
2. **Test replacements**: Measures performance with/without plugins
3. **Select optimal strategy**: Chooses best replacement combination
4. **Display results**: Shows latency, throughput, and optimization metrics

**Performance improvement**: Expect speedup from fused Conv+ReLU operations

## 💡 Key Takeaways

- **Python API**: Direct control over AutoTuner workflow
- **Pattern-based**: Uses reusable pattern definitions from `../patterns/`
- **Automatic optimization**: AutoTuner handles performance testing and selection
- **Config-based workflow**: Match patterns → save config → optimize with config

## 📁 Files

- `conv_relu_demo.py` - Demo script using Python API
- `config.yaml` - Generated pattern configuration
- `native_conv_relu_model.onnx` - Generated test model (if created)
- `optimized_conv_relu_model.onnx` - Generated optimized model

## 🔍 Related Demos

- **[`01_cli_demo`](../../../cli/plugin_autotuner/01_cli_demo/)** - CLI-based workflow (`polygraphy plugin autotune`)
- **[`02_swiglu_demo`](../02_swiglu_demo/)** - Multi-input/output patterns and nested patterns
- **[`04_weights_transformation_demo`](../04_weights_transformation_demo/)** - Weight transformation injection

## 📝 Notes

**Plugin Registration**: The demo automatically imports and registers `trt_conv_relu_plugin` with TensorRT.

**Pattern Directory**: Uses shared patterns from `../patterns/convReluPlugin/`.

**Performance**: Results vary based on GPU, TensorRT version, and model size.


