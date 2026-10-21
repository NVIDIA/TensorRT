# Timing Cache Demo - TensorRT Performance Optimization

This demo showcases TensorRT timing cache performance benefits. Timing cache stores kernel timing information from previous builds, allowing subsequent builds to skip kernel profiling.

## Benefits

- Skip repeated kernel timing measurements during engine builds
- Reduce build time by 50-80%
- Accelerate iterative model optimization
- Maintain consistent kernel selection across builds

## Features Demonstrated

### 1. Three-Mode Comparison
- **Without Cache**: Baseline (profiles all kernels)
- **Temp Cache**: Creates temporary cache during optimization
- **Given Cache**: Loads persistent cache file (`persistent_cache.trt`)

### 2. AutoTuner Integration
- Shows timing cache integration with AutoTuner workflow
- Tracks compile time statistics across multiple engine builds
- Large Conv+ReLU model (20 groups) demonstrates cache effectiveness

## Model Structure

```
Input -> Conv1+ReLU1 -> Abs1 -> Conv2+ReLU2 -> Abs2 -> ... -> ConvN+ReLUN -> Output
```

**Parameters:** Batch 1, 64 channels, 128x128, 3x3 kernels, 20 groups (default)

## Usage

```bash
# Run with default settings (20 groups)
python3 autotuner_timing_cache_demo.py

# Custom number of groups
python3 autotuner_timing_cache_demo.py -g 30

# Verbose logging
python3 autotuner_timing_cache_demo.py -v

# Custom input model
python3 autotuner_timing_cache_demo.py -i model.onnx
```

### Command Line Options
- `--input, -i`: Input ONNX model path (creates one if not provided)
- `--num-groups, -g`: Number of Conv+ReLU groups (default: 20)
- `--verbose, -v`: Enable verbose logging

## How It Works

```
Run 1 (No Cache):   Pattern Match -> Build Engines -> Profile Kernels -> Measure
Run 2 (Temp Cache): Pattern Match -> Build Engines -> Use Temp Cache -> Measure
Run 3 (Given Cache): Pattern Match -> Build Engines -> Load Cache -> Measure
```

**Cache File:**
- Location: `persistent_cache.trt`
- Size: ~100KB - 1MB
- Persistent across runs
- Shareable across similar hardware

## AutoTuner API Usage

```python
# Enable timing cache
autotuner = AutoTuner(
    builder=builder,
    config=config,
    enable_timing_cache=True,              # Enable cache
    timing_cache_path="my_cache.trt"       # Persistent cache path
)

# Run optimization (cache is used automatically)
result = autotuner.optimize_network(...)

# Check statistics
print(f"Total compile time: {result.total_compile_time:.2f}s")
print(f"Engine builds: {result.num_engine_builds}")
```

## Files

- `autotuner_timing_cache_demo.py`: Main demo script (uses AutoTuner)
- `pluginv3_timing_cache_demo.py`: Lower-level demo (direct TensorRT API)
- `persistent_cache.trt`: Generated timing cache file
- `config.yaml`: Plugin replacement configuration
