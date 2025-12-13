# Quick Reference: INT8 vs FP16 Accuracy Debugging

## One-Line Solutions

### Compare layer-wise outputs
```bash
polygraphy run model.onnx --trt --fp16 --trt-outputs mark all --save-outputs fp16.json && \
polygraphy run model.onnx --trt --int8 --calibration-cache calib.cache --trt-outputs mark all --load-outputs fp16.json --fail-fast
```

### Analyze errors from JSON
```bash
python3 parse_layer_errors.py --fp16-outputs fp16.json --int8-outputs int8.json --threshold 0.1
```

### Auto-find problematic layers
```bash
polygraphy debug precision model.onnx --int8 --calibration-cache calib.cache --precision float32 --mode bisect
```

### Build INT8 with calibration
```bash
polygraphy convert model.onnx --int8 --data-loader-script calibration_data_loader.py --calibration-cache calib.cache -o model.engine
```

### Fix specific layers to FP32
```bash
polygraphy convert model.onnx --int8 --calibration-cache calib.cache --layer-precisions Conv_0:float32 Conv_5:float32 --precision-constraints obey -o fixed.engine
```

### Use postprocessing script
```bash
polygraphy convert model.onnx --int8 --calibration-cache calib.cache --trt-network-postprocess-script fix_precision.py --precision-constraints obey -o fixed.engine
```

## Common Commands

### Generate calibration cache
```bash
polygraphy convert model.onnx --int8 --data-loader-script calibration_data_loader.py --calibration-cache calib.cache -o model.engine
```

### Inspect JSON outputs
```bash
polygraphy inspect data outputs.json --show-values
```

### Compare two engines
```bash
polygraphy run model_fp16.engine --trt --save-outputs fp16.json
polygraphy run model_int8.engine --trt --load-outputs fp16.json
```

### Use cache with trtexec
```bash
trtexec --onnx=model.onnx --int8 --calib=calib.cache --saveEngine=model.engine
```

## Python Snippets

### Parse JSON outputs
```python
from polygraphy.comparator import RunResults
results = RunResults.load("outputs.json")
for runner_name, [outputs] in results.items():
    for name, array in outputs.items():
        print(f"{name}: {array.shape}")
```

### Calibration data loader
```python
def load_data():
    for i in range(1000):
        data = load_your_image(i)  # Your loading logic
        yield {"input": data}
```

### Fix precision script
```python
import tensorrt as trt
def postprocess(network):
    fp32_layers = ["Conv_0", "Conv_5"]
    for layer in network:
        if layer.name in fp32_layers:
            layer.precision = trt.float32
            for i in range(layer.num_outputs):
                layer.set_output_type(i, trt.float32)
```

## Workflow

1. **Build engines:** FP16 and INT8
2. **Compare:** Layer-wise outputs
3. **Analyze:** Identify problematic layers
4. **Fix:** Set layers to FP32
5. **Verify:** Test accuracy
6. **Profile:** Check performance

## Troubleshooting

| Issue | Solution |
|-------|----------|
| All layers have high error | Check calibration data quality |
| Debug precision doesn't converge | Increase tolerance or try QAT |
| Performance degradation | Minimize FP32 layers, group them |
| JSON parsing fails | Check file format, use `inspect data` |
| Calibration fails | Verify data loader yields correct format |

## File Locations

- **Main Guide:** `/vercel/sandbox/documents/int8_fp16_accuracy_debugging_guide.md`
- **Examples:** `/vercel/sandbox/tools/Polygraphy/examples/cli/debug/03_comparing_int8_fp16_accuracy/`
- **Scripts:** `parse_layer_errors.py`, `calibration_data_loader.py`, `fix_precision.py`
- **Workflow:** `compare_int8_fp16.sh`

## Key Options

| Option | Description |
|--------|-------------|
| `--trt-outputs mark all` | Compare all layer outputs |
| `--fail-fast` | Stop at first error |
| `--precision-constraints obey` | Force precision constraints |
| `--precision-constraints prefer` | Prefer but allow override |
| `--calibration-cache` | Use/save calibration cache |
| `--data-loader-script` | Custom calibration data |
| `--trt-network-postprocess-script` | Modify network after parsing |
| `--layer-precisions` | Set specific layer precisions |

## Error Metrics

- **Max Absolute Error:** `max(|fp16 - int8|)`
- **Mean Absolute Error:** `mean(|fp16 - int8|)`
- **Max Relative Error:** `max(|fp16 - int8| / |fp16|)`
- **Cosine Similarity:** Measures output similarity

## Typical Thresholds

- **Segmentation:** 0.05 - 0.1 max absolute error
- **Classification:** 0.01 - 0.05 max absolute error
- **Detection:** 0.1 - 0.2 max absolute error

Adjust based on your model and requirements.
