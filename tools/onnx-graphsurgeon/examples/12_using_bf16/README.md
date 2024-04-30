# BFloat16

## Introduction

This example generates a model with bf16 weights.

Numpy currently doesn't support bf16 natively so data values are stored as float32 and the conversion happens prior to onnx export.
```python
tensor = gs.Constant(name="weight", values=np.ones(shape=(5, 3, 3, 3), dtype=np.float32), export_dtype=onnx.TensorProto.BFLOAT16)
# or
tensor = gs.Constant(name="weight", values=np.ones(shape=(5, 3, 3, 3), dtype=np.float32))
tensor.export_dtype = onnx.TensorProto.BFLOAT16

```

## Running the example

1. Generate the model:
    ```bash
    python3 generate.py
    ```

    This creates a model with bfloat16 weights

    ![../resources/12_bf16.onnx.png](../resources/12_bf16.onnx.png)

