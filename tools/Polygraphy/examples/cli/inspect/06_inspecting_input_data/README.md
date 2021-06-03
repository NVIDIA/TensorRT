# Inspecting Input Data

The `inspect data` subtool can display information about input data generated
by a data loader.

For example, first we'll generate some input data by running inference:

```bash
polygraphy run ../../../models/identity.onnx --onnxrt --save-inputs inputs.pkl
```

Next, we can inspect them:

```bash
polygraphy inspect data inputs.pkl --show-values
```

This will display something like:

```
[I] ==== Input Data (1 iterations) ====

    x [dtype=float32, shape=(1, 1, 2, 2)]
        [[[[-0.16595599  0.44064897]
           [-0.99977124 -0.39533487]]]]
```
