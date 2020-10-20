# Inspecting An ONNX Model

The `inspect model` subtool can display ONNX models.

For example:

```bash
polygraphy inspect model ../../../models/identity.onnx --mode=basic
```

This will display something like:

```
[I] ==== ONNX Model ====
    Name: test_identity | Opset: 8

    ---- 1 Graph Inputs ----
    {x [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 1 Graph Outputs ----
    {y [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 0 Initializers ----
    {}

    ---- 1 Nodes ----
    Node 0    |  [Op: Identity]
        {x [dtype=float32, shape=(1, 1, 2, 2)]}
        -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
```

It is also possible to show detailed layer information, including layer attributes, using `--mode=full`.
