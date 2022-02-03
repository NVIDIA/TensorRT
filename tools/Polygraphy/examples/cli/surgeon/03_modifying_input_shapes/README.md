# Modifying Input Shapes


## Introduction

The `surgeon sanitize` subtool can be used to modify the input shapes of an ONNX model.
This does not change the intermediate layers of the model, and as such, may cause issues if
the model makes assumptions about the input shapes (for example, a `Reshape` node with a hard-coded
new shape).

Output shapes can be inferred and so these are not modified (nor do they need to be).

*NOTE: Re-exporting the ONNX model with the desired shapes is strongly recommended.*
    *The method shown here should only be used when doing so is not possible.*

## Running The Example

1. Change the input shape of the model to a shape with a dynamic batch dimension,
    keeping other dimensions the same:

    ```bash
    polygraphy surgeon sanitize identity.onnx \
        --override-input-shapes x:['batch',1,2,2] \
        -o dynamic_identity.onnx
    ```

2. **[Optional]** You can use `inspect model` to confirm whether it looks correct:

    ```bash
    polygraphy inspect model dynamic_identity.onnx --show layers
    ```
