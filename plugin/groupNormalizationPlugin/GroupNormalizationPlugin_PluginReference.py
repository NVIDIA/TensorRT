import numpy as np

def ref(inputs, attributes, version = "1"):
    assert version == "1"
    num_groups = attributes["num_groups"][0]
    epsilon = attributes["eps"][0]
    input = inputs["input"]
    bias = inputs["bias"]
    scale = inputs["scale"]
    output = input.copy()

    assert len(input.shape) == 4
    B, C, H, W = input.shape

    # Groups are a subdivision of the channel dimension.
    assert C % num_groups == 0

    # Normalize every group.
    output = output.reshape((B * num_groups, -1))
    output -= np.mean(output, axis=-1, keepdims=True)
    output /= np.sqrt(epsilon + np.var(output, axis=-1, keepdims=True))

    # Apply per-channel scale and bias.
    output = output.reshape(input.shape)
    output = bias.reshape(1, C, 1, 1) + scale.reshape(1, C, 1, 1) * output

    return {"output": output}
