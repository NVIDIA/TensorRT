# bertQKVToContextPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

Takes query, key and value tensors and computes scaled multi-head attention - computes scaled dot product attention scores `softmax(K'Q/sqrt(HeadSize))` and returns values weighted by these attention scores.



### Structure

The `bertQKVToContextPlugin` takes two inputs; `input`, and optionally `input_mask`.

`input`
input is a tensor with shape `[S, B, 3 * E, 1, 1]` where `B` is the batch size and `E` is the hidden size. The input has two trailing dimensions in order to generate an output with the same trailing dimensions for a FC optimiazation further down the network.
This plugin makes strong assumptions about its input:
    - The input tensor contains all 3 matrices Q, K, V
    - This input tensor is computed by multiplying a tensor of size `[S, B, E]` with the weights `W_qkv` of size `[E, 3 * E]`
    - The weight matrix W_qkv is NOT just the vertical concatenation of individual matrices `W_tmp = [W_q', W_k', W_v']'`, but to start with `W_tmp`, reshaping it into `[E, 3, N, H]` (where `N * H = E` and `N` is number of heads, `H` is head size) transposing it into `[E, N, 3, H]` and reshaping it back to `[E, 3 * E]`. The interpretation is to layout the k-th heads of Q, K and V next to each other, instead of first all N heads of Q, then all N heads of K, then all heads of V

`input_mask`
input_mask is a tensor of shape `[B]` where `B` is the batch size. The input mask is in the encoded in the format described in `embLayerNormPlugin`, and contains the number of valid elements from the start of the sequence.
If provided, the attention scores, i.e. the softmax distribution, are only computed over the elements designated as valid by the input mask


The `bertQKVToContextPlugin` generates the following output:

`output`
output is a tensor with shape `[S, B, E, 1, 1]` where `B` is the batch size.


## Parameters

`bertQKVToContextPlugin` has plugin creator class `QKVToContextPluginDynamicCreator` and plugin class `CustomQKVToContextPluginDynamic`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               |  Version                          | Description
|----------|-----------------------------------------|-----------------------------------|-------------------------------------------------------------------
|`int`     |`type_id`                                |  1, 2                             |Integer encoding the DataType (0: FP32, 1: FP16, 2: INT8)
|`int`     |`hidden_size`                            |  1, 2, 3                          |The hidden size, denoted by `E` above.
|`int`     |`num_heads`                              |  1, 2, 3                          |The number of self-attention heads.
|`bool`    |`has_mask`                               |  1, 2                             |Whether to use the input_mask input.
|`float`   |`dq_probs`                               |  1, 2, 3                          |inner layer scale factor when run in int8 precision, default 1.f/127.f.
|`int`     |`var_seqlen`                             |  2                                |Whether to use variable sequence length (0: disable, 1: enable), default 0.
|`int`     |`use_int8_scale_max`                     |  2, 3                             |Whether to use INT8 scale factors to optimize softmax MAX reduction. Only active when `type_id==2`. (0: disable, 1: enable), default 1.

## Additional resources

**Networks:**
-   [Transformer](https://arxiv.org/abs/1706.03762)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

August 2022
Expose `use_int8_scale_max` plugin attribute (for versions 2 and 3 of the plugin). This allows users to enable/disable the usage of INT8 scale factors to optimize softmax MAX reduction.
This optimization is applied always when `type_id==2` by default.

September 2021
Add sequence length 512 support in v2 plugin
Add head size 32 support when sequence length is 128, 256 or 512 in v2 plugin

October 2020
Add v2 plugin that supports variable sequence length.
Add v3 plugin that supports int8 interleaved variable sequence length.

November 2019
This is the first release of this `README.md` file.


## Known issues

This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
