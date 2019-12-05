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
input is a tensor with shape `[S, B, 3 * E]` where `B` is the batch size and `E` is the hidden size.
This plugin makes strong assumptions about its input:
    - The input tensor contains all 3 matrices Q, K, V
    - This input tensor is computed by multiplying a tensor of size `[S, B, E]` with the weights `W_qkv` of size `[E, 3 * E]`
    - The weight matrix W_qkv is NOT just the vertical concatenation of individual matrices `W_tmp = [W_q', W_k', W_v']'`, but to start with `W_tmp`, reshaping it into `[E, 3, N, H]` (where `N * H = E` and `N` is number of heads, `H` is head size) transposing it into `[E, N, 3, H]` and reshaping it back to `[E, 3 * E]`. The interpretation is to layout the k-th heads of Q, K and V next to each other, instead of first all N heads of Q, then all N heads of K, then all heads of V

`input_mask`
input_mask is a tensor of shape `[B]` where `B` is the batch size. The input mask is in the encoded in the format described in `embLayerNormPlugin`, and contains the number of valid elements from the start of the sequence.
If provided, the attention scores, i.e. the softmax distribution, are only computed over the elements designated as valid by the input mask


The `bertQKVToContextPlugin` generates the following output:

`output`
output is a tensor with shape `[S, B, E]` where `B` is the batch size.


## Parameters

`bertQKVToContextPlugin` has plugin creator class `QKVToContextPluginDynamicCreator` and plugin class `CustomQKVToContextPluginDynamic`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               | Description
|----------|-----------------------------------------|-------------------------------------------------------------------
|`int`     |`type_id`                                |Integer encoding the DataType (0: FP32, 1: FP16)
|`int`     |`hidden_size`                            |The hidden size, denoted by `E` above.
|`int`     |`num_heads`                              |The number of self-attention heads.
|`bool`    |`has_mask`                               |Whether to use the input_mask input.


## Additional resources

**Networks:**
-   [Transformer](https://arxiv.org/abs/1706.03762)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

November 2019
This is the first release of this `README.md` file.


## Known issues

This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
