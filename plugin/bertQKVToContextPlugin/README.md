# bertQKVToContextPlugin [DEPRECATED]
**This plugin has been deprecated since TensorRT 10.15 and will be removed in a future release. No alternatives are planned to be provided.**

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
The input mask is encoded in the `maskIdx` format described in `embLayerNormPlugin`, and contains the number of valid elements from the start of the sequence.
If provided, the attention scores, i.e. the softmax distribution, are only computed over the elements designated as valid by the input mask


The `bertQKVToContextPlugin` generates the following output:

`output`
output is a tensor with shape `[S, B, E, 1, 1]` where `B` is the batch size.


## Parameters

`bertQKVToContextPlugin` has plugin creator class `QKVToContextPluginDynamicCreator` and plugin class `CustomQKVToContextPluginDynamic`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               |  Version                          | Description
|----------|-----------------------------------------|-----------------------------------|-------------------------------------------------------------------
|`int`     |`type_id`                                |  1, 2, 4, 5                       |Integer encoding the DataType (0: FP32, 1: FP16, 2: INT8)
|`int`     |`hidden_size`                            |  1, 2, 3, 4, 5, 6                 |The hidden size, denoted by `E` above.
|`int`     |`num_heads`                              |  1, 2, 3, 4, 5, 6                 |The number of self-attention heads.
|`bool`    |`has_mask`                               |  1, 2, 4, 5                       |Whether to use the input_mask input.
|`float`   |`dq_probs`                               |  1, 2, 3, 4, 5, 6                 |inner layer scale factor when run in int8 precision, default 1.f/127.f.
|`int`     |`var_seqlen`                             |  2, 5                             |Whether to use variable sequence length (0: disable, 1: enable), default 0.
|`int`     |`use_int8_scale_max`                     |  2, 3, 5, 6                       |Whether to use INT8 scale factors to optimize softmax MAX reduction. Only active when `type_id==2`. (0: disable, 1: enable), default 1.
|`int`     |`use_explicit_int8`                      |  3, 6                             |Whether to use explicit INT8, (0: disable, 1: enable), default 0.
|`float`   |`input_qkv_scale`                        |  3, 6                             |The int8 scale for the input qkv tensor when explicit precision is used, default 1.f.
|`float`   |`output_ctx_scale`                       |  3, 6                             |The int8 scale for the output context tensor when explicit precision is used, default 1.f.


## Additional resources

**Networks:**
-   [Transformer](https://arxiv.org/abs/1706.03762)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog
Oct 2025
Deprecated this plugin. No alternatives are planned to be provided.

Jan 2025
Added precompiled kernel cubins for gb100 (compute capability 10.0) and gb20x (compute capability 12.0) platforms.

Aug 2024
Ported v3 variant (IPluginV2 design) to v6 (IpluginV3 design), but preserves identical attributes and I/O to v3.
Ported v2 variant (IPluginV2 design) to v5 (IpluginV3 design), but preserves identical attributes and I/O to v2.
Ported v1 variant (IPluginV2 design) to v4 (IpluginV3 design), but preserves identical attributes and I/O to v1.

Feb 2024
The issue of the V2 plugin not supporting head sizes of 32 or less and variable sequences of 64, 96, and 384 has been resolved.

Oct 2023
Support explicit int8.

April 2023
Optimize the GPU memory usage by using the cublas handle from attachToContext.

October 2022
Add IGMMA/HGMMA sm90 fmha_v2 kernels

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
