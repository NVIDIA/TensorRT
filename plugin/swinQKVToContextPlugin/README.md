# swinQKVToContextPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

Takes query, key, and value tensors and computes scaled and biased multi-head window attention, as given by the formula $ Out=softmax(K^TQ * Scale + RelPosBias)V $.



### Structure

The `swinQKVToContextPlugin` takes three inputs; `input`, `input_mask`, and `rel_pos_bias`, in that order.

`input`
Parameter `input` is a tensor with shape `[B * nW, S, 3 * E, 1, 1]` or `[B * nW, S, 3 * E]` where `B` is the batch size, `nW` is the number of windows, `S` is the sequence length and `E` is the hidden size. With shape `[B * nW, S, 3 * E, 1, 1]`, `input` has two trailing dimensions in order to generate an output with the same trailing dimensions for an FC optimization further down the network. On the other hand, with a shape of `[B * nW, S, 3 * E]`, `input` can support the FCs implemented with MatMult + Add.
This plugin makes strong assumptions about the `input` tensor:
    - The `input` tensor contains all 3 matrices Q, K, V
    - This `input` tensor is computed by multiplying a tensor of size `[B * nW, S, E]` with the weights `W_qkv` of size `[E, 3 * E]`
    - The weight matrix W_qkv is NOT just the vertical concatenation of individual matrices `W_tmp = [W_q', W_k', W_v']'`, but to start with `W_tmp`, reshaping it into `[E, 3, N, H]` (where `N * H = E` and `N` is number of heads, `H` is head size) transposing it into `[E, N, 3, H]` and reshaping it back to `[E, 3 * E]`. The interpretation is to layout the k-th heads of Q, K and V next to each other, instead of first all N heads of Q, then all N heads of K, then all heads of V

`input_mask`
Parameter `input_mask` is a tensor of shape `[nW, Pad_S*Pad_S]` where `nW` is the number of windows. It is transformed from a mask tensor of shape `[nW, S, S]` to fit the plugin's requirements. If the MHA layer does not need a mask, please set the attribute `has_mask` to false, and provide a dummy tensor for `input_mask`. 
If `has_mask` is set to true, the attention scores, i.e. the softmax distribution, are only computed over the elements designated as valid by the input mask.

`rel_pos_bias`
Parameter `rel_pos_bias` is a tensor of shape `[nH, Pad_S*Pad_S]` where `nH` is the number of heads. It is transformed from `relative position bias` tensor of shape `[nH, S, S]` to fit the plugin's requirement. 


The `swinQKVToContextPlugin` generates the following output:

`output`
`output` is a tensor with shape `[B * nW, S, E, 1, 1]` or `[B * nW, S, E]` according to the `input` tensor.

## Parameters

`swinQKVToContextPlugin` has plugin creator class `QKVToContextWithPosBiasPluginCreator` and plugin class `QKVToContextWithPosBiasPlugin`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               |  Version                          | Description
|----------|-----------------------------------------|-----------------------------------|-------------------------------------------------------------------
|`int32_t`     |`type_id`                                |  1                                |Integer encoding the DataType (0: FP32, 1: FP16, 2: INT8). Required.
|`int32_t`     |`hidden_size`                            |  1                                |The hidden size, denoted by `E` above. Required.
|`int32_t`     |`num_heads`                              |  1                                |The number of self-attention heads. Required.
|`int32_t`     |`has_mask`                               |  1                                |Whether to use the `input_mask` input (0: no mask, 1: has mask). Required.
|`float`   |`qkv_scale`                              |  1                                |Scale value for $K^TQ$, default $1.0/\sqrt{HiddenSize/NumHeads}$
|`float`   |`dq_probs`                               |  1                                |Inner layer scale factor when run in INT8 precision, defaults to 1.F/127.F.


## Additional resources

**Networks:**
-   [Transformer](https://arxiv.org/abs/1706.03762)
-   [Swin TransformerSwin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

September 2022
This is the first release of this `README.md` file.


## Known issues

This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
