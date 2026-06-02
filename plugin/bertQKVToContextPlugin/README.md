# bertQKVToContextPlugin [DEPRECATED]
**This plugin has been deprecated since TensorRT 10.15 and will be removed in a future release. Use `IAttention` (via `INetworkDefinition::addAttention` / `addAttentionV2`) as the out-of-the-box replacement. See [Migration to IAttention](#migration-to-iattention) for the steps to adapt.**

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Migration to IAttention](#migration-to-iattention)
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


## Migration to [IAttention](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/Attention.html)

`IAttention` is the out-of-the-box attention layer in TensorRT and is functionally equivalent to this plugin for the multi-head attention computation, but it is not a drop-in replacement. Adapter steps:

1. **Split the fused QKV input.** This plugin consumes one `[S, B, 3*E, 1, 1]` tensor with the K-th heads of Q, K, V interleaved. `IAttention` takes three separate tensors with shape `[B, N, S, H]`. Reshape the fused QKV to `[S, B, N, 3, H]`, slice along the size-3 axis, then transpose each slice to `[B, N, S, H]`. If the upstream FC weight `W_qkv` was packed using the head-interleaved layout described in the Structure section, no weight repacking is needed. Otherwise, repack `W_qkv` to produce three contiguous `[B, N, S, H]` tensors directly.

   ```cpp
   // [S, B, 3*E, 1, 1] -> [B, N, S, 3, H], then slice axis 3.
   auto* shuf = network->addShuffle(*fusedQKV);
   shuf->setReshapeDimensions(Dims{5, {S, B, N, 3, H}});
   shuf->setSecondTranspose(Permutation{{1, 2, 0, 3, 4}});
   ITensor* t = shuf->getOutput(0);

   auto pick = [&](int32_t i) -> ITensor* {
       auto* sl = network->addSlice(*t, Dims{5, {0,0,0,i,0}},
                                    Dims{5, {B,N,S,1,H}}, Dims{5, {1,1,1,1,1}});
       auto* sq = network->addShuffle(*sl->getOutput(0));
       sq->setReshapeDimensions(Dims{4, {B, N, S, H}});
       return sq->getOutput(0);
   };
   ITensor *q = pick(0), *k = pick(1), *v = pick(2);
   ```

2. **Convert the mask.** This plugin uses the `maskIdx` format (count of valid tokens from the start of each sequence). `IAttention` takes a 4D boolean or additive mask of shape `[B, N, S_q, S_kv]` (broadcastable over `B` and `N`). For non-causal BERT-style self-attention, materialize a boolean mask from the valid-token counts. For causal generation, use `IAttention::setCausalKind(CausalMaskKind)` instead of a mask.

   ```cpp
   // Boolean mask from maskIdx ([B] int32). Broadcasts to [B, NQ, S_q, S_kv].
   auto* idx = network->addFill(Dims{1, {S}}, FillOperation::kLINSPACE, DataType::kINT32);
   idx->setAlpha(0.0); idx->setBeta(1.0);
   auto* less = network->addElementWise(*idx->getOutput(0), *maskIdx,
                                        ElementWiseOperation::kLESS);
   attn->setMask(*less->getOutput(0));

   // For causal autoregressive decode, skip the mask and use:
   // attn->setCausalKind(CausalMaskKind::kLOWER_RIGHT);
   ```

3. **Reshape the output.** This plugin produces `[S, B, E, 1, 1]`. `IAttention` produces `[B, N, S_q, H]`. Transpose and reshape as needed for downstream layers.

   ```cpp
   // [B, N, S_q, H] -> [S, B, E, 1, 1] for downstream layers expecting the plugin layout.
   auto* out = network->addShuffle(*attn->getOutput(0));
   out->setFirstTranspose(Permutation{{2, 0, 1, 3}}); // -> [S, B, N, H]
   out->setReshapeDimensions(Dims{5, {S, B, E, 1, 1}});
   ```

4. **INT8/FP8 quantization.** The plugin folded the entire INT8 attention pipeline into a single op, exposing only `type_id==2` (implicit per-tensor scales) or the explicit-INT8 attribute set. With `IAttention`, you express the QDQ pattern in the network so the builder can fuse it into the FP8/INT8 attention kernel. See also [Explicit Quantization](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html#explicit-quantization).

   ```cpp
   // QDQ Q, K, V (input_qkv_scale).
   auto qdq = [&](ITensor* x) -> ITensor* {
       auto* qn = network->addQuantize(*x, *qkvScale);
       qn->setOutputType(0, DataType::kINT8);
       auto* dq = network->addDequantize(*qn->getOutput(0), *qkvScale);
       return dq->getOutput(0);
   };
   ITensor *qq = qdq(q), *kq = qdq(k), *vq = qdq(v);

   auto* attn = network->addAttentionV2(*qq, *kq, *vq,
                                        AttentionNormalizationOp::kSOFTMAX,
                                        CausalMaskKind::kNONE);
   // dq_probs -> NormalizationQuantizeScale.
   attn->setNormalizationQuantizeScale(*dqProbsScale);

   // QDQ attention output (output_ctx_scale).
   auto* qOut = network->addQuantize(*attn->getOutput(0), *outScale);
   qOut->setOutputType(0, DataType::kINT8);
   auto* dqOut = network->addDequantize(*qOut->getOutput(0), *outScale);
   ```

5. **Variable sequence length.** This plugin accepts a `cu_seqlen` packed layout. `IAttention` does not natively consume packed cu-seqlen input; users relying on this path must either pad to a fixed sequence length or implement their own packing/unpacking around `IAttention`. See below for an example. 

   ```cpp
   // `scatterIdx`, `gatherIdx` are application-specific and are typically built from `cu_seqlen`

   // Packed input: [total_tokens, E]; cu_seqlen: [B+1] int32 prefix sums.
   // Scatter packed -> padded [B, S, E] (pad slots are filled with zeros at a sentinel index).
   auto* padded = network->addGather(*packedInput, *scatterIdx, /*axis=*/0); // [B*S, E]
   auto* padBSE = network->addShuffle(*padded->getOutput(0));
   padBSE->setReshapeDimensions(Dims{3, {B, S, E}});
   // ...split into Q/K/V (step 1), build mask from validLen = cu_seqlen[b+1]-cu_seqlen[b] (step 2),
   //    call addAttentionV2 (step 4 if quantized)...

   // Gather attention output [B, N, S_q, H] -> [B, S, E] -> packed [total_tokens, E].
   auto* outBSE = network->addShuffle(*attn->getOutput(0));
   outBSE->setFirstTranspose(Permutation{{0, 2, 1, 3}});
   outBSE->setReshapeDimensions(Dims{2, {B * S, E}});
   auto* repacked = network->addGather(*outBSE->getOutput(0), *gatherIdx, /*axis=*/0);
   ```


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
April 2026
Provide `IAttention` as the out-of-the-box replacement.

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
