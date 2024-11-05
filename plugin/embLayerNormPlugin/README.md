# embLayerNormPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

The plugin performs the following two tasks:
1. Embeds an input sequence consisting of token ids and segment ids. This consists of token embedding lookup, segment embedding lookup, adding positional embeddings and finally, layer normalization.

2. For version 1 of the plugin only, preprocesses input masks, that are used to mark valid input tokens in sequences that are padded to the target sequence length.
Assuming contiguous input masks, encodes the masks as a single number denoting the number of valid elements, e.g.:

    ```
    111100 => 4
    110000 => 2
    110100: Invalid mask, because it is not contiguous
    ```
    For subsequent versions (2,3,4,5), the input mask is returned after casting to `half` and reshaping to the shape of the embedded output.


### Structure

The version 1 `embLayerNormPlugin` takes three inputs; `token_id`, `segment_id`, and `input_mask`.
The subsequent versions 2,3,4,5 (variable seqlen) take four inputs; `token_id`, `segment_id`, `cu_seqlen`, and `max_seqlen`.

### Version 1 & 6
Inputs:
- `token_id`
An input sequence containing token ids. token_id is an `int32` tensor with shape `[S, B,]` where `S` is the sequence length and `B` is the batch size.
Tokens typically identify words or word pieces that were obtained by preprocessing the input text.

- `segment_id`
An input sequence containing segment ids. segment_id is an `int32` tensor with shape `[S, B]` where `S` is the sequence length and `B` is the batch size.
The segment id is used to distinguish between different parts of the input sequence that might serve different purposes. E.g. in a squad task, the input sequence might consist of a segment representing the knowledge base (i.e. a paragraph of text) and a segment representing the question.

- `input_mask`
input_mask is an `int32` tensor with shape `[S, B]` where `S` is the sequence length and `B` is the batch size.
The input mask denotes valid elements in a sequence that was padded to the sequence length `S`.

Outputs:

- `embedded_output`
embedded_output is a floating point tensor with shape `[S, B, E]` where `S` is sequence length, `B` is batch size, and `E` is hidden size.
The final output embedding is the sum of embeddings for the token, the segment and the position in the sequence.


- `maskIdx`
The `maskIdx` is a more compact representation of the input mask, consisting of the number of valid elements, assuming that the original mask was contiguous.
For fixed sequence length version 1, the `maskIdx` is an `int32` tensor with shape `[B, packSize]` where `B` is batch size, `packSize` is the packed mask size that depends on the sequence length.

### 6 > Version >= 2

Inputs:
- `token_id`
An input sequence containing token ids. token_id is a 1-D, `int32` tensor with shape `[SxB]` where `S` is the sequence length and `B` is the batch size.
Tokens typically identify words or word pieces that were obtained by preprocessing the input text.

- `segment_id`
An input sequence containing segment ids. segment_id is also a 1-D, `int32` tensor with shape `[SxB]` where `S` is the sequence length and `B` is the batch size.
The segment id is used to distinguish between different parts of the input sequence that might serve different purposes. E.g. in a squad task, the input sequence might consist of a segment representing the knowledge base (i.e. a paragraph of text) and a segment representing the question.

- `input_mask`
input_mask is also a 1-D, `int32` tensor with shape `[SxB]` where `S` is the sequence length and `B` is the batch size.
The input mask denotes valid elements in a sequence that was padded to the sequence length `S`.

- `cu_seqlen` (Version 2,3,4,5 only)
An input sequence containing the "cumulative sequence lengths", used to index into the right sequence when sequences have variable lengths. `cu_seqlen` is a 1-D, `int32` tensor with shape `[B+1]` where `B` is the batch size.

- `max_seqlen` (Version 2,3,4,5 only)
Scalar `int32` value that specifies the maximum sequence length.

Outputs:

- `embedded_output`
embedded_output is a floating point tensor with shape `[SxB, E, 1, 1]` where `S` is sequence length, `B` is batch size, and `E` is hidden size.
The final output embedding is the sum of embeddings for the token, the segment and the position in the sequence.

- `maskIdx`
(1) Huggingface variant (versions 2,4): An empty tensor (for backwards compatibility)
(2) Megatron variant (versions 3,5): The inputs masks returned as a `half` tensor with the same shape as `embedded_output` - `[SxB, E, 1, 1]`.



## Parameters

`embLayerNormPlugin` has plugin creator class `EmbLayerNormPluginDynamicCreator` and plugin class `CustomEmbLayerNormPluginDynamic`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                              |  Version          | Description
|----------|----------------------------------------|-------------------|--------------------------------------------------------
|`int`     |`output_fp16`                           |  1, 2, 3, 4, 5, 6 |Integer encoding the DataType, set 0 when build FP32 network and set 1 when build FP32/INT8 network (0: FP32, 1: FP16)
|`int`     |`full_mask`                             |  1, 6             |Whether to output the full mask that works with the specialized multi-head-attention plugin kernels (this is deprecated, please use mha_type_id)
|`int`     |`mha_type_id`                           |  1, 6             |Integer encoding the multi-head-attention plugin DataType (0: FP32, 1: FP16, 2: INT8)
|`Weights` |`bert_embeddings_layernorm_beta`        |  1, 2, 3, 4, 5, 6 |Beta parameter for layer norm. Shape: `[E,]` where `E` is hidden size
|`Weights` |`bert_embeddings_layernorm_gamma`       |  1, 2, 3, 4, 5, 6 |Gamma parameter for layer norm. Shape: `[E,]` where `E` is hidden size
|`Weights` |`bert_embeddings_word_embeddings`       |  1, 2, 3, 4, 5, 6 |Token embedding matrix. Shape: `[word_vocab_size, E]` where `E` is hidden size
|`Weights` |`bert_embeddings_token_type_embeddings` |  1, 2, 3, 4, 5, 6 |Token type embedding matrix. Shape: `[type_vocab_size, E]` where `E` is hidden size
|`Weights` |`bert_embeddings_position_embeddings`   |  1, 2, 3, 4, 5, 6 |Positional embedding matrix. Shape: `[S, E]` where `S` is the maximum sequence length and `E` is hidden size
Note: version 1, 2, 3 are deprecated and will be removed in a future release; please use their corresponding updated versions: 6, 4, 5 respectively.

## Additional resources

The following resources provide a deeper understanding of the `embLayerNormPlugin` plugin:

**Networks:**
-   [BERT](https://arxiv.org/abs/1810.04805)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

September 2024:
Added `EmblayerNormPlugin` version 6 that mirrors version 1 in IO and attributes (but uses underlying `IPluginV3` implementation instead of the deprecated `IPluginV2DynamicExt` interface)

July 2024:
Add `EmbLayerNormPlugin` versions 3 & 4 that duplicate the behavior of v2 and v3 plugins respectively, but implement the `IPluginV3` interface instead of the deprecated `IPluginV2DynamicExt` interface.
Update this README with updated description of I/O and structure.

October 2020:
Add V2 plugin that supports variable sequence length.

November 2019:
This is the first release of this `README.md` file.


## Known issues

This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
