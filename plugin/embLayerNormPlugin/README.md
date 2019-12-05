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

2. Preprocesses input masks, that are used to mark valid input tokens in sequences that are padded to the target sequence length.
Assuming contiguous input masks, encodes the masks as a single number denoting the number of valid elements, e.g.:

```
111100 => 4
110000 => 2
110100: Invalid mask, because it is not contiguous
```


### Structure

The `embLayerNormPlugin` takes three inputs; `token_id`, `segmend_id`, and `input_mask`.

`token_id`
An input sequence containing token ids. token_id is an `int32` tensor with shape `[S, B]` where `S` is the sequence length and `B` is the batch size.
Tokens typically identify words or word pieces that were obtained by preprocessing the input text.

`segment_id`
An input sequence containing segment ids. segment_id is an `int32` tensor with shape `[S, B]` where `S` is the sequence length and `B` is the batch size.
The segment id is used to distinguish between different parts of the input sequence that might serve different purposes. E.g. in a squad task, the input sequence might consist of a segment representing the knowledge base (i.e. a paragraph of text) and a segment representing the question.

`input_mask`
input_mask is an `int32` tensor with shape `[S, B]` where `S` is the sequence length and `B` is the batch size.
The input mask denotes valid elements in a sequence that was padded to the sequence length `S`.


The `embLayerNormPlugin` generates the following two outputs:

`embedded_input`
embedded_input is an floating point tensor with shape `[S, B, E]` where `S` is sequence length, `B` is batch size, and `E` is hidden size.
The final output embedding is the sum of embeddings for the token, the segment and the position in the sequence.


`maskIdx`
embedded_input is an `int32` tensor with shape `[B,]` where `B` is batch size.
The maskIdx is a more compact representation of the input mask, consisting of the number of valid elements, assuming that the original mask was contiguous.


## Parameters

`embLayerNormPlugin` has plugin creator class `EmbLayerNormPluginDynamicCreator` and plugin class `CustomEmbLayerNormPluginDynamic`.

The parameters are defined below and consists of the following attributes:

| Type     | Parameter                               | Description
|----------|-----------------------------------------|--------------------------------------------------------
|`int`     |`output_fp16`                            |Integer encoding the DataType (0: FP32, 1: FP16)
|`Weights` |`bert_embeddings_layernorm_beta`         |Beta parameter for layer norm. Shape: `[E,]` where `E` is hidden size
|`Weights` |`bert_embeddings_layernorm_gamma`        |Gamma parameter for layer norm. Shape: `[E,]` where `E` is hidden size
|`Weights` |`bert_embeddings_word_embeddings`        |Token embedding matrix. Shape: `[word_vocab_size, E]` where `E` is hidden size
|`Weights` |`bert_embeddings_token_type_embeddings`  |Token type embedding matrix. Shape: `[type_vocab_size, E]` where `E` is hidden size
|`Weights` |`bert_embeddings_position_embeddings`    |Positional embedding matrix. Shape: `[S, E]` where `S` is the maximum sequence length and `E` is hidden size


## Additional resources

The following resources provide a deeper understanding of the `embLayerNormPlugin` plugin:

**Networks:**
-   [BERT](https://arxiv.org/abs/1810.04805)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

November 2019
This is the first release of this `README.md` file.


## Known issues

This plugin only supports GPUs with compute capability >= 7.0. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
