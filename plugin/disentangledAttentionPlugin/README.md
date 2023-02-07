# Disentangled Attention Plugin

- [Disentangled Attention Plugin](#disentangled-attention-plugin)
  - [Description](#description)
  - [Structure](#structure)
    - [Input(s)](#inputs)
    - [Output(s)](#outputs)
    - [Parameters](#parameters)
  - [Additional Resources](#additional-resources)
  - [License](#license)
  - [Changelog](#changelog)
  
## Description
This TensorRT plugin implements an efficient algorithm to perform the calculation of disentangled attention matrices for DeBERTa-variant types of Transformers.

Unlike [BERT](https://arxiv.org/abs/1810.04805) where each word is represented by one vector that sums the content embedding and position embedding, [DeBERTa](https://arxiv.org/abs/2006.03654) design first proposed the concept of disentangled attention, which uses two vectors to encode content and position respectively and forms attention weights by summing disentangled matrices. Performance gap has been identified between the new attention scheme and the original self-attention, mainly due to extra indexing and gather opertaions. Major optimizations implemented in this plugin includes: (i) fusion of gather and pointwise operataions (ii) utilizing the pattern of relative position matrix and shortcuting out-of-boundary index calculation (iii) parallel index calculation. 

This TensorRT plugin is primarily intended to be used together with DeBERTa network (with HuggingFace [DeBERTa](https://huggingface.co/docs/transformers/model_doc/deberta) and [DeBERTa-V2](https://huggingface.co/docs/transformers/model_doc/deberta-v2) implementation), but also applies to generic architectures that adopt disentangeld attention.

## Structure
This plugin works for network with graph node named `DisentangledAttention_TRT`. The corresponding graph modification script can be found under the `demo/DeBERTa` folder of TensorRT OSS.

### Input(s)
This plugin takes three inputs:

* `data0`: Content-to-content ("c2c") Attention Matrix

  > **Input Shape:** `[batch_size*number_heads, sequence_length, sequence_length]`
  > 
  > **Data Type:** `float32` or `float16` or `int8`

  This is the content-to-content attention, Q<sub>c</sub>K<sub>c</sub><sup>T</sup>, which is essentially the BERT self-attention.

* `data1`: Content-to-position ("c2p") Attention Matrix

  > **Input Shape:** `[batch_size*number_heads, sequence_length, relative_distance*2]`
  > 
  > **Data Type:** `float32` or `float16` or `int8`

  This is the content-to-position attention, Q<sub>c</sub>K<sub>r</sub><sup>T</sup>.

* `data2`: Position-to-content ("p2c") Attention Matrix

  > **Input Shape:** `[batch_size*number_heads, sequence_length,  relative_distance*2]`
  > 
  > **Data Type:** `float32` or `float16` or `int8`

   This is the position-to-content attention, K<sub>c</sub>Q<sub>r</sub><sup>T</sup>. Relative distance is the distance span `k` for disentangled attention.

### Output(s)
This plugin generates one output.

* `result`: Disentangled Attention Matrix

  > **Input Shape:** `[batch_size*number_heads, sequence_length, sequence_length]`
  > 
  > **Data Type:** `float32` or `float16` or `int8`

  This is the disentangled attention matrix after applying the scaling factor.

### Parameters
| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`int`   |`span`      | The distance span `k` for relative position. Scalar.
|`float`   |`factor`           | The scaling factor (multiply) to be applied to the attention weights, `1/sqrt(3d)`, where `d` is hidden size per head `H/N`. `H` is hidden size, `N` is number of heads. Scalar (less than 1).

## Additional Resources
- [BERT](https://arxiv.org/abs/1810.04805)
- [DeBERTa](https://arxiv.org/abs/2006.03654)
- [DeBERTa HuggingFace Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/deberta)
- [DeBERTa-V2 HuggingFace Implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/deberta_v2)
  
## License
For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.

## Changelog
- 2022.04: This is the first release of this `README` file.
- 2022.07: Added log bucket for the relative position index calculation (since DeBERTa V2).