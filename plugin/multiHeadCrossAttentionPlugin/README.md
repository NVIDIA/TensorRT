# multiHeadCrossAttentionPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)


## Description

Takes query, key and value tensors and computes cross multi-head attention.


### Structure

The `multiHeadCrossAttentionPlugin` takes two inputs: `input_q` and `input_kv`.

`input_q`
input_q is a tensor with shape `[B, S_Q, #H, H]` where `B` is the batch size, `S_Q` is the sequence length for query, `#H` is the number of heads and `H` is the head size. 

`input_kv`
input_kv is a tensor with shape `[B, S_KV, #H, 2, H]` where `B` is the batch size, `S_KV` is the sequence length for key and value, `#H` is the number of heads and `H` is the head size. 

The `multiHeadCrossAttentionPlugin` generates the following output:

`output`
output is a tensor with shape `[B, S_Q, #H, H]` where `B` is the batch size, `S_Q` is the sequence length for query, `#H` is the number of heads and `H` is the head size.

## Additional resources

**Networks:**
-   [Transformer](https://arxiv.org/abs/1706.03762)


## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html)
documentation.


## Changelog

November 2022
This is the first release of this `README.md` file.


## Known issues

This plugin only supports GPUs with compute capability 8.0, 8.6 and 8.9. For more information see the [CUDA GPU Compute Capability Support Matrix](https://developer.nvidia.com/cuda-gpus#compute)
