# Multi-Device

## Table of Contents

- [Subtools](#subtools)
- [Usage](#usage)
- [Examples](#examples)


## Subtools

### Shard

The `shard` tool in Polygraphy is used to convert a single-device (SD) model to a multi-device (MD) model using a sharding hints file. 

#### Sharding Hints File Format

The hints file is a JSON file that describes how to shard the model. Example:

```json
{
    "parallelism": "CP",
    "attention_layers": [
        {
            "q": "q",
            "gather_kv": true,
            "gather_q": false,
            "polygraphy_class": "AttentionLayerHint"
        }
    ],
    "dist_collectives": {
        "group_size": 0,
        "root": -1,
        "nb_rank": 2,
        "reduce_op": "max",
        "groups": [],
        "polygraphy_class": "DistCollective"
    },
    "inputs": [
        {
            "name": "input",
            "seq_len_idx": 0,
            "rank": 3,
            "polygraphy_class": "ShardTensor"
        }
    ],
    "outputs": [
        {
            "name": "output",
            "seq_len_idx": 0,
            "rank": 3,
            "polygraphy_class": "ShardTensor"
        }
    ],
    "k_seq_len_idx": 0,
    "v_seq_len_idx": 0,
    "kv_rank": null,
    "polygraphy_class": "ShardHints"
}
```

- `parallelism`: Type of parallelism: `CP` (context parallelism) or `TP` (tensor parallelism)
- `attention_layers`: List of attention layer configs:
  - `q`: Name of the Q tensor
  - `gather_kv`: Whether to all-gather K/V
  - `gather_q`: Whether to all-gather Q
  - `replace`: (Optional) What (if any) replacement should be performed on this attention layer
- `dist_collectives`: Configuration of DistCollective ops.
  - `group_size`: Number of GPUs model will be run on. 0 indicates all available GPUs will run.
  - `root`: Root rank for collectives.
  - `nb_rank`: Number of ranks for DistCollective op.
  - `reduce_op`: Reduction operator to be used on reduce-scatter nodes.
  - `groups`: Indices of NCCL groups in which collective operations will run. A value of [] indicates collective operations will run on all ranks with no grouping.
- `inputs`: List of tensors that should be reduce-scattered.
  - `name`: Name of tensor.
  - `seq_len_idx`: Index of dimension that represents sequence length for this input tensor. A non-zero index will cause transpose tensors to be inserted before and after the DistCollective node to transpose the tensor to have sequence length be the first dimension.
  - `rank`: Rank of this tensor. Used as a fallback if no dimension can be obtained from the model and `seq_len_idx` != 0.
- `outputs`: List of tensors that should be all-gathered.
  - `name`: Name of tensor.
  - `seq_len_idx`: Index of dimension that represents sequence length for this output tensor. A non-zero index will cause transpose tensors to be inserted before and after the DistCollective node to transpose the tensor to have sequence length be the first dimension.
  - `rank`: Rank of this tensor. Used as a fallback if no dimension can be obtained from the model and `seq_len_idx` != 0.
- `k_seq_len_idx`: Index of dimension that represents sequence length for all K tensor(s). A non-zero index will cause transpose tensors to be inserted before and after the DistCollective node to transpose the tensor to have sequence length be the first dimension.
- `v_seq_len_idx`: Index of dimension that represents sequence length for all V tensor(s). A non-zero index will cause transpose tensors to be inserted before and after the DistCollective node to transpose the tensor to have sequence length be the first dimension.
- `kv_rank`: Rank all K and V tensor(s). Used as a fallback if no dimension can be obtained from the model and `k_seq_len_idx` or `v_seq_len_idx` != 0.

The `shard` tool also supports one type of replacement for attention: Fused. It can be selected by using `--cp-type fused` with `polygraphy template shard-hints`.

The options for Fused are:

- `is_causal`: Whether or not the attention is causal
- `q_shuffle`: Transpose permutation to apply to Q tensor before attention
- `k_shuffle`: Transpose permutation to apply to K tensor before attention
- `v_shuffle`: Transpose permutation to apply to V tensor before attention
- `nb_rank`: Number of ranks for Attention op.


## Usage

See `polygraphy multi-device -h` for usage information.


## Examples

For examples, see [this directory](../../../examples/cli/multi_device/)
