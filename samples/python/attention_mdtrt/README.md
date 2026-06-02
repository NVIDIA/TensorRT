# Multi-Device Attention Inference with TensorRT

This sample demonstrates how to run a self-attention model across multiple GPUs using TensorRT's multi-device inference feature. It covers single-GPU execution as a baseline and multi-GPU execution using MPI for process management and NCCL for GPU-to-GPU communication.

## Introduction

TensorRT supports splitting a single model across multiple GPUs for inference. This is useful when a model is too large to fit on a single GPU, or when you want to reduce latency by parallelizing computation across devices.

TensorRT supports multiple parallelism strategies for multi-device inference, including tensor parallelism (TP) and context parallelism (CP). This sample focuses on **context parallelism (CP)**, where the input sequence is split across GPUs along the sequence dimension. Each GPU processes its portion independently for most operations (linear projections, normalization), but attention requires cross-device communication because every token must attend to every other token. TensorRT handles this communication transparently using collective operations embedded in the model. For a detailed explanation of context parallelism, see [Context Parallelism for Scalable Million-Token Inference](https://arxiv.org/abs/2411.01783).

### How Does it Work?

To run a model on multiple GPUs, the model must be **sharded** (split into pieces that each GPU can execute independently). For context parallelism, sharding means dividing the input sequence across GPUs and inserting communication ops so that each GPU can still compute correct attention over the full sequence.

When a model is sharded for multi-device execution, special **DistCollective** operations are inserted into the ONNX graph:

- **ReduceScatter**: Splits and reduces input data across GPUs. Used at the start to distribute the input sequence.
- **AllGather**: Collects data from all GPUs into a full tensor. Used before attention so each GPU sees the full K/V tensors, and after the output projection to reconstruct the full output.

These ops are not present in the single-device ONNX model. They are inserted automatically by `polygraphy multi-device shard` using sharding hints that describe how the model should be split.

### What is hint.json?

The `hint.json` file tells the polygraphy sharding tool how to partition the model. The sharding tool reads the single-device ONNX graph, identifies the attention layers and I/O tensors specified in the hints, and inserts DistCollective ops at the appropriate points.

```json
{
    "parallelism": "CP",
    "attention_layers": [
        {
            "q": "q_scaled",          // Name of the Q tensor feeding into QK^T matmul
            "gather_kv": true,        // Gather K/V across devices before attention
            "gather_q": false         // Q stays local (each GPU has its own chunk)
        }
    ],
    "dist_collectives": {
        "nb_rank": 2,                 // Number of GPUs to shard across
        "reduce_op": "max"            // Reduction operation for ReduceScatter
    },
    "inputs": [
        {
            "name": "input",          // Input tensor name in the ONNX model
            "seq_len_idx": 0,         // Which dimension is the sequence length
            "rank": 3                 // Number of dimensions
        }
    ],
    "outputs": [
        {
            "name": "output",
            "seq_len_idx": 0,
            "rank": 3
        }
    ]
}
```

For full documentation on the sharding tool and hint format, see the [Polygraphy multi-device documentation](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/polygraphy/tools/multi_device).

## Prerequisites

- A machine with multi-GPU
- `polygraphy` >= 0.49.25 (for `multi-device shard` support)

## Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Generating the ONNX Models

### Step 1: Generate the single-device model

```bash
python3 create_onnx.py --output attention_sd.onnx
```

This creates a self-attention model with:
- Input/Output: `(sequence_length, batch_size, 4096)` in float16
- 32 attention heads, 128-dim per head
- Q/K/V projections, RMSNorm, scaled dot-product attention, output projection

### Step 2: Shard for multi-device

```bash
polygraphy multi-device shard attention_sd.onnx -s hint.json -o attention_md.onnx
```

This inserts DistCollective ops (ReduceScatter, AllGather) into the model based on `hint.json`. The resulting `attention_md.onnx` is designed to run on 2 GPUs.

## Running the Sample

### Single-GPU

```bash
python3 attention_mdtrt.py \
  --onnx-path attention_sd.onnx \
  --sequence-length 56320 \
  --batch-size 1 \
  --num-iterations 50
```

### Multi-GPU (2 GPUs)

```bash
mpirun -np 2 python3 attention_mdtrt.py \
  --onnx-path attention_md.onnx \
  --sequence-length 56320 \
  --batch-size 1 \
  --num-iterations 50
```

### Using a specific libnccl.so

```bash
LD_PRELOAD=/path/to/libnccl.so mpirun -np 2 python3 attention_mdtrt.py \
  --onnx-path attention_md.onnx
```

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

## Changelog

April 2026
Added `create_onnx.py` for ONNX model generation using the GraphSurgeon layer API. Added `--save-output` flag for saving inference output. Updated documentation with DistCollective and sharding explanations.

January 2026
Initial release of this sample.

## Known Issues
None
