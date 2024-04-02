# TensorRT FP8 Inference for NeMo models
**Deprecation:** For all users using TensorRT to accelerate Large Language Model inference, please use [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/). TensorRT-LLM includes full model coverage and functionalities of HuggingFace demo and NeMo demo. It also contains more optimizations and functionalities (e.g. model quantization, in-flight batching, etc.), multi-GPU support, better model coverage and much better inference performance. HuggingFace Demo and NeMo demo will not be maintained, and they will be removed from OSS in TRT 10.0 release.

This repository demonstrates TensorRT inference with NeMo Megatron models in FP8/FP16/BF16 precision.

Currently, this repository supports [NeMo GPT](https://huggingface.co/nvidia/nemo-megatron-gpt-5B/tree/fp8) models only.

# Environment Setup
It's recommended to run inside a container to avoid conflicts when installing dependencies. Please check out [`NGC TensorRT`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt/tags) and find a container with TensorRT 9.0 or above. A GPU with compute capability 8.9 or above is required to run the demo with FP8 precision.

```
# Run inside a TensorRT container
sh install.sh [--deps <directory>] [-j <nproc>] [--ninja]
```

All arguments are optional. `--deps` indicates the relative dependency download directory, `-j` indicates number of parallel jobs for building and `--ninja` installs the `ninja` build system which can speed up installation. See `sh install.sh --help` for more details on the arguments.

> The script will install required dependencies and it can take around 30 minutes or more.

**Please note that the [HuggingFace demo directory](demo/HuggingFace) needs to be visible when running this demo, so utility functions can be correctly imported.**

# File Structure
This demo follows simliar structure and command-line interface as in [HuggingFace demo](/demo/HuggingFace).
```
.
├── GPT3                              # GPT3 directory
│   ├── GPT3ModelConfig.py            # model configuration and variant-specific parameters
│   ├── frameworks.py                 # NeMo PyTorch inference script
│   ├── onnxrt.py                     # OnnxRT inference script
│   ├── trt.py                        # TensorRT inference script
│   ├── decoding.py                   # main inference logic for all runtimes
│   └── ...                           # files with utility functions for model export and inference
├── config.yaml                       # full configuration for model export and inference
├── interface.py                      # definitions of setup functions
├── nemo_export.py                    # export functions for NeMo model -> ONNX model -> TRT engine
└── run.py                            # main entry script
```

# Overview

This demo contains two scripts `run.py` and `nemo_export.py`. Script `run.py` accepts a NeMo model or an ONNX model as input, and performs end-to-end inference with various actions specified by the user. Script `nemo_export.py` accepts a NeMo model or an ONNX model as input, and exports the input to an ONNX model or a TensorRT engine.

# How to run inference
The `run` action will run end-to-end inference on sentences specified in [config.yaml](/demo/NeMo/config.yaml). A model, a variant, and precision are required to run this command.
```
python3 run.py run GPT3 <frameworks|trt> --variant gpt-5b --working-dir $(pwd)/temp --fp8 --bf16 --nemo-model=<model_fp8_bf16.nemo>
```

Expected output for the second sentence:
```
Batch 1: {'sentences': ['TensorRT is a Deep Learning compiler used for deep learning. It is a compiler for TensorFlow, CNTK, and Torch. It is a compiler for the TensorFlow, CNTK,'],
          'tokens': [['<|endoftext|>', 'T', 'ensor', 'RT', ' is', ' a', ' Deep', ' Learning', ' compiler', ' used', ' for', ' deep', ' learning', '.', ' It', ' is', ' a', ' compiler', ' for', ' T', 'ensor', 'Flow', ',', ' C', 'NT', 'K', ',', ' and', ' Torch', '.', ' It', ' is', ' a', ' compiler', ' for', ' the', ' T', 'ensor', 'Flow', ',', ' C', 'NT', 'K', ',']],
          'logprob': tensor([[-4.6415e+00, -6.9270e+00, -7.4458e+00, -1.9856e+00, -5.9787e-01,
                              -8.1058e+00, -7.9629e-02, -5.8013e+00, -5.5222e+00, -1.4401e+00,
                              -5.5644e+00, -3.3747e-01, -3.3463e+00, -1.1306e+00, -1.3685e+00,
                              -1.7793e+00, -2.8960e+00, -1.4127e+00, -2.3209e+00, -7.3454e-04,
                              -9.8682e-02, -1.3268e+00, -2.1373e+00, -3.9281e-01, -6.5222e-04,
                              -2.9425e-01, -1.4167e+00, -1.8416e+00, -9.2462e-01, -1.4805e+00,
                              -1.4299e+00, -2.0632e+00, -2.9947e+00, -9.1487e-01, -2.6651e+00,
                              -2.2772e+00, -4.7057e-03, -2.2852e-01, -2.4777e+00, -2.4731e-01,
                              -7.0602e-03, -4.7339e-04, -1.1645e-01]], device='cuda:0'),
         'full_logprob': None,
         'token_ids': [[50256, 51, 22854, 14181, 318, 257, 10766, 18252, 17050, 973, 329, 2769, 4673, 13, 632, 318, 257, 17050, 329, 309, 22854, 37535, 11, 327, 11251, 42, 11, 290, 34868, 13, 632, 318, 257, 17050, 329, 262, 309, 22854, 37535, 11, 327, 11251, 42, 11]],
         'offsets': [[0, 0, 1, 6, 8, 11, 13, 18, 27, 36, 41, 45, 50, 59, 60, 63, 66, 68, 77, 81, 83, 88, 92, 93, 95, 97, 98, 99, 103, 109, 110, 113, 116, 118, 127, 131, 135, 137, 142, 146, 147, 149, 151, 152]]}
```

# How to run with various configurations
- FP8, FP16, and BF16 precisions are supported, and they can be set through `--fp8`, `--fp16`, and `--bf16` respectively. Currently, the script has constraints on how precisions are specified, and supported combinations are:
  1. Pure FP16: `--fp16` (default)
  2. Pure BF16: `--bf16`
  3. FP8-FP16: `--fp8 --fp16`
  4. FP8-BF16: `--fp8 --bf16`

- `--nemo-model=<model.nemo>` or `--nemo-checkpoint=<model.ckpt>` can be used to load a NeMo model or checkpoint from a specified path, respectively. If these arguments are not provided, a NeMo model will be downloaded (and cached/re-used for subsequent runs) in the working directory.

- K-V cache can be enabled through `--use-cache`

- Batch size can be changed through `--batch-size=<bs>`

- Default max sequence length is `256`, can be changed through `--max-seq-len=<ms>`

# How to run performance benchmark
The `benchmark` action will run inference with specified input and output sequence lengths multiple times.
```
python3 run.py benchmark GPT3 <frameworks|trt> --variant gpt-5b --working-dir $(pwd)/temp --fp8 --bf16 --nemo-model=<model_fp8_bf16.nemo> --batch-size=16 --input-seq-len=128 --output-seq-len=20 --use-cache --warmup=10 --iterations=100
```

Expected output for `trt`:
```
***************************
Running 100 iterations with batch size: 16, input sequence length: 128 and output sequence length: 20
[E2E inference] Total Time: 11.55453 s, Average Time: 0.11555 s, 95th Percentile Time: 0.11581 s, 99th Percentile Time: 0.11587 s, Throughput: 2769.48 tokens/s
[Without tokenizer] Total Time: 10.44539 s, Average Time: 0.10445 s, 95th Percentile Time: 0.10459 s, 99th Percentile Time: 0.10465 s, Throughput: 3063.55 tokens/s
***************************
```

Expected output for `frameworks`:
```
***************************
Running 100 iterations with batch size: 16, input sequence length: 128 and output sequence length: 20
[E2E inference] Total Time: 55.23503 s, Average Time: 0.55235 s, 95th Percentile Time: 0.55525 s, 99th Percentile Time: 0.56992 s, Throughput: 579.34 tokens/s
[Without tokenizer] Total Time: 54.06591 s, Average Time: 0.54066 s, 95th Percentile Time: 0.54369 s, 99th Percentile Time: 0.55839 s, Throughput: 591.87 tokens/s
***************************
```

# How to run accuracy check
The `accuracy` action will run accuracy check on a dataset. Default is to use [LAMBADA](https://paperswithcode.com/dataset/lambada) dataset.
```
python3 run.py accuracy GPT3 <frameworks|trt> --variant gpt-5b --working-dir $(pwd)/temp --fp8 --bf16 --nemo-model=<model_fp8_bf16.nemo> --use-cache
```

Expected output for `trt`:
```
***************************
Lambada ppl(last token): 4.4756, ppl(sequence): 18.3254, acc(top1): 0.6722, acc(top3): 0.8597, acc(top5): 0.9076
***************************
```

Expected output for `frameworks`:
```
***************************
Lambada ppl(last token): 4.4669, ppl(sequence): 18.3161, acc(top1): 0.6765, acc(top3): 0.8612, acc(top5): 0.9082
***************************
```

# How to export a NeMo model to ONNX
NeMo to ONNX conversion consists of 3 steps:
1. Export ONNX from NeMo.
2. NeMo uses TransformerEngine to export FP8 models to ONNX (step 1) and the exported ONNX has custom TensorRT Q/DQ nodes. Script `convert_te_onnx_to_trt_onnx.py` can be used to convert the custom operators into standard opset19 ONNX Q/DQ nodes.
3. Add KV-cache inputs and outputs to the exported ONNX, so it is faster when performing inference on the model.

`nemo_export.py` has `--opset19` and `--use-cache` option to decide whether to perform step 2. and step 3., respectively:
```
python3 nemo_export.py --nemo-model=model.nemo --onnx=onnx/model.onnx --opset19 --use-cache
```
`--extra-configs` can be used to specified configs that are defined in `config.yml` but not being exposed from existing command-line interface.
Please specify `--help` to see more options.


# How to run sparsity for benchmark

*Note: this is for performance analysis. The pruned model should not be used for accuracy purpose unless it was fine-tuned for sparsity. The pruning may take minutes or hours depending on the model size.*


1. Enable sparsity knobs in `config.yaml`:
  * Set `onnx_export_options.prune` to `True` to enable pruning of the ONNX model.
  * Set `trt_export_options.sparse` to `True` to enable sparse tactics profiling in TensorRT.
2. Run the scripts. You should be able to see logs like below.

```
[2023-07-28 00:15:03,015][OSS][INFO] Prune ONNX model with: polygraphy surgeon prune ${OSS_ROOT}/demo/NeMo/temp/gpt-5b/GPT3-gpt-5b-fp8-fp16-ms256/onnx/model-16.opset19.onnx -o ${OSS_ROOT}/demo/NeMo/temp/gpt-5b/GPT3-gpt-5b-fp8-fp16-ms256/onnx/pruned.model-16.opset19.onnx --save-external-data ${OSS_ROOT}/demo/NeMo/temp/gpt-5b/GPT3-gpt-5b-fp8-fp16-ms256/onnx/pruned.model-16.opset19.onnx_data
[2023-07-28 00:15:03,016][OSS][INFO] This may take a while...
...

[2023-07-28 03:36:52,307][OSS][DEBUG] trtexec --onnx=${OSS_ROOT}/demo/NeMo/temp/gpt-5b/GPT3-gpt-5b-fp8-fp16-ms256/onnx/pruned.model-16.opset19.onnx --minShapes=input_ids:1x1,position_ids:1x1 --optShapes=input_ids:1x128,position_ids:1x128 --maxShapes=input_ids:1x256,position_ids:1x256 --fp8 --fp16 --sparsity=enable --timingCacheFile=functional.cache
```
