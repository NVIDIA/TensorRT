# TensorRT Inference for HuggingFace Transformers 🤗

This repository demonstrates TensorRT inference with models developed using [HuggingFace Transformers](https://huggingface.co/transformers/).

Currently, this repository supports the following models with checkpoints:

1. [GPT2 (text generation task)](https://huggingface.co/transformers/model_doc/gpt2). The sample supports following variants of GPT2 and GPT2-like models:

    - [GPT2](https://huggingface.co/transformers/model_doc/gpt2)
      - gpt2 (117M)
      - gpt2-medium (345M)
      - gpt2-large (774M)
      - gpt2-xl (1558M)
    - [GPT Neo](https://huggingface.co/docs/transformers/model_doc/gpt_neo)
      - EleutherAI/gpt-neo-125m
      - EleutherAI/gpt-neo-1.3B
      - EleutherAI/gpt-neo-2.7B
    - [GPT-Neox](https://huggingface.co/docs/transformers/model_doc/gpt_neox)
      - EleutherAI/gpt-neox-20b
    - [GPT-J](https://huggingface.co/docs/transformers/model_doc/gptj)
      - EleutherAI/gpt-j-6B (6053M)
    - [Cerebras-GPT](https://huggingface.co/cerebras)
      - cerebras/Cerebras-GPT-111M
      - cerebras/Cerebras-GPT-256M
      - cerebras/Cerebras-GPT-590M
      - cerebras/Cerebras-GPT-1.3B
      - cerebras/Cerebras-GPT-2.7B
      - cerebras/Cerebras-GPT-6.7B
      - cerebras/Cerebras-GPT-13B

1. [OPT (text generation task)](https://huggingface.co/docs/transformers/main/en/model_doc/opt). The sample supports following variants of OPT
    - facebook/opt-125m
    - facebook/opt-350m
    - facebook/opt-1.3b
    - facebook/opt-2.7b
    - facebook/opt-6.7b
    - facebook/opt-13b

1. [T5 (translation, premise task)](https://huggingface.co/transformers/model_doc/t5.html). The sample supports following variants of T5:

    - t5-small (60M)
    - t5-base (220M)
    - t5-large (770M)
    - t5-3b
    - t5-11b
    - google/flan-t5-small (60M)
    - google/flan-t5-base (220M)
    - google/flan-t5-large (770M)
    - google/flan-t5-xl (3B)
    - google/flan-t5-xxl (11B)

1. [BART (summarization task)](https://huggingface.co/docs/transformers/model_doc/bart). The sample supports the following variants of BART:

    - facebook/bart-base (139M)
    - facebook/bart-large (406M)
    - facebook/bart-large-cnn (406M)
    - facebook/mbart-large-50 (680M)

1. [BLOOM (text generation task)](https://huggingface.co/docs/transformers/main/en/model_doc/bloom). The sample supports following variants of BLOOM:

    - bigscience/bloom-560m
    - bigscience/bloom-1b1
    - bigscience/bloom-1b7
    - bigscience/bloom-3b
    - bigscience/bloom-7b1
    - bigscience/bloomz-560m
    - bigscience/bloomz-1b1
    - bigscience/bloomz-1b7
    - bigscience/bloomz-3b
    - bigscience/bloomz-7b1

1. [BLIP (image captioning task)](https://huggingface.co/docs/transformers/model_doc/blip). The sample supports following variants of BLIP:

    - Salesforce/blip-image-captioning-base
    - Salesforce/blip-image-captioning-large

## Setup


Follow the setup steps in the TensorRT OSS repository. It is recommended to experiment inside Docker container.
For a smoother setup experience, it is recommended to use [Poetry](https://python-poetry.org/) to install requirements and execute:

```bash
poetry install # one-time setup
poetry add <path_to_trt_wheel> # see top level repo README.md on how to get TensorRT wheels.
poetry run python run.py <args> # execute program
```

However requirements.txt are also provided.

```bash
pip3 install -r requirements.txt # install requirements
python run.py <args> # execute program
```

**Please note that due to end-of-life, Python <= 3.7 is no longer supported.**

## File Structure

```bash
.
├── BART      # BART directory
│   ├── BARTModelConfig.py # Model configuration and variant-specific parameters
│   ├── checkpoint.toml    # Example inputs and baseline outputs
│   ├── export.py          # Model conversions between Torch, TRT, ONNX
│   ├── frameworks.py      # PyTorch inference script
│   ├── onnxrt.py          # OnnxRT inference script
│   ├── trt.py             # TensorRT inference script
├── BLIP      # BLIP directory
│   └── ...
├── BLOOM     # BLOOM directory
│   └── ...
├── GPT2      # GPT2 directory
│   └── ...
├── NNDF      # common high-level abstraction of classes and utilities
├── Seq2Seq   # common concrete abstraction of classes and utilities for Sequence-to-Sequence models
├── T5        # T5 directory
│   └── ...
├── Vision2Seq  # common concrete abstraction of classes and utilities for Vision-to-Sequence models
├── notebooks # Jupyter notebooks
│   ├── playground.ipynb # A new playground for users to run and benchmark selected HF model and configurations
│   ├── GPT2.ipynb    # Jupyter notebooks for GPT2
│   ├── BART.ipynb    # Jupyter notebook for BART
└── run.py    # main entry script
```

## Commands

`run.py` is the main entry point for the demos. We have the following command for different purpose:
1. `accuracy`: Uses LAMBADA dataset to measure TensorRT's accuracy
1. `benchmark`: Uses random inputs to benchmark TensorRT's performance
1. `chat`: Uses user-provided inputs to chat interactively with LLM using TensorRT
1. `run` (Deprecated): Uses hardcoded checkpoint.toml to measure TensorRT's accuracy and performance for common tasks
1. `compare` (Deprecated): Uses `run` mode to compare over different frameworks


### How to run accuracy checks for TensorRT

The `accuracy` action will use LAMBADA dataset to measure the accuracy for each frameworks. It has the following arguments:
`--num-samples`: The number of samples in LAMBADA dataset to use. Default = 20
`--tokens-to-generate`: The number of tokens to generate for TopN accuracy and perplexity calculation. Default = 5. The smaller the number of tokens to generate, the higher the accuracy is (high TopN accuracy and low perplexity).
`--topN`: TopN means if the target is within the top N choice from the output logits, e.g. Top1 represents the absolute accuracy. We will always run Top1 and Top10, but users can pick any other topN. Default = 2.

For example:

```
python3 run.py accuracy GPT2 [frameworks | trt] --variant gpt2-xl --working-dir temp --batch-size 1 --info --fp16 --use-cache --num-samples 100 --tokens-to-generate 3
```

Expected output (The number that you see will be slightly different):

```
AccuracyResult(topN=[TopNAccuracy(n=1, accuracy=0.326), TopNAccuracy(n=10, accuracy=0.720), TopNAccuracy(n=5, accuracy=0.66)], token_perplexity=27.100, seq_perplexity=59.741)
```

Notes:
* Perplexity will be very high for some t5 variants because it tends to have very large perplexity between logits and token if they do not match.
* The `perplexity` field for each sample is actually log_perplexity, but the final reported results is an exponential over the mean of all log_perplexity for each sample.
* We will not calculate `perplexity` for beam search, but we will report TopN accuracy for beam search.

### How to run fixed-length performance test

The `benchmark` action will benchmark the specific script under the model directory using random input data with specified input/output sequence lengths. Note that since the input data is random, the accuracy is not guaranteed, but the benchmarking mode is useful for performance measurement since it allows arbitrary and controllable input/output sequence lengths with early stopping being disabled and allows apples-to-apples performance comparisons across different frameworks. Enabling KV-cache and FP16 mode is recommended for best performance.

```python
python3 run.py benchmark GPT2 [frameworks | trt] --variant [gpt2 | gpt2-medium | etc.] --working-dir temp --input-seq-len 128 --output-seq-len 256 --fp16 --use-cache
```


### How to run the demo interactively using `chat` command

With `chat` command, you can pick a model and feed customized inputs and acts like chatting! Here is an example:
```
python3 run.py chat T5 --compare frameworks trt --variant t5-small --working-dir temp
...
Welcome to TensorRT HuggingFace Demo Chatbox! Please type your prompts. Type 'exit' to quit the chat.
Setting up environment for frameworks
... (Setting up)
Setting up environment for trt
... (Setting up)
Prompt:translate English to German: TensorRT is a great software for machine learning.
frameworks: TensorRT ist eine großartige Software für das Maschinenlernen.. Time: 0.2299s
trt: TensorRT ist eine großartige Software für das Maschinenlernen.. Time: 0.0331s
Prompt:exit
```


### How to run functional and performance tests with checkpoints

The `run` action will run the specific script under the model directory. `--variant` designates the pre-trained HuggingFace model for testing. `--working-dir` saves the downloaded pre-trained models, onnx model files, and TRT engine files. Accuracy of 1.0 indicates correct results in consistency with the expected outputs in `checkpoint.toml`. By default, all running times reported are median numbers of 10 iterations.

```python
python3 run.py run GPT2 [frameworks | onnxrt | trt] --variant [gpt2 | gpt2-medium | etc.] --working-dir temp
```

Expected output:

```properties
NetworkCheckpointResult(network_results=[NetworkResult(
input='TensorRT is a Deep Learning compiler used for deep learning.\n',
output_tensor=tensor([   51, 22854, ....], device='cuda:0'),
semantic_output=['TensorRT is a Deep Learning compiler used for deep learning.\n\nThe main goal of the project is to ....'],
median_runtime=[NetworkRuntime(name='decoder', runtime=0.002254825085401535), NetworkRuntime(name='full', runtime=0.10705459117889404)]),...],
accuracy=1.0,perplexity=7.4631,
models=NetworkModels(torch=None, onnx=None,trt=[NetworkModel(name='gpt2_decoder', fpath='temp/GPT2/GPT2-gpt2-fp16.onnx.engine')]))
```

Notes:
* We will not be maintaining checkpoint.toml files, and will not be responsible for any discrepency between outputs and checkpoint.toml file. Please only report if you get unexpected accuracy from `accuracy` command.

### How to run comparison script

The `compare` action will by default compare all implemented frameworks, i.e. PyTorch framework & TRT & OnnxRT. Note that ONNXRT does not support kv cache.

```python
python3 run.py compare T5 --compare frameworks trt --variant [t5-small | T5-large | etc.] --working-dir temp
```

The above script compares the performance of PyTorch framework inference and TensorRT inference for T5.

| script     | accuracy | encoder (sec) | decoder (sec) | full (sec) |
|------------|----------|---------------|---------------|------------|
| frameworks | 1        | 0.00566595    | 0.00803628    | 0.0244497  |
| trt        | 1        | 0.000525772   | 0.000945443   | 0.00532533 |


## General Flags

### How to customize parameters for time measurement
Use `--iterations`, `--number`, `--warmup`, `--duration`, `--percentile` to control the time measurement process. Most common parameters are explained below:
* `--iterations <int>`: number of iterations to measure (default 10)
* `--warmup <int>`: number of warmup iterations before actual measurement occurs (default 3)
* `--percentile <int>`: key percentile number for measurement (default 50, i.e. median).

```python
python3 run.py run BART [frameworks | trt] --variant facebook/bart-base --working-dir temp --iterations 100 --percentile 99
```

Notes:
* Percentile numbers are representative only if the number of iterations are sufficiently large. Please consider increasing `--iterations` when combined with `--percentile`.
* To avoid conflict with the overall result printing structure, only one percentile number is allowed from command line. If the users need to measure multiple timing statistics from one run (such as p50, p90, p99), please (1) run the command multiple times by changing `--percentile <N>` -- engines will not be re-built from run to run so this is still efficient OR (2) use the [Jupyter notebook demo](./notebooks) for more flexible measurement that can measurement all percentiles in one run.
* This measurement does not work on `accuracy` mode.

### How to run with K-V cache

For all the models, use `--use-cache` option to get the same effect of HuggingFace's `use_cache` option. The old `--enable-kv-cache` flag has been deprecated. For encoder-decoder models, this option will use key & value cache in decoder for uni-directional self-attention and encoder-decoder cross-attention. KV cache could reduce the size of `input_ids` and improve runtime performance when `input_ids` is long. Current benchmarking result shows that at `input_seq_len = 1024` and `output_seq_len = 1024`, t5-large model with kv cache could achieve 3x faster than without kv cache in single NVIDIA A100 GPU. Therefore, it is **always** recommended to enable `--use-cache` flag.

```python
python3 run.py run BART [frameworks | trt] --variant facebook/bart-base --working-dir temp --use-cache
```

Notes:
* A cross_attn_cache_generator will be exported and is required as part of any encoder-decoder models like BART/T5 under this option. This is because cross attention cache is only related to `encoder_hidden_states`. Throughout a single decoding session, cross attention kv cache does not change. For framework PyTorch model, it still output cross attention for each decoding session, which is a memory waste if TRT does the same.


### How to run with beam search

In addition to greedy search, beam search is another widely-used decoding method to achieve better results. For all the models, use `--num-beams <N>` to enable beam search during decoding. Beam search can now be combined with kv cache.

```python
python3 run.py run BART [frameworks | trt] --variant facebook/bart-base --working-dir temp --num-beams 3 [--use-cache]
```

### How to run multi-batch inference with attention_mask

Across all models, please use `--batch-size <B>` to enable multi-batch inference.

```sh
python3 run.py run GPT2 trt --variant gpt2 --batch-size 4 --working-dir temp --fp16 --use-cache --use-mask
```

Users may also want to run multi-batch inference with various seq len, which requires input [`attention_mask`] (https://huggingface.co/docs/transformers/glossary#attention-mask) for masking out pad tokens. For all models, please pass `--use-mask` to enable attention_mask. For example:

```
python3 run.py run GPT2 trt --variant gpt2 --batch-size 4 --working-dir temp --fp16 --use-cache --use-mask
```

## TensorRT-specific Flags

### How to run with different precisions in TensorRT

Frameworks (PyTorch) by default run TF32 on Ampere devices and degrade to FP32 on pre-Ampere devices. Accordingly, in TensorRT run, TF32 is also set as the default precision. To experiment with different precisions, use `--fp16` for FP16. FP16 has longer engine building time, but would speed up decoding.

```python
python3 run.py run BART trt --variant facebook/bart-base --working-dir temp [--fp16]
```

### How to run multi-batch inference with dynamic batch sizes

Users can also pass in `--dynamic-batch` to construct TRT engines with dynamic batch sizes. When this dynamic batch mode is enabled, the additional optional arguments `--min-dynamic-batch <minb>` and `--max-dynamic-batch <maxb>` specify the range of batch sizes supported. An example run with a batch size of 4, min batch size of 1 and max batch size of 8 corresponds to

```sh
python3 run.py run GPT2 trt --variant gpt2 --batch-size 4 --working-dir temp --fp16 --use-cache --use-mask --dynamic-batch --min-dynamic-batch 1 --max-dynamic-batch 8
```

Notes:
* In dynamic batch mode, the constraint `min-dynamic-batch <= batch-size <= max-dynamic-batch` needs to be satisfied, else the program errors out.


### How to customize engine name

The demo defaults to reuse existing engine in the workspace to save space and time, but if you want to test the demo in different platforms without cleaning the engine, the demo might fail. Therefore, for the convenience of running the demo in different platforms, we provide a flag `--engine-postfix`, so users can tag engines. For example:

```
python3 run.py run GPT2 trt --variant gpt2 --working-dir temp --engine-postfix A100-PCIE-80GB
```

### How to run without the TensorRT `FASTER_DYNAMIC_SHAPES_0805` preview feature

`FASTER_DYNAMIC_SHAPES_0805` significantly improves TensorRT engine build time and is enabled by default in TRT 8.6+. Use the `--disable-preview-dynamic-shapes` option to disable this preview feature for any models. In rare cases, the runtime may increase, so we provide an option to disable it:

```python
python3 run.py run BART trt --variant facebook/bart-base --working-dir temp --disable-preview-dynamic-shapes
```

Notes:
* Preview argument is only for TensorRT runs. Hence, please avoid using `compare` action with `--disable-preview-dynamic-shapes` since the syntax doesn't exist for `frameworks` and `onnxrt` runs. Instead, it is recommended to test TensorRT `run` command seperately to obtain the performance without this preview feature.


## Advanced Topics

### How to run TensorRT Engine only

If you already have TRT engines, you can run TRT with the following extra flags:
```
python3 run.py run [GPT2 | T5 | BART | etc.] [frameworks | trt] --variant $variant_name --working-dir temp [--use-cache] [ --num_beams <N>] --decoder-engine $decoder_engine_path [--encoder-engine $encoder_engine_path --cache-generator-engine $cross_attn_cache_generator_path].
```

Notes:
* For encoder/decoder models, `--encoder-engine` is required.
* For encoder/decoder model with kv cache, `--cache-generator-engine` is required.
* You cannot optionally only provide 1 engine. You need to provide all of them, otherwise TRT will attempt to build all engines.
* You need to ensure that your TRT engine has the same names as the auto generated engines, and correctly-set optimization profiles.


### How to run with ONNX

If you already have TRT engines, you can run TRT with the following extra flags:
```
python3 run.py run [GPT2 | T5 | BART | etc.] [frameworks | trt] --variant $variant_name --working-dir temp [--use-cache] [ --num_beams <N>] --decoder-onnx $decoder_engine_path [--encoder-onnx $encoder_engine_path --cache-generator-onnx $cross_attn_cache_generator_path]
```
Same requirements apply for ONNX. You need to ensure that onnx has the same names as the auto generated onnx, and correct dynamic shapes


### How to run new models
**Please note that the demo does not support arbitrary customized model not in HuggingFace format, because the demo uses HuggingFace config to understand the model parameters.**

Currently, we only support a limited number of models with accuracy checks. However, this demo has the potential to run more HuggingFace models without accuracy checkpoints. If you have a HuggingFace model variant similar to the supported ones registered in HuggingFace Hub, you can run:

```
python3 run.py run [GPT2 | T5 | BART | etc.] [frameworks | trt] --variant your_model --working-dir temp [--use-cache] [ --num_beams <N>].
```

If you have a locally saved model, you may use `--torch-dir` to run the demo. Make sure that a HuggingFace-style `config.json` and the correct pytorch model is inside the folder.

```
python3 run.py run [GPT2 | T5 | BART | etc.] [frameworks | trt] --variant your_model --working-dir temp --torch-dir model_loc [--use-cache] [ --num_beams <N>]
```


## Testing

```python
pytest
```

It is recommended to use Pytest `4.6.x`. Your Python environment must have already had the setup completed.


## Troubleshooting

### cuBLAS Errors

```
CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```

It is possible that your LD_LIBRARY_PATH has a competing CUDA version stored inside, causing PyTorch to read the incorrect library.
Consider modifying LD_LIBRARY_PATH and removing your CUDA path.

### Out of Memory Errors

You may sometimes run into the following errors due to insufficient CPU/GPU memory on your machine:

1. GPU OOM for engine building:
```
[E] 4: Could not find any implementation for node {...} due to insufficient workspace. See verbose log for requested sizes.
```

2. GPU OOM for running inference or PyTorch frameworks:
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ... MiB (GPU 0; ... GiB total capacity; ... GiB already allocated; ... MiB free; ... GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
3. CPU OOM: Killed

As a rough but not guaranteed estimate, you should have at least `4*num_parameters` bytes of GPU memory in order to run in `--fp16` mode and at least `8*num_parameters` bytes of GPU memory in order to run in fp32 precision. You should also have at least `12*num_parameters` bytes of CPU memory for model loading and engine building and serialization. For example, for a 6B model, you should have >=24GB GPU memory for `--fp16`, or >=32GB GPU memory for fp32, and >=72GB CPU memory. It is recommended to run `--fp16 --use-cache` to optimize engine build and inference.

Furthermore, we have identified an issue with `torch.onnx.export` with any PyTorch version < 2.1.0 that causes it to increase memory usage by `4*num_parameters`, so in the case of CPU OOM, please ensure you are running with a cached ONNX model. This can be achieved by simply rerunning the exact same command after the ONNX model has been saved. This memory leak bug has been fixed in the latest PyTorch (2.1.0).
