# TensorRT Inference for HuggingFace Transformers 🤗

This repository demonstrates TensorRT inference with models developed using [HuggingFace Transformers](https://huggingface.co/transformers/).

Currently, this repository supports the following models:

1. [GPT2 (text generation task)](https://huggingface.co/transformers/model_doc/gpt2.html). The sample supports following variants of GPT2:

    gpt2 (117M), gpt2-medium (345M), gpt2-large (774M), gpt2-xl (1558M), EleutherAI/gpt-j-6B (6053M)

2. [T5 (translation, premise task)](https://huggingface.co/transformers/model_doc/t5.html). The sample supports following variants of T5:

    t5-small (60M), t5-base (220M), t5-large (770M), t5-3b(3B), t5-11b(11B)

3. [BART (summarization task)](https://huggingface.co/docs/transformers/model_doc/bart.html). The sample supports the following variants of BART:

    facebook/bart-base (139M), facebook/bart-large (406M), facebook/bart-large-cnn (406M), facebook/mbart-large-50 (680M)

## Setup

Follow the setup steps in the TensorRT OSS repository. It is recommended to experiment inside Docker container.
For a smoother setup experience, it is recommended to use [Poetry](https://python-poetry.org/) to install requirements and execute:

```bash
poetry install # one-time setup
poetry run python run.py <args> # execute program
```

However requirements.txt are also provided.

```bash
pip3 install -r requirements.txt # install requirements
python run.py <args> # execute program
```

**Please note that due to end-of-life, Python <= 3.6 is no longer supported.**

## File Structure

```bash
.
├── GPT2      # GPT2 directory
│   └── ...
├── T5        # T5 directory
│   └── ...
├── BART      # BART directory
│   ├── BartModelConfig.py # Model configuration and variant-specific parameters
│   ├── checkpoint.toml    # Example inputs and baseline outputs
│   ├── export.py          # Model conversions between Torch, TRT, ONNX
│   ├── frameworks.py      # PyTorch inference script
│   ├── onnxrt.py          # OnnxRT inference script
│   ├── trt.py             # TensorRT inference script
│   ├── hf.py              # HuggingFace inference script
│   └── measurements.py    # Performance measurement script
├── NNDF      # common high-level abstraction of classes and utilities
├── notebooks # Jupyter notebooks for GPT2 and T5
└── run.py    # main entry script
```

## How to run comparison script

`run.py` is the main entry point for the demos. `compare` and `run` are two most common actions to use with `run.py`.

The `compare` action will by default compare all implemented frameworks, e.g., PyTorch frameworks & TRT (for GPT2), PyTorch framework & TRT & OnnxRT (for T5 and BART).

```python
python3 run.py compare GPT2 --variant [gpt2 | gpt2-medium | gpt2-large | gpt2-xl | EleutherAI/gpt-j-6B] --working-dir temp
```

The above script compares the performance of PyTorch framework inference and TensorRT inference for GPT2:

| script     | accuracy | decoder (sec) | encoder (sec) | full (sec) |
|------------|----------|---------------|---------------|------------|
| frameworks | 1        | 0.0292865     | 0.0174382     | 0.122532   |
| trt        | 1        | 0.00494083    | 0.0068982     | 0.0239782  |

Notes: `--variant` designates the pre-trained model for testing. `--working-dir` saves the downloaded pre-trained models, onnx model files, and TRT engine files. accuracy of 1.0 indicates correct results in consistency with the expected outputs in `checkpoint.toml`. By default, all running times reported are median numbers of 10 iterations.

## How to run functional and performance benchmark

The `run` action will run the specific script under the model directory.

```python
python3 run.py run GPT2 [frameworks | trt] --variant [gpt2 | gpt2-medium | gpt2-large | gpt2-xl | EleutherAI/gpt-j-6B] --working-dir temp
```

Expected output:

```properties
NetworkCheckpointResult(network_results=[NetworkResult(
input='TensorRT is a Deep Learning compiler used for deep learning.\n',
output_tensor=tensor([   51, 22854, ....], device='cuda:0'),
semantic_output=['TensorRT is a Deep Learning compiler used for deep learning.\n\nThe main goal of the project is to create a tool that can be used to train deep learning algorithms.\n\n'],
median_runtime=[NetworkRuntime(name='gpt2_decoder', runtime=0.002254825085401535), NetworkRuntime(name='full', runtime=0.10705459117889404)],
models=NetworkModels(torch=None, onnx=[NetworkModel(name='gpt2_decoder', fpath='temp/GPT2/GPT2-gpt2-fp16.onnx')],
trt=[NetworkModel(name='gpt2_decoder', fpath='temp/GPT2/GPT2-gpt2-fp16.onnx.engine')]))], accuracy=1.0)
```

## How to run with different precisions in TensorRT

Frameworks (PyTorch) by default run TF32 on Ampere devices and degrade to FP32 on pre-Ampere devices. Accordingly, in TensorRT run, TF32 is also set as the default precision. To experiment with different precisions, use `--fp16` for FP16:

```python
python3 run.py run BART trt --variant facebook/bart-base --working-dir temp [--fp16]
```

## How to customize parameters for time measurement
Use `--iterations`, `--number`, `--warmup`, `--duration`, `--percentile` to control the time measurement process. Most common parameters are explained below:
* `--iterations <int>`: number of iterations to measure (default 10)
* `--warmup <int>`: number of warmup iterations before actual measurement occurs (default 3)
* `--percentile <int>`: key percentile number for measurement (default 50, i.e. median).

```python
python3 run.py run BART trt --variant facebook/bart-base --working-dir temp --iterations 100 --percentile 99
```

Notes:
* Percentile numbers are representative only if the number of iterations are sufficiently large. Please consider increasing `--iterations` when combined with `--percentile`.
* To avoid conflict with the overall result printing structure, only one percentile number is allowed from command line. If the users need to measure multiple timing statistics from one run (such as p50, p90, p99), please (1) run the command multiple times by changing `--percentile <N>` -- engines will not be re-built from run to run so this is still efficient OR (2) use the [Jupyter notebook demo](./notebooks) for more flexible measurement that can measurement all percentiles in one run.

## How to run with K-V cache

For all the models (GPT2/BART/T5), use `--enable-kv-cache` option to get the same effect of HuggingFace's `use_cache` option. For encoder-decoder models, this option will use key & value cache in decoder for uni-directional self-attention and encoder-decoder cross-attention. KV cache could reduce the size of `input_ids` and improve runtime performance when `input_ids` is long. Current benchmarking result shows that at `input_seq_len = 1024` and `output_seq_len = 1024`, t5-large model with kv cache could achieve 3x faster than without kv cache in single NVIDIA A100 GPU. 

```python
python3 run.py run BART [frameworks | trt] --variant facebook/bart-base --working-dir temp --enable-kv-cache
```

Notes:
* For BART and GPT2, K-V cache decoder with TensorRT requires exporting 2 onnx files and building separate engines respectively, called "non-kv" and "kv". For the first decoder run, KV Cache needs to be generated with only `input_ids` and `encoder_hidden_states`(if encoder_decoder), which is named "non-kv". For the other decoder iterations, previous KV Cache and other inputs are passed into the model to generate the updated KV Cache and decoder_hidden_states, which is named "kv". Because current onnx export cannot handle dynamic number of inputs, 2 onnx files with slightly different configurations are used together.
* For T5, the code has been optimized according to the latest TensorRT features. (1) Cross attention kv does not change throughout decoding session, so it is only calculated once at the first decoding session. `onnx.export` cannot handle this logic properly for HuggingFace, so we creates a "cross attention kv generator" using only `encoder_hidden_states`. (2) TensorRT's "zero tensor" feature is used for self attention kv cache growth starting at empty. (3) Self attention input and output are the same location to avoid D2D copy for kv cache. A similar optimization will be ported to all the models.


## How to run with beam search

In addition to greedy search, beam search is another widely used decoding method. For all the models, use `--num-beams <N>` to enable beam search during decoding.

```python
python3 run.py run BART [frameworks | trt] --variant facebook/bart-base --working-dir temp --num-beams 3
```

Notes:
* K-V cache with beam search may have memory concurrency issues with TensorRT Optimization. We are currently working on this issue.


## How to run with TensorRT `FASTER_DYNAMIC_SHAPES_0805` preview feature

Use the `--preview-dynamic-shapes` option to enable this preview feature for BART, GPT2, or T5. This feature will significantly improve TensorRT engine build time.

```python
python3 run.py run BART trt --variant facebook/bart-base --working-dir temp --preview-dynamic-shapes
```

Notes: 
* preview feature functionality is only supported in TensorRT 8.5+.
* preview argument is only for TensorRT runs. Hence, please avoid using `compare` action with `--preview-dynamic-shapes` since the syntax doesn't exist for `frameworks` and `onnxrt` runs. Instead, it is recommended to test TensorRT `run` command seperately to get the performance with preview feature functionality.

## How to run in performance benchmarking mode

The `benchmark` action will benchmark the specific script under the model directory using random input data with specified input/output sequence lengths. Note that since the input data is random, the accuracy is not guaranteed, but the benchmarking mode is useful for performance measurement since it allows arbitrary and controllable input/output sequence lengths with early stopping being disabled and allows apples-to-apples performance comparisons across different frameworks.

```python
python3 run.py benchmark GPT2 [frameworks | trt] --variant [gpt2 | gpt2-medium | gpt2-large | gpt2-xl | EleutherAI/gpt-j-6B] --working-dir temp --input-seq-len 128 --output-seq-len 256
```

## How to run in performance benchmarking mode

The `benchmark` action will benchmark the specific script under the model directory using random input data with specified input/output sequence lengths. Note that since the input data is random, the accuracy is not guaranteed, but the benchmarking mode is useful for performance measurement since it allows arbitrary and controllable input/output sequence lengths with early stopping being disabled and allows apples-to-apples performance comparisons across different frameworks.

```python
python3 run.py benchmark GPT2 [frameworks | trt] --variant [gpt2 | gpt2-large] --working-dir temp --input-seq-len 128 --output-seq-len 256
```

## Testing

```python
pytest
```

It is recommended to use Pytest `4.6.x`. Your Python environment must have already had the setup completed.
