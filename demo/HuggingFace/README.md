# TensorRT Inference for HuggingFace Transformers 🤗

This repository demonstrates TensorRT inference with models developed using [HuggingFace Transformers](https://huggingface.co/transformers/).

Currently, this repository supports the following models:

1. [GPT2 (text generation task)](https://huggingface.co/transformers/model_doc/gpt2.html). The sample supports following variants of GPT2:

    gpt2 (117M), gpt2-large (774M)

2. [T5 (translation, premise task)](https://huggingface.co/transformers/model_doc/t5.html). The sample supports following variants of T5:

    t5-small (60M), t5-base (220M), t5-large (770M)

3. [BART (summarization task)](https://huggingface.co/docs/transformers/model_doc/bart.html). The sample supports the following variants of BART:

    facebook/bart-base (139M), facebook/bart-large (406M), facebook/bart-large-cnn (406M)

## Setup

Follow the setup steps in the TensorRT OSS repository, and then install the additional dependencies below. It is recommended to experiment inside Docker container.

```python
pip3 install -r requirements.txt
```

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
python3 run.py compare GPT2 --variant [gpt2 | gpt2-large] --working-dir temp
```

The above script compares the performance of PyTorch framework inference and TensorRT inference for GPT2:

| script     | accuracy | decoder (sec) | encoder (sec) | full (sec) |
|------------|----------|---------------|---------------|------------|
| frameworks | 1        | 0.0292865     | 0.0174382     | 0.122532   |
| trt        | 1        | 0.00494083    | 0.0068982     | 0.0239782  |

Notes: `--variant` designates the pre-trained model for testing. `--working-dir` saves the downloaded pre-trained models, onnx model files, and TRT engine files. accuracy of 1.0 indicates correct results in consistency with the expected outputs in `checkpoint.toml`.

## How to run functional and performance benchmark

The `run` action will run the specific script under the model directory.

```python
python3 run.py run GPT2 [frameworks | trt] --variant [gpt2 | gpt2-large] --working-dir temp
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

## How to run with K-V cache

For BART, use `--enable-kv-cache` option to get the same effect of HuggingFace's `use_cache` option. For encoder-decoder models, this option will use key & value cache in decoder for uni-directional self-attention and encoder-decoder cross-attention.

```python
python3 run.py run BART frameworks --variant facebook/bart-base --working-dir temp --enable-kv-cache
```

## Testing

```python
pytest
```

It is recommended to use Pytest `4.6.x`. Your Python environment must have already had the setup completed.
