# TensorRT Inference for HuggingFace Transformers 🤗

This repository demonstrates TensorRT inference with models developed using [HuggingFace Transformers](https://huggingface.co/transformers/).

Currently, this repository supports the following models:

1. [GPT2 (text generation task)](https://huggingface.co/transformers/model_doc/gpt2.html). The sample supports following variants of GPT2:

    gpt2 (117M), gpt2-large (774M)

2. [T5 (translation, premise task)](https://huggingface.co/transformers/model_doc/t5.html). The sample supports following variants of T5:

    t5-small (60M), t5-base (220M), t5-large (770M)


## Installation

```python
pip3 install -r requirements.txt
```

## How to run comparison script

```python
python3 run.py compare GPT2 --variant [gpt2 | gpt2-large] --working-dir temp
```

The above script reports :

| script     | accuracy | decoder (sec) | encoder (sec) | full (sec) |
|------------|----------|---------------|---------------|------------|
| frameworks | 1        | 0.0292865     | 0.0174382     | 0.122532   |
| trt        | 1        | 0.00494083    | 0.0068982     | 0.0239782  |


## How to run functional and performance benchmark

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
