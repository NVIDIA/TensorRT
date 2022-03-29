# BERT Inference Using TensorRT

This subfolder of the BERT TensorFlow repository, tested and maintained by NVIDIA, provides scripts to perform high-performance inference using NVIDIA TensorRT.


## Table Of Contents

- [Model Overview](#model-overview)
   * [Model Architecture](#model-architecture)
   * [TensorRT Inference Pipeline](#tensorrt-inference-pipeline)
   * [Version Info](#version-info)
- [Setup](#setup)
   * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
  * [(Optional) Trying a different configuration](#optional-trying-a-different-configuration)
- [Advanced](#advanced)
  * [Scripts and sample code](#scripts-and-sample-code)
  * [Command-line options](#command-line-options)
  * [TensorRT inference process](#tensorrt-inference-process)
- [Accuracy](#accuracy)
  * [Evaluating Post-Training-Quantization INT8 accuracy](#evaluating-ptq-post-training-quantization-int8-accuracy-using-the-squad-dataset)
  * [Evaluating Quantization-Aware-Training INT8 accuracy](#evaluating-qat-quantization-aware-training-int8-accuracy-using-the-squad-dataset)
- [Performance](#performance)
  * [Benchmarking](#benchmarking)
       * [TensorRT inference benchmark](#tensorrt-inference-benchmark)
  * [Results](#results)
    * [Inference performance: NVIDIA A100](#inference-performance-nvidia-a100-40gb)
      * [BERT Base](#bert-base)
      * [BERT Large](#bert-large)
    * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4-16gb)
      * [BERT Base](#bert-base-1)
      * [BERT Large](#bert-large-1)
    * [Inference performance: NVIDIA V100](#inference-performance-nvidia-v100-16gb)
      * [BERT Base](#bert-base-2)
      * [BERT Large](#bert-large-2)
- [Experimental](#experimental)
  * [Variable sequence length](#variable-sequence-length)
      * [Run command lines](#run-command-lines)


## Model overview

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. NVIDIA's BERT is an optimized version of [Google's official implementation](https://github.com/google-research/bert), leveraging mixed precision arithmetic and Tensor Cores for faster inference times while maintaining target accuracy.

Other publicly available implementations of BERT include:
1. [NVIDIA PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
2. [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT)
3. [codertimo](https://github.com/codertimo/BERT-pytorch)
4. [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/master/scripts/bert)
5. [Google's official implementation](https://github.com/google-research/bert)


### Model architecture

BERT's model architecture is a multi-layer bidirectional Transformer encoder. Based on the model size, we have the following two default configurations of BERT:

| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feed-forward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERT-Base |12 encoder| 768| 12|4 x  768|512|110M|
|BERT-Large|24 encoder|1024| 16|4 x 1024|512|330M|

Typically, the language model is followed by a few task-specific layers. The model used here includes layers for question answering.

### TensorRT Inference Pipeline

BERT inference consists of three main stages: tokenization, the BERT model, and finally a projection of the tokenized prediction onto the original text.
Since the tokenizer and projection of the final predictions are not nearly as compute-heavy as the model itself, we run them on the host. The BERT model is GPU-accelerated via TensorRT.

The tokenizer splits the input text into tokens that can be consumed by the model. For details on this process, see [this tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/).

To run the BERT model in TensorRT, we construct the model using TensorRT APIs and import the weights from a pre-trained TensorFlow checkpoint from [NGC](https://ngc.nvidia.com/models/nvidian:bert_tf_v2_large_fp16_128). Finally, a TensorRT engine is generated and serialized to the disk. The various inference scripts then load this engine for inference.

Lastly, the tokens predicted by the model are projected back to the original text to get a final result.

### Version Info

The following software version configuration has been tested:

|Software|Version|
|--------|-------|
|Python|3.6.9|
|TensorRT|7.2.3.4|
|CUDA|11.1.1|


## Setup

The following section lists the requirements that you need to meet in order to run the BERT model.

### Requirements

This demo BERT application can be run within the TensorRT Open Source build container. If running in a different environment, ensure you have the following packages installed.

* [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) - for downloading BERT checkpoints from NGC.
* PyPI Packages:
  * [pycuda](https://pypi.org/project/pycuda/) 2019.1.2
  * [onnx](https://pypi.org/project/onnx/1.6.0/) 1.6.0
  * [tensorflow](https://pypi.org/project/tensorflow/1.15.3/) 1.15
* NVIDIA [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/), [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU with NVIDIA Driver 450.37 or later.


## Quick Start Guide

1. Build and launch the TensorRT-OSS build container. On x86 with Ubuntu 18.04 for example:
    ```bash
    cd <TensorRT-OSS>
    ./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu --os 18.04 --cuda 11.0
    ./docker/launch.sh --tag tensorrt-ubuntu --gpus all --release $TRT_RELEASE --source $TRT_SOURCE
    ```

    **Note:** After this point, all commands should be run from within the container.

2. Build the TensorRT Plugins library from source and install the TensorRT python bindings:
    ```bash
    cd $TRT_SOURCE
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out
    make -j$(nproc)

    pip3 install /tensorrt/python/tensorrt-7.2*-cp36-none-linux_x86_64.whl
    ```
    **Note:** While the workflow and Performance Data presented here are based on plugin library built from source, the BERT sample is also expected to work with pre-compiled libraries shipped with TensorRT releases.

3. Download the SQuAD dataset and BERT checkpoints:
    ```bash
    cd $TRT_SOURCE/demo/BERT
    ```

    Download SQuAD v1.1 training and dev dataset.
    ```bash
    bash scripts/download_squad.sh
    ```

    Download Tensorflow checkpoints for BERT large model with sequence length 128, fine-tuned for SQuAD v2.0.
    ```bash
    bash scripts/download_model.sh
    ````

**Note:** Since the datasets and checkpoints are stored in the directory mounted from the host, they do *not* need to be downloaded each time the container is launched. 

4. Build a TensorRT engine. To build an engine, run the `builder.py` script. For example:
    ```bash
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/model.ckpt -o /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine -b 1 -s 128 --fp16 -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), and sequence length of 128 (`-s 128`) using mixed precision (`--fp16`) using the BERT Large SQuAD v2 FP16 Sequence Length 128 checkpoint (`-c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1`).

5. Run inference. Two options are provided for running the model.

    a. `inference.py` script
    This script accepts a passage and question and then runs the engine to generate an answer.
    For example:
    ```bash
    python3 inference.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/vocab.txt
    ```

    b. `inference.ipynb` Jupyter Notebook
    The Jupyter Notebook includes a passage and various example questions and allows you to interactively make modifications and see the outcome.
    To launch the Jupyter Notebook from inside the container, run:
    ```bash
    jupyter notebook --ip 0.0.0.0 inference.ipynb
    ```
    Then, use your browser to open the link displayed. The link should look similar to: `http://127.0.0.1:8888/?token=<TOKEN>`
    
6. Run inference with CUDA Graph support.

    A separate python `inference_c.py` script is provided to run inference with CUDA Graph support. This is necessary since CUDA Graph is only supported through CUDA C/C++ APIs, not pyCUDA. The `inference_c.py` script uses pybind11 to interface with C/C++ for CUDA graph capturing and launching. The cmdline interface is the same as `inference.py` except for an extra `--enable-graph` option.
    
    ```bash
    mkdir -p build
    cd build; cmake ..
    make; cd ..
    python3 inference_c.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine --enable-graph -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/vocab.txt
    ```

    A separate C/C++ inference benchmark executable `perf` (compiled from `perf.cpp`) is provided to run inference benchmarks with CUDA Graph. The cmdline interface is the same as `perf.py` except for an extra `--enable_graph` option.
    
    ```bash
    build/perf -e /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine -b 1 -s 128 -w 100 -i 1000 --enable_graph
    ```
    

### (Optional) Trying a different configuration

If you would like to run another configuration, you can manually download checkpoints using the included script. For example, run:
```bash
bash scripts/download_model.sh base
```
to download a BERT Base model instead of the default BERT Large model.

To view all available model options, run:
```bash
bash scripts/download_model.sh -h
```

## Advanced

The following sections provide greater details on inference with TensorRT.

### Scripts and sample code

In the `root` directory, the most important files are:

- `builder.py` - Builds an engine for the specified BERT model
- `Dockerfile` - Container which includes dependencies and model checkpoints to run BERT
- `inference.ipynb` - Runs inference interactively
- `inference.py` - Runs inference with a given passage and question
- `perf.py` - Runs inference benchmarks

The `scripts/` folder encapsulates all the one-click scripts required for running various supported functionalities, such as:

- `build.sh` - Builds a Docker container that is ready to run BERT
- `launch.sh` - Launches the container created by the `build.sh` script.
- `download_model.sh` - Downloads pre-trained model checkpoints from NGC
- `inference_benchmark.sh` - Runs an inference benchmark and prints results

Other folders included in the `root` directory are:

- `helpers` - Contains helpers for tokenization of inputs

The `infer_c/` folder contains all the necessary C/C++ files required for CUDA Graph support.
- `bert_infer.h` - Defines necessary data structures for running BERT inference
- `infer_c.cpp` - Defines C/C++ interface using pybind11 that can be plugged into `inference_c.py`
- `perf.cpp` - Runs inference benchmarks. It is equivalent to `perf.py`, with an extra option `--enable_graph` to enable CUDA Graph support.

### Command-line options

To view the available parameters for each script, you can use the help flag (`-h`).

### TensorRT inference process

As mentioned in the [Quick Start Guide](#quick-start-guide), two options are provided for running inference:
1. The `inference.py` script which accepts a passage and a question and then runs the engine to generate an answer. Alternatively, this script can be used to run inference on the Squad dataset.
2. The `inference.ipynb` Jupyter Notebook which includes a passage and various example questions and allows you to interactively make modifications and see the outcome.

## Accuracy

### Evaluating PTQ (post-training quantization) Int8 Accuracy Using The SQuAD Dataset
1.  Download Tensorflow checkpoints for a BERT Large FP16 SQuAD v2 model with a sequence length of 384:
    ```bash
    bash scripts/download_model.sh large 384 v2
    ```

2. Build an engine:

    **Turing and Ampere GPUs**
    ```bash
    # QKVToContextPlugin and SkipLayerNormPlugin supported with INT8 I/O. To enable, use -imh and -iln builder flags respectively.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/model.ckpt -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100 -iln -imh
    ```

    **Xavier GPU**
    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/model.ckpt -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100 -iln 
    ```

    **Volta GPU**
    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/model.ckpt -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), calibration dataset squad (`--squad-json ./squad/train-v1.1.json`), calibration sentences number 100 (`--calib-num 100`), and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:
    ```bash
    python3 inference.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```
### Evaluating QAT (quantization aware training) Int8 Accuracy Using The SQuAD Dataset
1.  Download checkpoint for BERT Large FP16 SQuAD v1.1 model with sequence length of 384:
    ```bash
    bash scripts/download_model.sh pyt v1_1
    ```

2. Build an engine:

    **Turing and Ampere GPUs**
    ```bash
    # QKVToContextPlugin and SkipLayerNormPlugin supported with INT8 I/O. To enable, use -imh and -iln builder flags respectively.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln -imh
    ```

    **Xavier GPU**
    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln 
    ```

    **Volta GPU**
    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx 
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:

    ```bash
    python3 inference.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in inference modes.

#### TensorRT inference benchmark

The inference benchmark is performed on a single GPU by the `inference_benchmark.sh` script, which takes the following steps for each set of model parameters:

1. Downloads checkpoints and builds a TensorRT engine if it does not already exist.

2. Runs 100 warm-up iteration then runs inference for 1000 to 2000 iterations for each batch size specified in the script, selecting the profile best for each size.

**Note:** The time measurements do not include the time required to copy inputs to the device and copy outputs to the host.

To run the inference benchmark script, run:
```bash
bash scripts/inference_benchmark.sh --gpu <arch>
```
Options for `<arch>` are: 'Volta', 'Xavier', 'Turing', 'Ampere'

Note: Some of the configurations in the benchmark script require 16GB of GPU memory. On GPUs with smaller amounts of memory, parts of the benchmark may fail to run.

Also note that BERT Large engines, especially using mixed precision with large batch sizes and sequence lengths may take a couple hours to build.

### Results

The following sections provide details on how we achieved our performance and inference.

#### Inference performance: NVIDIA A100 (40GB)

Our results were obtained by running the `scripts/inference_benchmark.sh --gpu Ampere` script in the container generated by the TensorRT OSS Dockerfile on NVIDIA A100 with (1x A100 40G) GPUs.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.61 | 0.75 | 0.60 | 0.89 | 0.90 | 0.73 |
| 128 | 2 | 0.82 | 0.82 | 0.66 | 1.10 | 1.10 | 0.89 |
| 128 | 4 | 1.03 | 1.03 | 0.82 | 1.15 | 1.47 | 1.16 |
| 128 | 8 | 1.02 | 1.31 | 1.03 | 1.67 | 1.67 | 1.67 |
| 128 | 12 | 1.42 | 1.43 | 1.42 | 2.07 | 2.08 | 2.07 |
| 128 | 16 | 1.54 | 1.54 | 1.53 | 2.49 | 2.50 | 2.49 |
| 128 | 24 | 2.24 | 2.25 | 2.24 | 3.52 | 3.58 | 3.48 |
| 128 | 32 | 2.74 | 2.75 | 2.74 | 4.39 | 4.43 | 4.36 |
| 128 | 64 | 4.88 | 4.89 | 4.87 | 8.73 | 8.77 | 8.69 |
| 128 | 128 | 9.21 | 9.27 | 9.17 | 17.11 | 17.13 | 16.96 |
| 384 | 1 | 1.22 | 1.23 | 1.22 | 1.44 | 1.45 | 1.44 |
| 384 | 2 | 1.40 | 1.41 | 1.40 | 1.90 | 1.93 | 1.89 |
| 384 | 4 | 1.88 | 1.88 | 1.87 | 2.48 | 2.49 | 2.47 |
| 384 | 8 | 2.64 | 2.65 | 2.64 | 3.81 | 3.81 | 3.78 |
| 384 | 12 | 3.80 | 3.81 | 3.80 | 5.36 | 5.41 | 5.34 |
| 384 | 16 | 4.71 | 4.71 | 4.70 | 7.34 | 7.36 | 7.27 |
| 384 | 24 | 6.56 | 6.57 | 6.56 | 10.50 | 10.59 | 10.44 |
| 384 | 32 | 8.75 | 8.75 | 8.74 | 14.35 | 14.35 | 14.29 |
| 384 | 64 | 16.68 | 16.69 | 16.52 | 27.60 | 27.69 | 27.46 |
| 384 | 128 | 31.76 | 31.84 | 31.50 | 54.13 | 54.21 | 53.44 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.34 | 1.34 | 1.34 | 1.81 | 1.82 | 1.81 |
| 128 | 2 | 1.57 | 2.00 | 1.58 | 2.25 | 2.26 | 2.25 |
| 128 | 4 | 2.09 | 2.17 | 2.08 | 3.33 | 3.34 | 3.33 |
| 128 | 8 | 3.30 | 3.31 | 3.28 | 4.90 | 4.91 | 4.88 |
| 128 | 12 | 3.94 | 3.94 | 3.91 | 6.01 | 6.01 | 5.95 |
| 128 | 16 | 5.19 | 5.20 | 5.19 | 7.95 | 7.95 | 7.92 |
| 128 | 24 | 6.49 | 6.49 | 6.48 | 11.12 | 11.13 | 11.02 |
| 128 | 32 | 8.60 | 8.61 | 8.58 | 14.70 | 14.72 | 14.64 |
| 128 | 64 | 14.91 | 14.93 | 14.83 | 28.05 | 28.30 | 27.93 |
| 128 | 128 | 28.51 | 28.56 | 28.23 | 54.47 | 54.60 | 54.38 |
| 384 | 1 | 2.76 | 2.77 | 2.76 | 3.45 | 3.46 | 3.45 |
| 384 | 2 | 3.24 | 3.25 | 3.24 | 4.74 | 4.75 | 4.73 |
| 384 | 4 | 4.71 | 4.72 | 4.70 | 6.78 | 6.80 | 6.77 |
| 384 | 8 | 8.16 | 8.17 | 8.15 | 12.45 | 12.56 | 12.38 |
| 384 | 12 | 10.70 | 10.71 | 10.69 | 17.68 | 17.69 | 17.62 |
| 384 | 16 | 13.91 | 13.92 | 13.90 | 23.47 | 23.49 | 23.30 |
| 384 | 24 | 19.66 | 19.67 | 19.64 | 34.39 | 34.50 | 34.23 |
| 384 | 32 | 25.75 | 25.76 | 25.63 | 45.10 | 45.20 | 44.99 |
| 384 | 64 | 50.39 | 50.50 | 49.96 | 88.18 | 88.33 | 87.37 |
| 384 | 128 | 97.32 | 97.69 | 96.60 | 171.10 | 171.30 | 170.92 |


#### Inference performance: NVIDIA T4 (16GB)

Our results were obtained by running the `scripts/inference_benchmark.sh --gpu Turing` script in the container generated by the TensorRT OSS Dockerfile on NVIDIA T4 with (1x T4 16G) GPUs.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.61 | 1.63 | 1.43 | 2.06 | 2.11 | 2.02 |
| 128 | 2 | 1.91 | 1.93 | 1.86 | 2.72 | 2.76 | 2.66 |
| 128 | 4 | 2.78 | 2.79 | 2.75 | 4.48 | 4.59 | 4.40 |
| 128 | 8 | 4.82 | 4.93 | 4.70 | 9.18 | 9.74 | 8.77 |
| 128 | 12 | 6.65 | 6.72 | 6.55 | 12.89 | 13.44 | 12.44 |
| 128 | 16 | 9.59 | 9.84 | 9.43 | 19.01 | 19.67 | 18.39 |
| 128 | 24 | 14.06 | 14.36 | 13.64 | 26.77 | 27.22 | 26.05 |
| 128 | 32 | 20.30 | 20.76 | 19.70 | 36.23 | 36.82 | 35.59 |
| 128 | 64 | 37.36 | 37.87 | 36.66 | 70.76 | 70.93 | 69.85 |
| 128 | 128 | 74.34 | 74.93 | 73.16 | 138.77 | 139.19 | 137.61 |
| 384 | 1 | 2.64 | 2.67 | 2.59 | 4.06 | 4.14 | 3.95 |
| 384 | 2 | 4.21 | 4.21 | 4.12 | 6.93 | 7.22 | 6.81 |
| 384 | 4 | 7.59 | 7.76 | 7.43 | 13.24 | 13.38 | 13.03 |
| 384 | 8 | 15.09 | 15.21 | 14.85 | 27.61 | 28.03 | 27.08 |
| 384 | 12 | 23.00 | 23.26 | 22.54 | 41.45 | 41.81 | 40.77 |
| 384 | 16 | 30.48 | 30.78 | 29.29 | 55.37 | 55.40 | 54.47 |
| 384 | 24 | 45.16 | 45.45 | 44.30 | 85.40 | 85.85 | 84.04 |
| 384 | 32 | 60.02 | 60.46 | 58.68 | 113.39 | 114.02 | 112.26 |
| 384 | 64 | 120.93 | 121.49 | 117.98 | 223.25 | 223.71 | 221.53 |
| 384 | 128 | 243.54 | 244.02 | 242.56 | 426.67 | 427.66 | 424.75 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 3.86 | 3.95 | 3.80 | 5.60 | 5.67 | 5.51 |
| 128 | 2 | 5.29 | 5.31 | 5.22 | 8.18 | 8.39 | 8.01 |
| 128 | 4 | 8.76 | 8.83 | 8.53 | 15.37 | 15.87 | 14.89 |
| 128 | 8 | 14.88 | 14.98 | 14.73 | 29.34 | 29.85 | 28.61 |
| 128 | 12 | 23.76 | 24.02 | 23.55 | 45.18 | 45.45 | 44.31 |
| 128 | 16 | 30.49 | 30.78 | 29.91 | 60.83 | 61.30 | 58.96 |
| 128 | 24 | 49.40 | 50.11 | 48.53 | 88.70 | 89.08 | 87.23 |
| 128 | 32 | 61.83 | 62.36 | 60.19 | 119.50 | 120.45 | 117.72 |
| 128 | 64 | 121.34 | 121.72 | 120.08 | 241.78 | 242.50 | 238.93 |
| 128 | 128 | 240.73 | 241.28 | 238.18 | 460.31 | 461.01 | 457.69 |
| 384 | 1 | 7.59 | 7.62 | 7.49 | 12.51 | 12.64 | 12.25 |
| 384 | 2 | 12.94 | 12.96 | 12.65 | 23.62 | 24.00 | 23.09 |
| 384 | 4 | 25.29 | 25.48 | 24.96 | 46.84 | 47.00 | 45.86 |
| 384 | 8 | 51.09 | 51.54 | 50.25 | 91.36 | 92.03 | 90.22 |
| 384 | 12 | 70.58 | 71.29 | 69.61 | 134.00 | 135.07 | 132.70 |
| 384 | 16 | 93.90 | 94.35 | 92.87 | 178.72 | 179.71 | 176.76 |
| 384 | 24 | 142.36 | 142.79 | 141.27 | 268.85 | 270.07 | 265.63 |
| 384 | 32 | 187.65 | 188.15 | 186.25 | 361.09 | 362.01 | 358.48 |
| 384 | 64 | 375.83 | 376.57 | 373.65 | 702.71 | 703.86 | 700.19 |
| 384 | 128 | 759.70 | 760.76 | 756.42 | 1461.61 | 1462.63 | 1459.52 |


#### Inference performance: NVIDIA V100 (16GB, 300W)

Our results were obtained by running the `scripts/inference_benchmark.sh --gpu Volta` script in the container generated by the TensorRT OSS Dockerfile on NVIDIA V100 with (1x V100 16G) GPUs.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.33 | 1.34 | 1.26 | 1.28 | 1.29 | 1.15 |
| 128 | 2 | 1.82 | 1.83 | 1.75 | 1.85 | 1.92 | 1.85 |
| 128 | 4 | 2.45 | 2.54 | 2.44 | 2.26 | 2.35 | 2.24 |
| 128 | 8 | 3.67 | 3.68 | 3.66 | 3.51 | 3.53 | 3.48 |
| 128 | 12 | 4.94 | 4.95 | 4.90 | 4.76 | 4.79 | 4.74 |
| 128 | 16 | 6.62 | 6.63 | 6.57 | 6.30 | 6.33 | 6.27 |
| 128 | 24 | 8.87 | 8.91 | 8.81 | 8.70 | 8.70 | 8.62 |
| 128 | 32 | 12.15 | 12.19 | 12.05 | 11.96 | 11.98 | 11.87 |
| 128 | 64 | 22.99 | 23.02 | 22.79 | 22.58 | 22.67 | 22.45 |
| 128 | 128 | 43.37 | 43.53 | 43.03 | 43.35 | 43.50 | 43.13 |
| 384 | 1 | 2.58 | 2.67 | 2.56 | 2.23 | 2.33 | 2.22 |
| 384 | 2 | 3.41 | 3.42 | 3.38 | 3.40 | 3.41 | 3.39 |
| 384 | 4 | 6.17 | 6.18 | 6.14 | 5.83 | 5.84 | 5.79 |
| 384 | 8 | 10.61 | 10.63 | 10.54 | 10.55 | 10.57 | 10.47 |
| 384 | 12 | 17.71 | 17.75 | 17.59 | 16.10 | 16.13 | 16.01 |
| 384 | 16 | 20.76 | 20.85 | 20.65 | 20.71 | 20.76 | 20.56 |
| 384 | 24 | 30.45 | 30.58 | 30.23 | 30.31 | 30.45 | 30.21 |
| 384 | 32 | 40.35 | 40.43 | 40.06 | 40.41 | 40.60 | 40.10 |
| 384 | 64 | 79.35 | 79.49 | 78.92 | 79.41 | 79.66 | 79.08 |
| 384 | 128 | 156.58 | 156.74 | 155.80 | 156.38 | 156.53 | 155.71 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 2.99 | 3.00 | 2.98 | 3.11 | 3.12 | 3.10 |
| 128 | 2 | 4.15 | 4.16 | 4.14 | 4.03 | 4.04 | 4.01 |
| 128 | 4 | 6.60 | 6.62 | 6.55 | 6.34 | 6.36 | 6.29 |
| 128 | 8 | 10.80 | 10.86 | 10.71 | 11.00 | 11.04 | 10.90 |
| 128 | 12 | 15.67 | 15.77 | 15.54 | 15.60 | 15.67 | 15.44 |
| 128 | 16 | 20.79 | 20.89 | 20.63 | 20.69 | 20.75 | 20.54 |
| 128 | 24 | 31.03 | 31.16 | 30.63 | 29.41 | 29.57 | 29.21 |
| 128 | 32 | 38.17 | 38.37 | 37.82 | 38.21 | 38.34 | 37.89 |
| 128 | 64 | 75.87 | 75.94 | 75.35 | 71.86 | 72.14 | 71.37 |
| 128 | 128 | 141.83 | 142.27 | 141.13 | 141.39 | 141.59 | 140.58 |
| 384 | 1 | 6.05 | 6.08 | 6.01 | 6.24 | 6.26 | 6.21 |
| 384 | 2 | 10.26 | 10.29 | 10.19 | 10.12 | 10.16 | 10.04 |
| 384 | 4 | 18.33 | 18.40 | 18.21 | 18.11 | 18.17 | 17.97 |
| 384 | 8 | 36.36 | 36.50 | 35.99 | 34.24 | 34.36 | 33.98 |
| 384 | 12 | 48.78 | 48.91 | 48.41 | 48.97 | 49.05 | 48.60 |
| 384 | 16 | 65.64 | 65.94 | 65.13 | 64.77 | 64.87 | 64.30 |
| 384 | 24 | 95.20 | 95.36 | 94.62 | 94.54 | 94.70 | 94.27 |
| 384 | 32 | 124.50 | 124.89 | 123.70 | 124.46 | 124.62 | 123.87 |
| 384 | 64 | 246.21 | 246.50 | 244.77 | 245.84 | 246.10 | 244.45 |

## Experimental
### Variable sequence length
In our prior implementation, we used inputs padded to max length along with corresponding input masks to handle variable sequence length inputs in a batch. The padding results in some wasted computations which can be avoided by handling variable sequence length inputs natively. Now we have a new approach called the variable sequence length method. By concatenating each input id into a single long input id, and concatenating each input segment id into a single long segment id, TensorRT can know the exact starts and ends by providing an extra sequence length buffer that contains the start and end positions of each sequence. Now we can eliminate the wasted computation in the input paddings.

Note this is an experimental feature because we only support Xavier+ GPUs, also there is neither FP32 support nor INT8 PTQ calibration.

#### Run command lines

1.  Download checkpoint for BERT Large FP16 SQuAD v1.1 model with sequence length of 384:
    ```bash
    bash scripts/download_model.sh pyt v1_1
    ```

2. Build an engine:

    **FP16 engine**
    ```bash
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder_varseqlen.py -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -o /workspace/TensorRT/demo/BERT/engines/bert_varseq_fp16.engine -b 1 -s 64 --fp16 -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 64 (`-s 64`) using FP16 precision computation where possible (`--fp16`).


    **INT8 engine**
    ```bash
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder_varseqlen.py -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -o /workspace/TensorRT/demo/BERT/engines/bert_varseq_int8.engine -b 1 -s 256 --int8 --fp16 -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 256 (`-s 256`) using INT8 precision computation where possible (`--int8`).

3. Run inference 

    Evaluate the F1 score and exact match score using the squad dataset:
    
    ```bash
    python3 inference_varseqlen.py -e /workspace/TensorRT/demo/BERT/engines/bert_varseq_int8.engine -s 256 -sq ./squad/dev-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```

    Run the quesion and answer mode:

    ```bash
    python3 inference_varseqlen.py -e /workspace/TensorRT/demo/BERT/engines/bert_varseq_int8.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -s 256
    ```

3. Collect performance data

    ```bash
    python3 perf_varseqlen.py -e /workspace/TensorRT/demo/BERT/engines/bert_varseq_int8.engine -b 1 -s 256
    ```

    This will collect performance data run use batch size 1 (`-b 1`) and sequence length of 256 (`-s 256`). 

4. Collect performance data with CUDA graph enabled

    We can use the same `inference_c.py` and `build/perf` to collect performance data with cuda graph enabled. The command line is the same as run without variable sequence length. 

