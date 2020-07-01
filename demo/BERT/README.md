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
|TensorRT|7.1.3.4|
|CUDA|11.0.171|


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
    export LD_LIBRARY_PATH=`pwd`/build/out:$LD_LIBRARY_PATH:/tensorrt/lib
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out
    make -j$(nproc)

    pip3 install /tensorrt/python/tensorrt-7.1*-cp36-none-linux_x86_64.whl
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

    Download Tensorflow checkpoints for BERT large model with sequence length 128 and fp16 weights, fine-tuned for SQuAD v2.0.
    ```bash
    bash scripts/download_model.sh
    ````

**Note:** Since the datasets and checkpoints are stored in the directory mounted from the host, they do *not* need to be downloaded each time the container is launched. 

4. Build a TensorRT engine. To build an engine, run the `builder.py` script. For example:
    ```bash
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/model.ckpt-8144 -o /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine -b 1 -s 128 --fp16 -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), and sequence length of 128 (`-s 128`) using mixed precision (`--fp16`) using the BERT Large V2 FP16 Sequence Length 128 checkpoint (`-c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2`).

5. Run inference. Two options are provided for running the model.

    a. `inference.py` script
    This script accepts a passage and question and then runs the engine to generate an answer.
    For example:
    ```bash
    python3 inference.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/vocab.txt
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
    python3 inference_c.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine --enable-graph -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/vocab.txt
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
    bash scripts/download_model.sh large fp16 384 v2
    ```

2. Build an engine:

    **Turing and Ampere GPUs**
    ```bash
    # QKVToContextPlugin and SkipLayerNormPlugin supported with INT8 I/O. To enable, use -imh and -iln builder flags respectively.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/model.ckpt-8144 -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 --squad-json ./squad/train-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt --calib-num 100 -iln -imh
    ```

    **Xavier GPU**
    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/model.ckpt-8144 -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 --squad-json ./squad/train-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt --calib-num 100 -iln 
    ```

    **Volta GPU**
    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/model.ckpt-8144 -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 --squad-json ./squad/train-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt --calib-num 100
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), calibration dataset squad (`--squad-json ./squad/train-v1.1.json`), calibration sentences number 100 (`--calib-num 100`), and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:
    ```bash
    python3 inference.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -o ./predictions.json
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
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln -imh
    ```

    **Xavier GPU**
    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln 
    ```

    **Volta GPU**
    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx 
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:

    ```bash
    python3 inference.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -o ./predictions.json
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
| 128 | 1 | 0.77 | 0.77 | 0.77 | 0.78 | 0.80 | 0.78 |
| 128 | 2 | 0.76 | 0.77 | 0.76 | 0.92 | 0.93 | 0.92 |
| 128 | 4 | 0.93 | 1.18 | 0.93 | 1.19 | 1.51 | 1.19 |
| 128 | 8 | 1.19 | 1.20 | 1.19 | 1.78 | 1.78 | 1.77 |
| 128 | 12 | 1.57 | 1.57 | 1.56 | 2.07 | 2.08 | 2.05 |
| 128 | 16 | 1.88 | 1.89 | 1.88 | 2.54 | 2.60 | 2.52 |
| 128 | 24 | 2.65 | 2.65 | 2.64 | 3.65 | 3.70 | 3.61 |
| 128 | 32 | 3.21 | 3.22 | 3.21 | 4.71 | 4.74 | 4.67 |
| 128 | 64 | 5.69 | 5.70 | 5.64 | 8.87 | 8.96 | 8.81 |
| 128 | 128 | 10.84 | 10.85 | 10.70 | 17.61 | 17.62 | 17.44 |
| 384 | 1 | 1.34 | 1.35 | 1.34 | 1.46 | 1.46 | 1.45 |
| 384 | 2 | 1.56 | 1.79 | 1.56 | 1.85 | 1.85 | 1.84 |
| 384 | 4 | 2.02 | 2.03 | 2.02 | 2.46 | 2.46 | 2.45 |
| 384 | 8 | 2.94 | 2.95 | 2.94 | 3.91 | 3.92 | 3.86 |
| 384 | 12 | 4.07 | 4.07 | 4.06 | 5.54 | 5.55 | 5.47 |
| 384 | 16 | 5.22 | 5.23 | 5.21 | 7.78 | 7.79 | 7.69 |
| 384 | 24 | 7.42 | 7.42 | 7.37 | 10.75 | 10.76 | 10.63 |
| 384 | 32 | 9.92 | 9.93 | 9.77 | 14.58 | 14.73 | 14.52 |
| 384 | 64 | 18.74 | 18.78 | 18.61 | 28.66 | 28.70 | 28.39 |
| 384 | 128 | 36.40 | 36.42 | 36.05 | 55.36 | 55.90 | 55.21 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.60 | 1.61 | 1.60 | 1.87 | 1.88 | 1.87 |
| 128 | 2 | 1.94 | 1.95 | 1.94 | 2.36 | 2.37 | 2.35 |
| 128 | 4 | 2.45 | 2.46 | 2.44 | 3.36 | 3.37 | 3.36 |
| 128 | 8 | 3.82 | 3.83 | 3.79 | 4.98 | 5.00 | 4.95 |
| 128 | 12 | 4.22 | 4.23 | 4.22 | 6.45 | 6.46 | 6.38 |
| 128 | 16 | 5.75 | 5.75 | 5.74 | 8.50 | 8.53 | 8.43 |
| 128 | 24 | 7.10 | 7.11 | 7.04 | 11.47 | 11.49 | 11.31 |
| 128 | 32 | 9.61 | 9.61 | 9.51 | 15.49 | 15.50 | 15.25 |
| 128 | 64 | 17.25 | 17.25 | 17.11 | 29.43 | 29.73 | 29.29 |
| 128 | 128 | 33.25 | 33.58 | 33.05 | 56.98 | 57.17 | 56.68 |
| 384 | 1 | 3.00 | 3.01 | 2.99 | 3.52 | 3.53 | 3.51 |
| 384 | 2 | 3.71 | 3.72 | 3.71 | 4.97 | 4.99 | 4.97 |
| 384 | 4 | 5.08 | 5.09 | 5.08 | 7.01 | 7.01 | 6.92 |
| 384 | 8 | 9.04 | 9.05 | 9.04 | 12.71 | 12.72 | 12.67 |
| 384 | 12 | 11.65 | 11.71 | 11.57 | 18.24 | 18.25 | 18.04 |
| 384 | 16 | 15.63 | 15.63 | 15.49 | 24.24 | 24.28 | 23.94 |
| 384 | 24 | 22.57 | 22.61 | 22.36 | 35.77 | 35.78 | 35.38 |
| 384 | 32 | 29.66 | 29.66 | 29.33 | 47.09 | 47.11 | 46.81 |
| 384 | 64 | 57.20 | 57.34 | 56.93 | 92.12 | 92.49 | 91.61 |
| 384 | 128 | 112.00 | 112.42 | 111.24 | 180.61 | 181.02 | 179.56 |


#### Inference performance: NVIDIA T4 (16GB)

Our results were obtained by running the `scripts/inference_benchmark.sh --gpu Turing` script in the container generated by the TensorRT OSS Dockerfile on NVIDIA T4 with (1x T4 16G) GPUs.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.67 | 1.67 | 1.66 | 1.82 | 1.96 | 1.76 |
| 128 | 2 | 1.94 | 1.95 | 1.89 | 2.58 | 2.67 | 2.50 |
| 128 | 4 | 2.73 | 2.80 | 2.64 | 4.30 | 4.34 | 4.17 |
| 128 | 8 | 4.93 | 4.96 | 4.81 | 8.85 | 9.74 | 8.36 |
| 128 | 12 | 6.85 | 7.05 | 6.70 | 12.83 | 13.19 | 12.34 |
| 128 | 16 | 9.65 | 9.89 | 9.43 | 17.70 | 18.27 | 17.01 |
| 128 | 24 | 15.04 | 15.70 | 14.68 | 27.00 | 27.87 | 26.50 |
| 128 | 32 | 20.55 | 21.01 | 19.88 | 34.51 | 34.81 | 33.83 |
| 128 | 64 | 40.48 | 41.29 | 39.87 | 67.84 | 68.57 | 67.03 |
| 128 | 128 | 82.17 | 82.53 | 80.95 | 132.78 | 133.23 | 131.64 |
| 384 | 1 | 2.75 | 2.78 | 2.67 | 3.73 | 3.79 | 3.63 |
| 384 | 2 | 4.22 | 4.38 | 4.09 | 6.68 | 7.27 | 6.53 |
| 384 | 4 | 7.87 | 8.07 | 7.75 | 13.22 | 13.50 | 12.83 |
| 384 | 8 | 16.07 | 16.13 | 15.77 | 28.01 | 28.72 | 27.48 |
| 384 | 12 | 23.87 | 24.15 | 23.53 | 40.96 | 41.51 | 39.39 |
| 384 | 16 | 31.87 | 32.25 | 30.99 | 51.56 | 51.83 | 51.00 |
| 384 | 24 | 48.14 | 48.33 | 47.22 | 82.06 | 82.56 | 80.13 |
| 384 | 32 | 64.07 | 64.48 | 63.19 | 102.64 | 103.33 | 101.20 |
| 384 | 64 | 129.58 | 130.37 | 125.79 | 215.79 | 216.38 | 213.87 |
| 384 | 128 | 258.69 | 259.74 | 245.91 | 414.96 | 415.57 | 413.16 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 4.20 | 4.35 | 4.10 | 5.05 | 5.21 | 4.91 |
| 128 | 2 | 5.41 | 5.70 | 5.30 | 7.99 | 8.31 | 7.79 |
| 128 | 4 | 8.48 | 8.68 | 8.32 | 14.87 | 15.28 | 14.44 |
| 128 | 8 | 15.20 | 15.22 | 14.91 | 29.66 | 30.20 | 28.97 |
| 128 | 12 | 23.54 | 23.72 | 23.21 | 45.48 | 45.90 | 44.91 |
| 128 | 16 | 31.04 | 31.38 | 30.46 | 62.06 | 62.61 | 60.27 |
| 128 | 24 | 48.00 | 48.59 | 47.44 | 84.17 | 84.50 | 83.43 |
| 128 | 32 | 64.41 | 64.77 | 63.54 | 113.60 | 113.98 | 112.32 |
| 128 | 64 | 128.03 | 128.45 | 126.36 | 223.89 | 224.83 | 220.75 |
| 128 | 128 | 246.96 | 247.80 | 245.00 | 441.52 | 442.26 | 439.65 |
| 384 | 1 | 7.88 | 8.06 | 7.73 | 11.84 | 12.11 | 11.51 |
| 384 | 2 | 13.00 | 13.18 | 12.80 | 23.59 | 24.13 | 23.12 |
| 384 | 4 | 25.14 | 25.32 | 24.70 | 46.66 | 46.69 | 45.81 |
| 384 | 8 | 50.14 | 50.65 | 49.41 | 86.74 | 87.47 | 85.40 |
| 384 | 12 | 72.92 | 73.01 | 71.86 | 127.10 | 127.44 | 125.66 |
| 384 | 16 | 97.00 | 97.26 | 95.47 | 169.41 | 169.93 | 167.55 |
| 384 | 24 | 149.70 | 150.28 | 148.00 | 258.26 | 258.88 | 255.79 |
| 384 | 32 | 192.74 | 193.85 | 190.59 | 339.87 | 340.55 | 337.86 |
| 384 | 64 | 385.85 | 387.66 | 383.62 | 692.10 | 692.88 | 689.73 |
| 384 | 128 | 780.95 | 781.81 | 778.82 | 1367.61 | 1368.85 | 1365.16 |


#### Inference performance: NVIDIA V100 (16GB)

Our results were obtained by running the `scripts/inference_benchmark.sh --gpu Volta` script in the container generated by the TensorRT OSS Dockerfile on NVIDIA V100 with (1x V100 16G) GPUs.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.39 | 1.39 | 1.39 | 1.23 | 1.23 | 1.23 |
| 128 | 2 | 1.76 | 1.76 | 1.75 | 1.49 | 1.49 | 1.48 |
| 128 | 4 | 2.35 | 2.36 | 2.34 | 2.12 | 2.13 | 2.11 |
| 128 | 8 | 3.69 | 3.7 | 3.65 | 3.35 | 3.36 | 3.32 |
| 128 | 12 | 4.79 | 4.83 | 4.75 | 4.65 | 4.67 | 4.61 |
| 128 | 16 | 6.7 | 6.72 | 6.64 | 6.3 | 6.35 | 6.25 |
| 128 | 24 | 8.95 | 8.96 | 8.9 | 8.68 | 8.71 | 8.6 |
| 128 | 32 | 14.74 | 14.77 | 14.59 | 14.16 | 14.18 | 14.06 |
| 128 | 64 | 24.12 | 24.14 | 23.98 | 22.57 | 22.63 | 22.47 |
| 128 | 128 | 45.59 | 45.65 | 45.53 | 43.45 | 43.51 | 43.25 |
| 384 | 1 | 2.17 | 2.18 | 2.16 | 1.98 | 1.98 | 1.97 |
| 384 | 2 | 3.4 | 3.42 | 3.38 | 3.11 | 3.11 | 3.08 |
| 384 | 4 | 5.61 | 5.62 | 5.57 | 5.5 | 5.52 | 5.46 |
| 384 | 8 | 10.58 | 10.63 | 10.49 | 10.26 | 10.29 | 10.17 |
| 384 | 12 | 16.55 | 16.57 | 16.43 | 15.8 | 15.83 | 15.69 |
| 384 | 16 | 21.15 | 21.19 | 21.04 | 20.09 | 20.12 | 19.94 |
| 384 | 24 | 30.95 | 31 | 30.77 | 29.44 | 29.51 | 29.24 |
| 384 | 32 | 47.94 | 48.03 | 47.66 | 47.97 | 48.05 | 47.56 |
| 384 | 64 | 81.8 | 81.91 | 81.62 | 76.84 | 77.05 | 76.4 |
| 384 | 128 | 159.87 | 160.06 | 159.47 | 151.4 | 151.61 | 150.85 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 3.43 | 3.44 | 3.42 | 3.06 | 3.07 | 3.05 |
| 128 | 2 | 4.35 | 4.37 | 4.33 | 3.79 | 3.8 | 3.79 |
| 128 | 4 | 6.8 | 6.83 | 6.74 | 6.02 | 6.05 | 5.98 |
| 128 | 8 | 11 | 11.07 | 10.93 | 10.57 | 10.62 | 10.47 |
| 128 | 12 | 16.28 | 16.31 | 16.15 | 15.06 | 15.1 | 14.96 |
| 128 | 16 | 20.33 | 20.44 | 20.13 | 20.47 | 20.51 | 20.25 |
| 128 | 24 | 30.63 | 30.66 | 30.33 | 28.65 | 28.8 | 28.48 |
| 128 | 32 | 45.28 | 45.35 | 45.09 | 46.88 | 47.02 | 46.43 |
| 128 | 64 | 75.33 | 75.57 | 74.82 | 71.88 | 71.97 | 71.47 |
| 128 | 128 | 148.1 | 148.31 | 147.59 | 140.81 | 140.97 | 140.35 |
| 384 | 1 | 6.16 | 6.17 | 6.12 | 5.7 | 5.72 | 5.66 |
| 384 | 2 | 10.25 | 10.27 | 10.18 | 9.46 | 9.49 | 9.37 |
| 384 | 4 | 18.44 | 18.5 | 18.27 | 17.22 | 17.28 | 17.09 |
| 384 | 8 | 34.67 | 34.71 | 34.41 | 32.71 | 32.79 | 32.45 |
| 384 | 12 | 49.04 | 49.13 | 48.79 | 47.53 | 47.77 | 47.27 |
| 384 | 16 | 67.08 | 67.21 | 66.75 | 62.86 | 63.01 | 62.76 |
| 384 | 24 | 94.22 | 94.39 | 94.04 | 92.08 | 92.2 | 91.86 |
| 384 | 32 | 148.96 | 149.11 | 148.59 | 147.7 | 147.84 | 147.23 |
| 384 | 64 | 245.91 | 246.09 | 244.67 | 240.16 | 240.43 | 239.07 |
