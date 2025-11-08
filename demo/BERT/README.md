# BERT Inference Using TensorRT

This subfolder of the BERT TensorFlow repository, tested and maintained by NVIDIA, provides scripts to perform high-performance inference using NVIDIA TensorRT.

## Table Of Contents

- [Model Overview](#model-overview)
  - [Model Architecture](#model-architecture)
  - [TensorRT Inference Pipeline](#tensorrt-inference-pipeline)
  - [Version Info](#version-info)
- [Setup](#setup)
  - [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
  - [(Optional) Trying a different configuration](#optional-trying-a-different-configuration)
- [Advanced](#advanced)
  - [Scripts and sample code](#scripts-and-sample-code)
  - [Command-line options](#command-line-options)
  - [TensorRT inference process](#tensorrt-inference-process)
- [Accuracy](#accuracy)
  - [Evaluating Post-Training-Quantization INT8 accuracy](#evaluating-ptq-post-training-quantization-int8-accuracy-using-the-squad-dataset)
  - [Evaluating Quantization-Aware-Training INT8 accuracy](#evaluating-qat-quantization-aware-training-int8-accuracy-using-the-squad-dataset)
- [Experimental](#experimental)
  - [Variable sequence length](#variable-sequence-length)
    - [Run command lines](#run-command-lines)
  - [Sparsity with Quantization Aware Training](#sparsity-with-quantization-aware-training)
    - [Megatron-LM for Question Answering](#megatron-lm-for-question-answering)
- [Performance](#performance)
  - [Benchmarking](#benchmarking)
    - [TensorRT inference benchmark](#tensorrt-inference-benchmark)
  - [Results](#results)
    - [Inference performance: NVIDIA A100](#inference-performance-nvidia-a100-40gb)
    - [Inference performance: NVIDIA L4](#inference-performance-nvidia-l4)
    - [Inference performance: NVIDIA L40S](#inference-performance-nvidia-l40s)

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

| **Model**  | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feed-forward filter size** | **Max sequence length** | **Parameters** |
| :--------: | :---------------: | :------------------: | :-----------------: | :--------------------------: | :---------------------: | :------------: |
| BERT-Base  |    12 encoder     |         768          |         12          |           4 x 768            |           512           |      110M      |
| BERT-Large |    24 encoder     |         1024         |         16          |           4 x 1024           |           512           |      330M      |

Typically, the language model is followed by a few task-specific layers. The model used here includes layers for question answering.

### TensorRT Inference Pipeline

BERT inference consists of three main stages: tokenization, the BERT model, and finally a projection of the tokenized prediction onto the original text.
Since the tokenizer and projection of the final predictions are not nearly as compute-heavy as the model itself, we run them on the host. The BERT model is GPU-accelerated via TensorRT.

The tokenizer splits the input text into tokens that can be consumed by the model. For details on this process, see [this tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/).

To run the BERT model in TensorRT, we construct the model using TensorRT APIs and import the weights from a pre-trained TensorFlow checkpoint from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_tf_ckpt_large_qa_squad2_amp_128). Finally, a TensorRT engine is generated and serialized to the disk. The various inference scripts then load this engine for inference.

Lastly, the tokens predicted by the model are projected back to the original text to get a final result.

### Version Info

The following software version configuration has been tested:

| Software | Version |
| -------- | ------- |
| Python   | >=3.8   |
| TensorRT | 10.11   |
| CUDA     | 12.9    |

## Setup

The following section lists the requirements that you need to meet in order to run the BERT model.

### Requirements

This demo BERT application can be run within the TensorRT OSS build container. If running in a different environment, following packages are required.

- [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) - for downloading BERT checkpoints from NGC.
- PyPI Packages:
  - [cuda-python](https://pypi.org/project/cuda-python/) (tested v13.0.1)
  - [onnx](https://pypi.org/project/onnx) (tested v1.12.0)
  - [tensorflow](https://pypi.org/project/tensorflow/) (tested v2.9.1)
  - [torch](https://pypi.org/project/torch/) (tested v1.11.0)
- NVIDIA [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/), [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU.

## Quick Start Guide

1. Build and launch the container as described in [TensorRT OSS README](https://github.com/NVIDIA/TensorRT/blob/master/README.md).

   **Note:** After this point, all commands should be run from within the container.

2. Verify TensorRT installation by printing the version:
   For example:

   ```bash
   python3 -c "import tensorrt as trt; print(trt.__version__)"
   ```

3. Download the SQuAD dataset and BERT checkpoints:

   ```bash
   cd $TRT_OSSPATH/demo/BERT
   ```

   Download SQuAD v1.1 training and dev dataset.

   ```bash
   bash ./scripts/download_squad.sh
   ```

   Download Tensorflow checkpoints for BERT large model with sequence length 128, fine-tuned for SQuAD v2.0.

   ```bash
   bash scripts/download_model.sh
   ```

**Note:** Since the datasets and checkpoints are stored in the directory mounted from the host, they do _not_ need to be downloaded each time the container is launched.

**Warning:** In the event of encountering an error message stating, "Missing API key and missing Email Authentication. This command requires an API key or authentication via browser login", the recommended steps for resolution are as follows:

- Generate an API key by logging in https://ngc.nvidia.com/setup/api-key and copy the generated API key.
- Execute the command `ngc config set` in the docker and paste the copied API key into the prompt as directed.

Completing these steps should resolve the error you encountered and allow the command to proceed successfully.

4. Build a TensorRT engine. To build an engine, run the `builder.py` script. For example:

   ```bash
   mkdir -p engines && python3 builder.py -m models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/model.ckpt -o engines/bert_large_128.engine -b 1 -s 128 --fp16 -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1
   ```

   This will build an engine with a maximum batch size of 1 (`-b 1`), and sequence length of 128 (`-s 128`) using mixed precision (`--fp16`) using the BERT Large SQuAD v2 FP16 Sequence Length 128 checkpoint (`-c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1`).

5. Run inference. Two options are provided for running the model.

   a. `inference.py` script
   This script accepts a passage and question and then runs the engine to generate an answer.
   For example:

   ```bash
   python3 inference.py -e engines/bert_large_128.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/vocab.txt
   ```

   b. `inference.ipynb` Jupyter Notebook
   The Jupyter Notebook includes a passage and various example questions and allows you to interactively make modifications and see the outcome.
   To launch the Jupyter Notebook from inside the container, run:

   ```bash
   jupyter notebook --ip 0.0.0.0 inference.ipynb
   ```

   Then, use your browser to open the link displayed. The link should look similar to: `http://127.0.0.1:8888/?token=<TOKEN>`

6. Run inference with CUDA Graph support.

   A separate python `inference_c.py` script is provided to run inference with CUDA Graph support. This is necessary since CUDA Graph is only supported through CUDA C/C++ APIs. The `inference_c.py` script uses pybind11 to interface with C/C++ for CUDA graph capturing and launching. The cmdline interface is the same as `inference.py` except for an extra `--enable-graph` option.

   ```bash
   mkdir -p build; pushd build
   cmake .. -DPYTHON_EXECUTABLE=$(which python)
   make -j
   popd
   python3 inference_c.py -e engines/bert_large_128.engine --enable-graph -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/vocab.txt
   ```

   A separate C/C++ inference benchmark executable `perf` (compiled from `perf.cpp`) is provided to run inference benchmarks with CUDA Graph. The cmdline interface is the same as `perf.py` except for an extra `--enable_graph` option.

   ```bash
   build/perf -e engines/bert_large_128.engine -b 1 -s 128 -w 100 -i 1000 --enable_graph
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

**Note:** In the builder scripts (`builder.py` and `builder_varseqlen.py`), the options `--use-deprecated-plugins` and `--use-v3-plugins` toggle the underlying implementation of the plugins used in demoBERT. They are mutually exclusive, and enabling either should not affect functionality, or performance. The `--use-deprecated-plugins` uses plugin versions that inherit from `IPluginV2DynamicExt`, while `--use-v3-plugins` uses plugin versions that inherit from `IPluginV3` classes.
If unspecified, `--use-deprecated-plugins` is used by default.

**Additional Note:** Using `--use-v3-plugins` is not recommended on Blackwell platforms (See [Platform support section](#hardware-platform-support)). Prefer the default path instead (`--use-deprecated-plugins`).

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

2.  Build an engine:

    **Turing and Ampere GPUs**

    ```bash
    # QKVToContextPlugin and SkipLayerNormPlugin supported with INT8 I/O. To enable, use -imh and -iln builder flags respectively.
    mkdir -p engines && python3 builder.py -m models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/model.ckpt -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100 -iln -imh
    ```

    **Xavier GPU**

    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p engines && python3 builder.py -m models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/model.ckpt -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100 -iln
    ```

    **Volta GPU**

    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p engines && python3 builder.py -m models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/model.ckpt -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 --squad-json ./squad/train-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt --calib-num 100
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), calibration dataset squad (`--squad-json ./squad/train-v1.1.json`), calibration sentences number 100 (`--calib-num 100`), and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3.  Run inference using the squad dataset, and evaluate the F1 score and exact match score:
    ```bash
    python3 inference.py -e engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```

### Evaluating QAT (quantization aware training) Int8 Accuracy Using The SQuAD Dataset

1.  Download checkpoint for BERT Large FP16 SQuAD v1.1 model with sequence length of 384:

    ```bash
    bash scripts/download_model.sh pyt v1_1
    ```

2.  Build an engine:

    **Turing and Ampere GPUs**

    ```bash
    # QKVToContextPlugin and SkipLayerNormPlugin supported with INT8 I/O. To enable, use -imh and -iln builder flags respectively.
    mkdir -p engines && python3 builder.py -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln -imh
    ```

    **Xavier GPU**

    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p engines && python3 builder.py -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln
    ```

    **Volta GPU**

    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p engines && python3 builder.py -o engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3.  Run inference using the squad dataset, and evaluate the F1 score and exact match score:

    ```bash
    python3 inference.py -e engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```

## Experimental

### Variable sequence length

In our prior implementation, we used inputs padded to max length along with corresponding input masks to handle variable sequence length inputs in a batch. The padding results in some wasted computations which can be avoided by handling variable sequence length inputs natively. Now we have a new approach called the variable sequence length method. By concatenating each input id into a single long input id, and concatenating each input segment id into a single long segment id, TensorRT can know the exact starts and ends by providing an extra sequence length buffer that contains the start and end positions of each sequence. Now we can eliminate the wasted computation in the input paddings.

Note this is an experimental feature because we only support Xavier+ GPUs, also there is neither FP32 support nor INT8 PTQ calibration.

1.  Download checkpoint for BERT Large FP16 SQuAD v1.1 model with sequence length of 384:

    ```bash
    bash scripts/download_model.sh pyt v1_1
    ```

2.  Build an engine:

    **FP16 engine**

    ```bash
    mkdir -p engines && python3 builder_varseqlen.py -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -o engines/bert_varseq_fp16.engine -b 1 -s 64 --fp16 -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 64 (`-s 64`) using FP16 precision computation where possible (`--fp16`).

    **INT8 engine**

    ```bash
    mkdir -p engines && python3 builder_varseqlen.py -x models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -o engines/bert_varseq_int8.engine -b 1 -s 256 --int8 --fp16 -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 256 (`-s 256`) using INT8 precision computation where possible (`--int8`).

3.  Run inference

    Evaluate the F1 score and exact match score using the squad dataset:

    ```bash
    python3 inference_varseqlen.py -e engines/bert_varseq_int8.engine -s 256 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```

    Run the quesion and answer mode:

    ```bash
    python3 inference_varseqlen.py -e engines/bert_varseq_int8.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -s 256
    ```

4.  Collect performance data

    ```bash
    python3 perf_varseqlen.py -e engines/bert_varseq_int8.engine -b 1 -s 256
    ```

    This will collect performance data run use batch size 1 (`-b 1`) and sequence length of 256 (`-s 256`).

5.  Collect performance data with CUDA graph enabled

    We can use the same `inference_c.py` and `build/perf` to collect performance data with cuda graph enabled. The command line is the same as run without variable sequence length.

### Sparsity with Quantization Aware Training

Fine-grained 2:4 structured sparsity support introduced in NVIDIA Ampere GPUs can produce significant performance gains in BERT inference. The network is first trained using dense weights, then fine-grained structured pruning is applied, and finally the remaining non-zero weights are fine-tuned with additional training steps. This method results in virtually no loss in inferencing accuracy.

Using INT8 precision with quantization scales obtained from Post-Training Quantization (PTQ) can produce additional performance gains, but may also result in accuracy loss. Alternatively, for PyTorch-trained models, NVIDIA [PyTorch-Quantization toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) can be leveraged to perform quantized fine tuning (a.k.a. Quantization Aware Training or QAT) and generate the INT8 quantization scales as part of training. This generally results in higher accuracy compared to PTQ.

To demonstrate the potential speedups from these optimizations in demoBERT, we provide the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) transformer model finetuned for SQuAD 2.0 task with sparsity and quantization.

The sparse weights are generated by finetuning with INT8 Quantization Aware Training recipe. This feature can be used with the fixed or variable sequence length implementations by passing in `-sp` flag to demoBERT builder.

#### Megatron-LM for Question Answering

##### Example: Megatron-LM Large SQuAD v2.0 with sparse weights for sequence length 384

**Build the TensorRT engine**:

Options specified:

- `--megatron` : assume Megatron style residuals instead of vanilla BERT.
- `--pickle` : specify a pickle file containing the PyTorch statedict corresponding to fine-tuned Megatron model.
- `-sp` : enable sparsity during engine optimization and treat the weights as sparse.
- `--int8 --il` : enable int8 tactics/plugins with interleaving.

```bash
bash ./scripts/download_model.sh 384 v1_1 # BERT-large model checkpoint fine-tuned for SQuAD 1.1
bash ./scripts/download_model.sh pyt megatron-large int8-qat sparse # Megatron-LM model weights
export CKPT_PATH=models/fine-tuned/bert_pyt_statedict_megatron_sparse_int8qat_v21.03.0/bert_pyt_statedict_megatron_sparse_int8_qat
mkdir -p engines && python3 builder_varseqlen.py -c models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1 -b 1 -s 384 -o engines/megatron_large_seqlen384_int8qat_sparse.engine --fp16 --int8 --strict -il --megatron --pickle $CKPT_PATH -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -sp
```

**Ask a question**:

```bash
python3 inference_varseqlen.py -e engines/megatron_large_seqlen384_int8qat_sparse.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -s 256
```

**Evaluate F1 score**:

```bash
python3 inference_varseqlen.py -e engines/megatron_large_seqlen384_int8qat_sparse.engine -s 384 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
```

Expected output:

```
&&&& PASSED TensorRT BERT Squad Accuracy matches reference.
{"exact_match": 84.03973509933775, "f1": 90.88667129897755}
```

## Performance

### Benchmarking

The following section shows how to run the inference benchmarks for BERT.

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

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA A100 (40G).

##### BERT base

| Sequence Length | Batch Size | INT8 Latency (ms) |                 |         | FP16 Latency (ms) |                 |         |
| --------------- | ---------- | ----------------- | --------------- | ------- | ----------------- | --------------- | ------- |
|                 |            | 95th Percentile   | 99th Percentile | Average | 95th Percentile   | 99th Percentile | Average |
| 128             | 1          | 0.67              | 0.67            | 0.54    | 0.62              | 0.80            | 0.62    |
| 128             | 2          | 0.76              | 0.76            | 0.60    | 0.92              | 0.92            | 0.73    |
| 128             | 4          | 0.73              | 0.93            | 0.73    | 0.93              | 0.93            | 0.93    |
| 128             | 8          | 0.94              | 1.21            | 0.95    | 1.31              | 1.31            | 1.31    |
| 128             | 12         | 1.20              | 1.20            | 1.20    | 1.72              | 2.20            | 1.72    |
| 128             | 16         | 1.34              | 1.34            | 1.34    | 2.07              | 2.08            | 2.05    |
| 128             | 24         | 1.82              | 1.82            | 1.82    | 3.02              | 3.08            | 3.01    |
| 128             | 32         | 2.23              | 2.24            | 2.23    | 3.89              | 3.91            | 3.85    |
| 128             | 64         | 4.16              | 4.16            | 4.12    | 7.57              | 7.63            | 7.55    |
| 128             | 128        | 8.07              | 8.09            | 8.02    | 15.23             | 15.24           | 15.15   |
| 384             | 1          | 1.14              | 1.46            | 1.14    | 1.25              | 1.61            | 1.26    |
| 384             | 2          | 1.32              | 1.32            | 1.32    | 1.55              | 1.55            | 1.55    |
| 384             | 4          | 1.66              | 1.66            | 1.66    | 2.12              | 2.12            | 2.12    |
| 384             | 8          | 2.20              | 2.21            | 2.20    | 3.34              | 3.36            | 3.31    |
| 384             | 12         | 3.31              | 3.31            | 3.31    | 4.78              | 4.82            | 4.77    |
| 384             | 16         | 4.00              | 4.00            | 4.00    | 6.38              | 6.40            | 6.33    |
| 384             | 24         | 5.70              | 5.70            | 5.70    | 9.31              | 9.31            | 9.22    |
| 384             | 32         | 7.64              | 7.64            | 7.64    | 12.90             | 12.90           | 12.79   |
| 384             | 64         | 14.87             | 14.91           | 14.74   | 24.96             | 25.19           | 24.74   |
| 384             | 128        | 29.01             | 29.02           | 28.74   | 49.05             | 49.28           | 48.64   |

##### BERT large

| Sequence Length | Batch Size | INT8 Latency (ms) |                 |         | FP16 Latency (ms) |                 |         |
| --------------- | ---------- | ----------------- | --------------- | ------- | ----------------- | --------------- | ------- |
|                 |            | 95th Percentile   | 99th Percentile | Average | 95th Percentile   | 99th Percentile | Average |
| 128             | 1          | 1.23              | 1.23            | 1.23    | 1.54              | 1.55            | 1.54    |
| 128             | 2          | 1.42              | 1.42            | 1.42    | 1.82              | 2.02            | 1.82    |
| 128             | 4          | 1.79              | 1.79            | 1.78    | 2.52              | 2.53            | 2.52    |
| 128             | 8          | 2.64              | 2.65            | 2.64    | 3.93              | 3.94            | 3.89    |
| 128             | 12         | 3.11              | 3.11            | 3.11    | 5.03              | 5.07            | 5.00    |
| 128             | 16         | 4.09              | 4.09            | 4.08    | 6.93              | 6.94            | 6.86    |
| 128             | 24         | 5.28              | 5.28            | 5.27    | 9.70              | 9.70            | 9.65    |
| 128             | 32         | 7.00              | 7.02            | 6.95    | 12.95             | 12.96           | 12.83   |
| 128             | 64         | 12.85             | 12.89           | 12.74   | 24.85             | 25.06           | 24.63   |
| 128             | 128        | 25.07             | 25.08           | 24.99   | 49.15             | 49.42           | 48.69   |
| 384             | 1          | 2.55              | 2.55            | 2.55    | 2.96              | 2.96            | 2.96    |
| 384             | 2          | 3.03              | 3.03            | 3.03    | 3.90              | 3.90            | 3.89    |
| 384             | 4          | 4.01              | 4.01            | 4.01    | 5.73              | 5.79            | 5.67    |
| 384             | 8          | 7.16              | 7.16            | 7.16    | 11.12             | 11.16           | 11.01   |
| 384             | 12         | 9.14              | 9.14            | 9.13    | 15.31             | 15.45           | 15.27   |
| 384             | 16         | 12.28             | 12.28           | 12.28   | 20.99             | 20.99           | 20.92   |
| 384             | 24         | 17.67             | 17.72           | 17.57   | 30.75             | 31.03           | 30.66   |
| 384             | 32         | 23.29             | 23.31           | 23.06   | 41.01             | 41.26           | 40.61   |
| 384             | 64         | 44.96             | 45.30           | 44.83   | 79.97             | 80.27           | 79.26   |
| 384             | 128        | 87.99             | 88.02           | 87.69   | 156.51            | 156.99          | 155.47  |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |                 |         |
| --------------- | ---------- | --------------------- | --------------- | ------- |
|                 |            | 95th Percentile       | 99th Percentile | Average |
| 128             | 1          | 1.13                  | 1.44            | 1.14    |
| 128             | 2          | 1.37                  | 1.37            | 1.37    |
| 128             | 4          | 1.78                  | 1.78            | 1.77    |
| 128             | 8          | 2.45                  | 2.46            | 2.45    |
| 128             | 12         | 3.11                  | 3.12            | 3.10    |
| 128             | 16         | 3.91                  | 3.91            | 3.90    |
| 128             | 24         | 4.89                  | 4.89            | 4.88    |
| 128             | 32         | 6.96                  | 6.97            | 6.91    |
| 128             | 64         | 11.64                 | 11.65           | 11.63   |
| 128             | 128        | 21.82                 | 21.83           | 21.69   |
| 384             | 1          | 1.69                  | 1.69            | 1.69    |
| 384             | 2          | 2.21                  | 2.22            | 2.21    |
| 384             | 4          | 3.63                  | 3.63            | 3.62    |
| 384             | 8          | 5.72                  | 5.72            | 5.71    |
| 384             | 12         | 8.38                  | 8.39            | 8.37    |
| 384             | 16         | 10.35                 | 10.35           | 10.34   |
| 384             | 24         | 14.49                 | 14.49           | 14.48   |
| 384             | 32         | 18.75                 | 18.81           | 18.73   |
| 384             | 64         | 36.38                 | 36.41           | 36.11   |
| 384             | 128        | 69.82                 | 69.95           | 69.34   |

#### Inference performance: NVIDIA A30

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA A30.

##### BERT base

| Sequence Length | Batch Size | INT8 Latency (ms) |                 |         | FP16 Latency (ms) |                 |         |
| --------------- | ---------- | ----------------- | --------------- | ------- | ----------------- | --------------- | ------- |
|                 |            | 95th Percentile   | 99th Percentile | Average | 95th Percentile   | 99th Percentile | Average |
| 128             | 1          | 0.62              | 0.62            | 0.61    | 1.01              | 1.02            | 1.00    |
| 128             | 2          | 0.79              | 0.80            | 0.77    | 1.33              | 1.35            | 1.31    |
| 128             | 4          | 1.16              | 1.16            | 1.13    | 2.23              | 2.23            | 2.16    |
| 128             | 8          | 1.93              | 1.98            | 1.91    | 3.70              | 3.83            | 3.69    |
| 128             | 12         | 2.69              | 2.69            | 2.63    | 5.42              | 5.46            | 5.36    |
| 128             | 16         | 3.38              | 3.39            | 3.32    | 6.77              | 6.78            | 6.71    |
| 128             | 24         | 4.87              | 4.87            | 4.77    | 10.72             | 10.81           | 10.56   |
| 128             | 32         | 6.22              | 6.35            | 6.18    | 14.13             | 14.14           | 13.97   |
| 128             | 64         | 13.69             | 13.85           | 13.56   | 31.28             | 31.69           | 31.05   |
| 128             | 128        | 30.49             | 30.72           | 29.90   | 69.99             | 70.38           | 68.61   |
| 384             | 1          | 1.31              | 1.31            | 1.30    | 2.10              | 2.10            | 2.09    |
| 384             | 2          | 1.85              | 1.86            | 1.85    | 3.19              | 3.21            | 3.14    |
| 384             | 4          | 3.00              | 3.00            | 2.94    | 5.77              | 5.89            | 5.74    |
| 384             | 8          | 5.58              | 5.60            | 5.48    | 11.49             | 11.59           | 11.38   |
| 384             | 12         | 8.22              | 8.37            | 8.13    | 17.39             | 17.40           | 17.16   |
| 384             | 16         | 10.98             | 10.99           | 10.89   | 23.38             | 23.78           | 23.02   |
| 384             | 24         | 17.33             | 17.47           | 17.09   | 38.54             | 39.55           | 37.57   |
| 384             | 32         | 23.82             | 24.18           | 23.56   | 51.12             | 51.24           | 50.62   |
| 384             | 64         | 50.08             | 50.28           | 49.10   | 105.60            | 106.08          | 104.59  |
| 384             | 128        | 113.95            | 114.53          | 112.15  | 209.55            | 209.93          | 208.35  |

##### BERT large

| Sequence Length | Batch Size | INT8 Latency (ms) |                 |         | FP16 Latency (ms) |                 |         |
| --------------- | ---------- | ----------------- | --------------- | ------- | ----------------- | --------------- | ------- |
|                 |            | 95th Percentile   | 99th Percentile | Average | 95th Percentile   | 99th Percentile | Average |
| 128             | 1          | 1.80              | 1.80            | 1.78    | 3.12              | 3.12            | 3.10    |
| 128             | 2          | 2.51              | 2.52            | 2.45    | 4.36              | 4.38            | 4.34    |
| 128             | 4          | 3.70              | 3.72            | 3.60    | 6.91              | 6.94            | 6.82    |
| 128             | 8          | 6.40              | 6.41            | 6.30    | 12.81             | 12.96           | 12.76   |
| 128             | 12         | 8.53              | 8.60            | 8.36    | 18.79             | 18.96           | 18.43   |
| 128             | 16         | 11.25             | 11.34           | 11.18   | 25.61             | 25.85           | 25.34   |
| 128             | 24         | 16.25             | 16.28           | 16.01   | 36.21             | 36.22           | 35.95   |
| 128             | 32         | 21.65             | 21.68           | 21.37   | 49.57             | 49.76           | 49.17   |
| 128             | 64         | 44.98             | 45.44           | 44.57   | 107.87            | 108.20          | 106.77  |
| 128             | 128        | 93.97             | 94.63           | 93.01   | 216.03            | 216.54          | 214.69  |
| 384             | 1          | 3.47              | 3.48            | 3.45    | 6.64              | 6.75            | 6.43    |
| 384             | 2          | 5.57              | 5.58            | 5.46    | 10.63             | 10.65           | 10.49   |
| 384             | 4          | 9.79              | 9.93            | 9.62    | 20.78             | 21.19           | 20.38   |
| 384             | 8          | 18.38             | 18.39           | 18.22   | 39.85             | 40.17           | 38.38   |
| 384             | 12         | 26.50             | 26.74           | 26.39   | 61.30             | 61.76           | 59.94   |
| 384             | 16         | 37.19             | 37.48           | 36.70   | 81.72             | 82.15           | 80.66   |
| 384             | 24         | 55.13             | 55.69           | 54.64   | 131.37            | 131.61          | 130.29  |
| 384             | 32         | 76.86             | 77.41           | 75.98   | 166.22            | 166.56          | 165.16  |
| 384             | 64         | 165.08            | 165.56          | 163.82  | 344.18            | 344.61          | 342.97  |
| 384             | 128        | 334.73            | 335.97          | 332.16  | 670.67            | 671.67          | 668.80  |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |                 |         |
| --------------- | ---------- | --------------------- | --------------- | ------- |
|                 |            | 95th Percentile       | 99th Percentile | Average |
| 128             | 1          | 1.51                  | 1.51            | 1.49    |
| 128             | 2          | 2.07                  | 2.09            | 2.03    |
| 128             | 4          | 2.98                  | 3.02            | 2.92    |
| 128             | 8          | 5.06                  | 5.07            | 5.05    |
| 128             | 12         | 6.70                  | 6.77            | 6.63    |
| 128             | 16         | 8.81                  | 8.82            | 8.74    |
| 128             | 24         | 13.18                 | 13.19           | 13.09   |
| 128             | 32         | 17.43                 | 17.44           | 17.34   |
| 128             | 64         | 36.26                 | 36.70           | 35.86   |
| 128             | 128        | 79.70                 | 79.88           | 79.06   |
| 384             | 1          | 2.80                  | 2.81            | 2.75    |
| 384             | 2          | 4.21                  | 4.21            | 4.15    |
| 384             | 4          | 7.64                  | 7.66            | 7.53    |
| 384             | 8          | 14.96                 | 14.98           | 14.83   |
| 384             | 12         | 21.62                 | 21.66           | 21.46   |
| 384             | 16         | 28.40                 | 28.57           | 28.31   |
| 384             | 24         | 45.11                 | 45.45           | 44.78   |
| 384             | 32         | 60.86                 | 61.08           | 59.88   |
| 384             | 64         | 126.53                | 126.80          | 126.06  |
| 384             | 128        | 255.35                | 256.27          | 253.63  |

### Inference Performance NVIDIA L40S

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA L40S.

##### BERT base

| Sequence Length | Batch Size | INT8 Latency (ms) |                 |         | FP16 Latency (ms) |                 |         |
| --------------- | ---------- | ----------------- | --------------- | ------- | ----------------- | --------------- | ------- |
|                 |            | 95th Percentile   | 99th Percentile | Average | 95th Percentile   | 99th Percentile | Average |
| 128             | 1          | 0.34              | 0.34            | 0.34    | 0.48              | 0.48            | 0.48    |
| 128             | 2          | 0.41              | 0.41            | 0.41    | 0.56              | 0.56            | 0.55    |
| 128             | 4          | 0.50              | 0.51            | 0.50    | 0.77              | 0.78            | 0.77    |
| 128             | 8          | 0.68              | 0.68            | 0.67    | 1.26              | 1.26            | 1.25    |
| 128             | 12         | 0.91              | 0.91            | 0.91    | 1.69              | 1.69            | 1.68    |
| 128             | 16         | 1.11              | 1.11            | 1.11    | 2.24              | 2.24            | 2.23    |
| 128             | 24         | 1.46              | 1.46            | 1.46    | 3.18              | 3.19            | 3.18    |
| 128             | 32         | 1.82              | 1.82            | 1.81    | 3.94              | 3.94            | 3.93    |
| 128             | 64         | 3.44              | 3.44            | 3.42    | 7.98              | 8.08            | 7.90    |
| 128             | 128        | 7.25              | 7.29            | 7.20    | 17.35             | 17.40           | 17.13   |
| 384             | 1          | 0.73              | 0.73            | 0.73    | 1.04              | 1.04            | 1.03    |
| 384             | 2          | 0.88              | 0.88            | 0.88    | 1.35              | 1.35            | 1.35    |
| 384             | 4          | 1.17              | 1.17            | 1.17    | 2.14              | 2.14            | 2.13    |
| 384             | 8          | 1.70              | 1.71            | 1.69    | 3.47              | 3.47            | 3.46    |
| 384             | 12         | 2.72              | 2.72            | 2.72    | 5.08              | 5.09            | 5.06    |
| 384             | 16         | 3.26              | 3.26            | 3.24    | 7.18              | 7.19            | 7.15    |
| 384             | 24         | 4.94              | 4.94            | 4.89    | 9.98              | 10.00           | 9.92    |
| 384             | 32         | 6.11              | 6.13            | 6.09    | 13.35             | 13.38           | 13.25   |
| 384             | 64         | 12.96             | 13.00           | 12.84   | 28.93             | 29.37           | 28.41   |
| 384             | 128        | 27.22             | 27.36           | 26.87   | 59.55             | 59.91           | 58.44   |

##### BERT large

| Sequence Length | Batch Size | INT8 Latency (ms) |                 |         | FP16 Latency (ms) |                 |         |
| --------------- | ---------- | ----------------- | --------------- | ------- | ----------------- | --------------- | ------- |
|                 |            | 95th Percentile   | 99th Percentile | Average | 95th Percentile   | 99th Percentile | Average |
| 128             | 1          | 0.89              | 0.89            | 0.89    | 1.30              | 1.30            | 1.30    |
| 128             | 2          | 0.98              | 0.98            | 0.98    | 1.45              | 1.46            | 1.45    |
| 128             | 4          | 1.35              | 1.35            | 1.34    | 2.32              | 2.32            | 2.31    |
| 128             | 8          | 1.93              | 1.95            | 1.92    | 3.59              | 3.60            | 3.58    |
| 128             | 12         | 2.73              | 2.73            | 2.72    | 5.70              | 5.71            | 5.63    |
| 128             | 16         | 3.19              | 3.21            | 3.17    | 6.48              | 6.49            | 6.45    |
| 128             | 24         | 4.50              | 4.53            | 4.48    | 9.89              | 9.90            | 9.81    |
| 128             | 32         | 5.66              | 5.68            | 5.62    | 12.26             | 12.30           | 12.16   |
| 128             | 64         | 11.42             | 11.43           | 11.30   | 27.40             | 27.60           | 27.16   |
| 128             | 128        | 24.68             | 24.70           | 24.36   | 61.49             | 61.76           | 60.81   |
| 384             | 1          | 1.68              | 1.68            | 1.68    | 2.73              | 2.73            | 2.73    |
| 384             | 2          | 2.28              | 2.28            | 2.27    | 3.83              | 3.83            | 3.82    |
| 384             | 4          | 3.28              | 3.28            | 3.26    | 6.26              | 6.26            | 6.24    |
| 384             | 8          | 4.97              | 4.98            | 4.95    | 10.32             | 10.33           | 10.30   |
| 384             | 12         | 7.89              | 7.89            | 7.86    | 17.49             | 17.50           | 17.43   |
| 384             | 16         | 9.47              | 9.49            | 9.44    | 21.50             | 21.62           | 21.24   |
| 384             | 24         | 14.64             | 14.66           | 14.54   | 33.26             | 33.30           | 33.01   |
| 384             | 32         | 19.20             | 19.37           | 18.97   | 44.56             | 44.69           | 43.95   |
| 384             | 64         | 42.15             | 42.38           | 41.56   | 98.89             | 99.28           | 97.19   |
| 384             | 128        | 84.15             | 84.40           | 83.34   | 196.98            | 197.83          | 194.18  |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |                 |         |
| --------------- | ---------- | --------------------- | --------------- | ------- |
|                 |            | 95th Percentile       | 99th Percentile | Average |
| 128             | 1          | 0.76                  | 0.76            | 0.76    |
| 128             | 2          | 0.90                  | 0.90            | 0.90    |
| 128             | 4          | 1.13                  | 1.13            | 1.13    |
| 128             | 8          | 1.71                  | 1.71            | 1.71    |
| 128             | 12         | 2.26                  | 2.26            | 2.25    |
| 128             | 16         | 2.72                  | 2.73            | 2.72    |
| 128             | 24         | 4.44                  | 4.45            | 4.43    |
| 128             | 32         | 5.07                  | 5.11            | 5.04    |
| 128             | 64         | 10.06                 | 10.09           | 9.97    |
| 128             | 128        | 20.42                 | 20.46           | 20.30   |
| 384             | 1          | 1.13                  | 1.13            | 1.13    |
| 384             | 2          | 1.63                  | 1.65            | 1.62    |
| 384             | 4          | 2.52                  | 2.53            | 2.51    |
| 384             | 8          | 4.93                  | 4.94            | 4.90    |
| 384             | 12         | 6.47                  | 6.47            | 6.45    |
| 384             | 16         | 8.41                  | 8.42            | 8.36    |
| 384             | 24         | 12.52                 | 12.53           | 12.44   |
| 384             | 32         | 16.66                 | 16.72           | 16.57   |
| 384             | 64         | 34.12                 | 34.22           | 33.81   |
| 384             | 128        | 71.98                 | 72.13           | 71.52   |

## Hardware Platform Support

The scripts call TensorRT Plugins underneath, whose kernel optimizations depend on the NVIDIA GPU Architecture.
The compute capability of an NVIDIA GPU can be found out using the `nvidia-smi` commandline utility. One can execute the following command in the terminal to find out the compute capability of the GPU.

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Currently, this demo is supported on the following compute capabilities. This list is subject to change as new architectures are released.

- Volta architecture - 7.2, 7.5
- Ampere architecture - 8.0, 8.6, 8.7, 8.9
- Hopper architecture - 9.0 (since October 2022)
- Blackwell architecture - 10.0, 12.0 (since Jan 2025). Not recommended with `--use-v3-plugins` option.
