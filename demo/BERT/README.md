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
- [Experimental](#experimental)
  * [Variable sequence length](#variable-sequence-length)
      * [Run command lines](#run-command-lines)
  * [Sparsity with Quantization Aware Training](#sparsity-with-quantization-aware-training)
      * [Megatron-LM for Question Answering](#megatron-lm-for-question-answering)
- [Performance](#performance)
  * [Benchmarking](#benchmarking)
       * [TensorRT inference benchmark](#tensorrt-inference-benchmark)
  * [Results](#results)
    * [Inference performance: NVIDIA A100](#inference-performance-nvidia-a100-40gb)
    * [Inference performance: NVIDIA A30](#inference-performance-nvidia-a30)
    * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4-16gb)


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
|Python|>=3.6|
|TensorRT|8.4.3|
|CUDA|11.6|

## Setup

The following section lists the requirements that you need to meet in order to run the BERT model.

### Requirements

This demo BERT application can be run within the TensorRT OSS build container. If running in a different environment, following packages are required.

* [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) - for downloading BERT checkpoints from NGC.
* PyPI Packages:
  * [pycuda](https://pypi.org/project/pycuda/) (tested v2019.1.2)
  * [onnx](https://pypi.org/project/onnx) (tested v1.12.0)
  * [tensorflow](https://pypi.org/project/tensorflow/) (tested v2.9.1)
  * [torch](https://pypi.org/project/torch/) (tested v1.11.0)
* NVIDIA [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/), [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU.


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

**Note:** Since the datasets and checkpoints are stored in the directory mounted from the host, they do *not* need to be downloaded each time the container is launched. 

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

    A separate python `inference_c.py` script is provided to run inference with CUDA Graph support. This is necessary since CUDA Graph is only supported through CUDA C/C++ APIs, not pyCUDA. The `inference_c.py` script uses pybind11 to interface with C/C++ for CUDA graph capturing and launching. The cmdline interface is the same as `inference.py` except for an extra `--enable-graph` option.
    
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

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:
    ```bash
    python3 inference.py -e engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
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

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:

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

2. Build an engine:

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

3. Run inference 

    Evaluate the F1 score and exact match score using the squad dataset:
    
    ```bash
    python3 inference_varseqlen.py -e engines/bert_varseq_int8.engine -s 256 -sq ./squad/dev-v1.1.json -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```

    Run the quesion and answer mode:

    ```bash
    python3 inference_varseqlen.py -e engines/bert_varseq_int8.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_384_v19.03.1/vocab.txt -s 256
    ```

4. Collect performance data

    ```bash
    python3 perf_varseqlen.py -e engines/bert_varseq_int8.engine -b 1 -s 256
    ```

    This will collect performance data run use batch size 1 (`-b 1`) and sequence length of 256 (`-s 256`). 

5. Collect performance data with CUDA graph enabled

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
* `--megatron`  : assume Megatron style residuals instead of vanilla BERT.
* `--pickle`    : specify a pickle file containing the PyTorch statedict corresponding to fine-tuned Megatron model.
* `-sp`         : enable sparsity during engine optimization and treat the weights as sparse.
* `--int8 --il` : enable int8 tactics/plugins with interleaving.

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

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.71 | 0.71 | 0.58 | 0.83 | 0.83 | 0.67 |
| 128 | 2 | 0.78 | 0.78 | 0.62 | 0.83 | 1.02 | 0.83 |
| 128 | 4 | 0.85 | 0.85 | 0.77 | 1.01 | 1.29 | 1.02 |
| 128 | 8 | 0.95 | 0.95 | 0.95 | 1.38 | 1.77 | 1.38 |
| 128 | 12 | 1.24 | 1.24 | 1.22 | 1.87 | 1.88 | 1.87 |
| 128 | 16 | 1.37 | 1.37 | 1.37 | 2.20 | 2.23 | 2.18 |
| 128 | 24 | 1.84 | 1.84 | 1.84 | 3.37 | 3.38 | 3.34 |
| 128 | 32 | 2.28 | 2.28 | 2.27 | 4.12 | 4.13 | 4.10 |
| 128 | 64 | 4.24 | 4.28 | 4.20 | 8.12 | 8.12 | 8.05 |
| 128 | 128 | 8.24 | 8.24 | 8.18 | 16.24 | 16.30 | 16.14 |
| 384 | 1 | 1.15 | 1.47 | 1.15 | 1.31 | 1.31 | 1.31 |
| 384 | 2 | 1.33 | 1.33 | 1.33 | 1.64 | 1.64 | 1.64 |
| 384 | 4 | 1.69 | 1.70 | 1.69 | 2.31 | 2.32 | 2.31 |
| 384 | 8 | 2.22 | 2.24 | 2.22 | 3.63 | 3.64 | 3.62 |
| 384 | 12 | 3.36 | 3.36 | 3.36 | 5.07 | 5.11 | 5.03 |
| 384 | 16 | 4.10 | 4.10 | 4.10 | 6.74 | 6.80 | 6.72 |
| 384 | 24 | 5.80 | 5.80 | 5.80 | 9.89 | 9.95 | 9.79 |
| 384 | 32 | 7.80 | 7.82 | 7.74 | 13.63 | 13.64 | 13.51 |
| 384 | 64 | 15.25 | 15.25 | 15.16 | 26.18 | 26.31 | 26.11 |
| 384 | 128 | 29.47 | 29.48 | 29.34 | 51.85 | 51.96 | 51.26 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.24 | 1.57 | 1.25 | 1.64 | 1.64 | 1.64 |
| 128 | 2 | 1.47 | 1.85 | 1.47 | 2.05 | 2.05 | 2.05 |
| 128 | 4 | 1.82 | 1.82 | 1.82 | 2.98 | 2.98 | 2.97 |
| 128 | 8 | 2.71 | 2.72 | 2.71 | 4.52 | 4.52 | 4.51 |
| 128 | 12 | 3.10 | 3.11 | 3.10 | 5.30 | 5.32 | 5.26 |
| 128 | 16 | 4.02 | 4.02 | 4.02 | 7.79 | 7.81 | 7.75 |
| 128 | 24 | 5.36 | 5.38 | 5.32 | 10.21 | 10.23 | 10.10 |
| 128 | 32 | 7.11 | 7.11 | 7.09 | 13.92 | 13.92 | 13.85 |
| 128 | 64 | 13.24 | 13.29 | 13.09 | 26.35 | 26.52 | 26.20 |
| 128 | 128 | 25.78 | 25.81 | 25.45 | 52.00 | 52.30 | 51.61 |
| 384 | 1 | 2.81 | 2.81 | 2.81 | 3.13 | 3.13 | 3.12 |
| 384 | 2 | 3.06 | 3.06 | 3.06 | 4.16 | 4.16 | 4.15 |
| 384 | 4 | 4.03 | 4.04 | 4.03 | 6.01 | 6.01 | 5.96 |
| 384 | 8 | 7.16 | 7.18 | 7.16 | 11.60 | 11.68 | 11.53 |
| 384 | 12 | 9.25 | 9.34 | 9.25 | 16.33 | 16.50 | 16.22 |
| 384 | 16 | 12.51 | 12.52 | 12.51 | 22.00 | 22.00 | 21.92 |
| 384 | 24 | 18.05 | 18.10 | 17.88 | 32.64 | 32.82 | 32.43 |
| 384 | 32 | 23.42 | 23.49 | 23.17 | 43.10 | 43.10 | 43.01 |
| 384 | 64 | 45.62 | 45.63 | 45.40 | 84.26 | 84.56 | 83.63 |
| 384 | 128 | 89.51 | 89.55 | 89.01 | 164.56 | 164.95 | 163.70 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.17 | 1.18 | 1.14 |
| 128 | 2 | 1.43 | 1.82 | 1.43 |
| 128 | 4 | 1.90 | 1.90 | 1.90 |
| 128 | 8 | 3.08 | 3.08 | 3.05 |
| 128 | 12 | 3.36 | 3.36 | 3.36 |
| 128 | 16 | 4.42 | 4.42 | 4.42 |
| 128 | 24 | 6.01 | 6.01 | 6.00 |
| 128 | 32 | 7.75 | 7.76 | 7.75 |
| 128 | 64 | 13.91 | 14.04 | 13.81 |
| 128 | 128 | 27.11 | 27.12 | 26.85 |
| 384 | 1 | 1.71 | 1.71 | 1.71 |
| 384 | 2 | 2.37 | 2.37 | 2.37 |
| 384 | 4 | 3.92 | 3.92 | 3.92 |
| 384 | 8 | 6.80 | 6.80 | 6.80 |
| 384 | 12 | 9.02 | 9.03 | 9.02 |
| 384 | 16 | 12.15 | 12.16 | 12.15 |
| 384 | 24 | 17.54 | 17.55 | 17.41 |
| 384 | 32 | 22.94 | 22.96 | 22.71 |
| 384 | 64 | 43.88 | 43.90 | 43.61 |
| 384 | 128 | 85.42 | 85.45 | 84.89 |


#### Inference performance: NVIDIA A30

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA A30.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.93 | 0.93 | 0.63 | 0.88 | 1.28 | 0.89 |
| 128 | 2 | 0.76 | 0.76 | 0.76 | 1.13 | 1.13 | 1.13 |
| 128 | 4 | 1.05 | 1.59 | 1.06 | 1.55 | 1.71 | 1.54 |
| 128 | 8 | 1.48 | 1.51 | 1.45 | 2.53 | 2.55 | 2.50 |
| 128 | 12 | 2.02 | 2.05 | 1.98 | 3.58 | 3.64 | 3.55 |
| 128 | 16 | 2.39 | 2.41 | 2.37 | 4.45 | 4.49 | 4.44 |
| 128 | 24 | 3.49 | 3.49 | 3.47 | 6.90 | 6.94 | 6.86 |
| 128 | 32 | 4.46 | 4.46 | 4.41 | 8.76 | 8.77 | 8.63 |
| 128 | 64 | 8.64 | 8.65 | 8.52 | 17.31 | 17.32 | 17.17 |
| 128 | 128 | 16.68 | 16.80 | 16.50 | 33.91 | 34.07 | 33.58 |
| 384 | 1 | 1.33 | 2.04 | 1.33 | 1.70 | 1.70 | 1.70 |
| 384 | 2 | 1.70 | 1.70 | 1.70 | 2.38 | 2.39 | 2.36 |
| 384 | 4 | 2.29 | 2.30 | 2.28 | 3.91 | 3.91 | 3.87 |
| 384 | 8 | 4.26 | 4.39 | 4.23 | 7.55 | 7.62 | 7.53 |
| 384 | 12 | 6.08 | 6.10 | 6.02 | 10.67 | 10.73 | 10.58 |
| 384 | 16 | 8.12 | 8.12 | 8.06 | 14.65 | 14.66 | 14.48 |
| 384 | 24 | 11.86 | 11.87 | 11.77 | 21.40 | 21.43 | 21.17 |
| 384 | 32 | 15.52 | 15.53 | 15.40 | 28.21 | 28.23 | 28.00 |
| 384 | 64 | 30.76 | 30.82 | 30.44 | 54.72 | 54.88 | 54.32 |
| 384 | 128 | 60.99 | 61.15 | 60.20 | 107.94 | 108.19 | 107.15 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.76 | 1.76 | 1.75 | 2.14 | 2.14 | 2.14 |
| 128 | 2 | 1.86 | 1.91 | 1.86 | 2.97 | 2.97 | 2.94 |
| 128 | 4 | 2.72 | 2.75 | 2.70 | 4.85 | 4.89 | 4.81 |
| 128 | 8 | 4.34 | 4.41 | 4.26 | 8.44 | 8.46 | 8.35 |
| 128 | 12 | 5.65 | 5.73 | 5.60 | 11.02 | 11.02 | 10.91 |
| 128 | 16 | 7.67 | 7.68 | 7.64 | 15.17 | 15.18 | 15.08 |
| 128 | 24 | 10.51 | 10.53 | 10.40 | 21.23 | 21.34 | 21.02 |
| 128 | 32 | 14.11 | 14.15 | 13.97 | 29.38 | 29.40 | 29.14 |
| 128 | 64 | 26.49 | 26.77 | 26.48 | 55.79 | 55.82 | 55.38 |
| 128 | 128 | 52.29 | 52.38 | 51.73 | 109.43 | 109.92 | 108.52 |
| 384 | 1 | 3.47 | 3.48 | 3.47 | 4.61 | 4.62 | 4.57 |
| 384 | 2 | 4.35 | 4.39 | 4.31 | 7.04 | 7.08 | 6.98 |
| 384 | 4 | 7.23 | 7.37 | 7.22 | 12.47 | 12.51 | 12.32 |
| 384 | 8 | 12.92 | 12.93 | 12.76 | 23.45 | 23.48 | 23.21 |
| 384 | 12 | 18.60 | 18.69 | 18.43 | 34.94 | 35.14 | 34.58 |
| 384 | 16 | 24.35 | 24.37 | 24.01 | 45.35 | 45.62 | 45.29 |
| 384 | 24 | 35.60 | 35.97 | 35.30 | 67.29 | 67.42 | 66.62 |
| 384 | 32 | 47.03 | 47.20 | 46.56 | 88.35 | 88.71 | 87.57 |
| 384 | 64 | 92.04 | 92.37 | 91.21 | 174.21 | 174.91 | 173.29 |
| 384 | 128 | 180.77 | 181.11 | 179.78 | 343.25 | 343.80 | 342.30 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.43 | 1.43 | 1.43 |
| 128 | 2 | 1.90 | 1.90 | 1.90 |
| 128 | 4 | 3.12 | 3.13 | 3.09 |
| 128 | 8 | 4.79 | 4.79 | 4.78 |
| 128 | 12 | 6.38 | 6.39 | 6.35 |
| 128 | 16 | 8.63 | 8.67 | 8.55 |
| 128 | 24 | 11.99 | 12.00 | 11.92 |
| 128 | 32 | 16.42 | 16.43 | 16.37 |
| 128 | 64 | 30.11 | 30.12 | 29.91 |
| 128 | 128 | 58.93 | 59.03 | 58.39 |
| 384 | 1 | 2.70 | 2.70 | 2.70 |
| 384 | 2 | 4.18 | 4.18 | 4.17 |
| 384 | 4 | 7.33 | 7.35 | 7.26 |
| 384 | 8 | 13.78 | 13.79 | 13.63 |
| 384 | 12 | 19.47 | 19.48 | 19.30 |
| 384 | 16 | 25.55 | 25.56 | 25.34 |
| 384 | 24 | 37.13 | 37.15 | 36.55 |
| 384 | 32 | 48.76 | 48.78 | 48.20 |
| 384 | 64 | 95.57 | 95.85 | 94.96 |
| 384 | 128 | 186.36 | 186.83 | 185.37 |


#### Inference performance: NVIDIA T4 (16GB)

Results were obtained by running `scripts/inference_benchmark.sh --gpu Turing` on NVIDIA T4 (16G).

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.74 | 1.87 | 1.69 | 1.97 | 2.00 | 1.95 |
| 128 | 2 | 1.88 | 2.04 | 1.82 | 2.75 | 2.80 | 2.71 |
| 128 | 4 | 2.71 | 2.71 | 2.63 | 4.59 | 4.93 | 4.53 |
| 128 | 8 | 4.78 | 4.81 | 4.66 | 9.77 | 10.29 | 9.09 |
| 128 | 12 | 6.72 | 6.85 | 6.55 | 12.95 | 13.55 | 12.53 |
| 128 | 16 | 9.54 | 9.67 | 9.37 | 18.79 | 19.47 | 17.90 |
| 128 | 24 | 13.97 | 14.46 | 13.62 | 27.56 | 28.05 | 26.86 |
| 128 | 32 | 19.49 | 20.32 | 18.93 | 37.18 | 37.92 | 36.63 |
| 128 | 64 | 37.24 | 37.61 | 36.54 | 72.91 | 73.46 | 72.04 |
| 128 | 128 | 74.01 | 74.54 | 73.15 | 145.94 | 146.59 | 144.20 |
| 384 | 1 | 2.57 | 2.63 | 2.52 | 4.12 | 4.20 | 4.04 |
| 384 | 2 | 4.11 | 4.12 | 3.98 | 6.86 | 7.27 | 6.61 |
| 384 | 4 | 7.60 | 7.64 | 7.43 | 13.49 | 13.88 | 13.02 |
| 384 | 8 | 14.80 | 14.83 | 14.56 | 28.61 | 29.99 | 27.81 |
| 384 | 12 | 22.25 | 22.36 | 21.81 | 42.66 | 42.99 | 41.82 |
| 384 | 16 | 29.80 | 30.42 | 28.79 | 56.11 | 56.71 | 55.31 |
| 384 | 24 | 45.15 | 45.50 | 44.40 | 85.48 | 86.04 | 84.11 |
| 384 | 32 | 59.97 | 60.45 | 58.41 | 117.31 | 118.06 | 114.90 |
| 384 | 64 | 122.02 | 122.61 | 120.95 | 233.78 | 234.90 | 227.73 |
| 384 | 128 | 243.04 | 243.48 | 242.14 | 461.06 | 462.01 | 458.76 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 3.71 | 3.77 | 3.65 | 5.61 | 5.74 | 5.53 |
| 128 | 2 | 5.08 | 5.13 | 4.98 | 8.40 | 8.43 | 8.22 |
| 128 | 4 | 8.70 | 8.87 | 8.57 | 16.08 | 16.60 | 15.47 |
| 128 | 8 | 14.74 | 14.83 | 14.56 | 28.75 | 29.60 | 28.14 |
| 128 | 12 | 23.00 | 23.09 | 22.64 | 45.83 | 46.10 | 44.94 |
| 128 | 16 | 30.61 | 30.85 | 30.11 | 59.91 | 60.36 | 57.73 |
| 128 | 24 | 47.91 | 48.37 | 47.33 | 91.87 | 92.11 | 90.51 |
| 128 | 32 | 61.75 | 62.47 | 60.01 | 122.96 | 123.40 | 121.09 |
| 128 | 64 | 120.95 | 121.41 | 119.67 | 242.82 | 243.78 | 241.35 |
| 128 | 128 | 242.70 | 243.27 | 241.11 | 484.03 | 484.54 | 482.84 |
| 384 | 1 | 7.41 | 7.50 | 7.29 | 12.66 | 12.72 | 12.45 |
| 384 | 2 | 12.35 | 12.55 | 12.17 | 23.28 | 23.59 | 22.80 |
| 384 | 4 | 24.80 | 24.84 | 24.23 | 46.77 | 47.35 | 46.21 |
| 384 | 8 | 50.09 | 50.45 | 49.24 | 94.18 | 94.69 | 92.92 |
| 384 | 12 | 70.45 | 70.71 | 69.52 | 141.42 | 141.94 | 139.28 |
| 384 | 16 | 93.90 | 94.30 | 92.83 | 187.71 | 188.78 | 185.85 |
| 384 | 24 | 142.22 | 142.79 | 140.97 | 282.94 | 284.07 | 280.74 |
| 384 | 32 | 188.36 | 188.81 | 186.60 | 374.13 | 376.03 | 370.87 |
| 384 | 64 | 379.40 | 380.04 | 376.93 | 745.96 | 746.77 | 741.74 |
| 384 | 128 | 758.84 | 759.99 | 755.50 | 1523.14 | 1525.3 | 1518.35 |
