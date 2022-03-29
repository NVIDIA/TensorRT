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
      * [BERT Base](#bert-base)
      * [BERT Large](#bert-large)
      * [Megatron Large Sparse](#megatron-large-with-sparsity)
    * [Inference performance: NVIDIA T4](#inference-performance-nvidia-t4-16gb)
      * [BERT Base](#bert-base-1)
      * [BERT Large](#bert-large-1)


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
|Python|>=3.6.x|
|TensorRT|8.0.1.6|
|CUDA|11.3.1|


## Setup

The following section lists the requirements that you need to meet in order to run the BERT model.

### Requirements

This demo BERT application can be run within the TensorRT OSS build container. If running in a different environment, following packages are required.

* [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) - for downloading BERT checkpoints from NGC.
* PyPI Packages:
  * [pycuda](https://pypi.org/project/pycuda/) (tested v2019.1.2)
  * [onnx](https://pypi.org/project/onnx) (tested v1.8.1)
  * [tensorflow](https://pypi.org/project/tensorflow/) (tested v2.4.1)
  * [torch](https://pypi.org/project/torch/1.6.0/) (tested v1.8.1)
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

Our results for BERT were obtained by running the `scripts/inference_benchmark.sh --gpu Ampere` script in the container generated by the TensorRT OSS Dockerfile on NVIDIA A100 with (1x A100 40G) GPUs.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.72 | 0.72 | 0.59 | 0.66 | 0.81 | 0.65 |
| 128 | 2 | 0.68 | 0.68 | 0.64 | 0.97 | 0.97 | 0.79 |
| 128 | 4 | 0.99 | 0.99 | 0.79 | 1.02 | 1.29 | 1.02 |
| 128 | 8 | 0.94 | 1.21 | 0.94 | 1.38 | 1.39 | 1.38 |
| 128 | 12 | 1.22 | 1.23 | 1.22 | 1.91 | 1.92 | 1.91 |
| 128 | 16 | 1.40 | 1.40 | 1.40 | 2.19 | 2.20 | 2.19 |
| 128 | 24 | 1.93 | 1.94 | 1.92 | 3.37 | 3.38 | 3.34 |
| 128 | 32 | 2.48 | 2.48 | 2.47 | 4.08 | 4.14 | 4.07 |
| 128 | 64 | 4.31 | 4.31 | 4.27 | 8.08 | 8.09 | 8.00 |
| 128 | 128 | 8.37 | 8.38 | 8.31 | 16.14 | 16.21 | 16.02 |
| 384 | 1 | 1.15 | 1.47 | 1.15 | 1.30 | 1.65 | 1.31 |
| 384 | 2 | 1.34 | 1.72 | 1.35 | 1.66 | 1.67 | 1.66 |
| 384 | 4 | 1.69 | 1.70 | 1.69 | 2.27 | 2.28 | 2.27 |
| 384 | 8 | 2.29 | 2.30 | 2.28 | 3.67 | 3.70 | 3.66 |
| 384 | 12 | 3.46 | 3.46 | 3.45 | 5.06 | 5.08 | 5.01 |
| 384 | 16 | 4.20 | 4.20 | 4.19 | 6.73 | 6.75 | 6.67 |
| 384 | 24 | 5.94 | 5.95 | 5.94 | 9.86 | 9.87 | 9.75 |
| 384 | 32 | 7.93 | 7.94 | 7.92 | 13.56 | 13.61 | 13.44 |
| 384 | 64 | 15.48 | 15.49 | 15.39 | 26.09 | 26.26 | 25.94 |
| 384 | 128 | 29.92 | 29.95 | 29.68 | 51.65 | 51.71 | 51.02 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.24 | 1.25 | 1.24 | 1.58 | 1.60 | 1.58 |
| 128 | 2 | 1.51 | 1.52 | 1.51 | 2.00 | 2.02 | 2.00 |
| 128 | 4 | 1.83 | 1.84 | 1.82 | 2.95 | 2.96 | 2.95 |
| 128 | 8 | 2.69 | 2.70 | 2.68 | 4.44 | 4.45 | 4.43 |
| 128 | 12 | 3.11 | 3.12 | 3.11 | 5.25 | 5.30 | 5.23 |
| 128 | 16 | 4.05 | 4.06 | 4.05 | 7.65 | 7.72 | 7.63 |
| 128 | 24 | 5.24 | 5.25 | 5.23 | 10.14 | 10.16 | 10.09 |
| 128 | 32 | 7.01 | 7.07 | 7.01 | 13.89 | 13.89 | 13.77 |
| 128 | 64 | 13.15 | 13.18 | 13.05 | 26.10 | 26.13 | 26.00 |
| 128 | 128 | 25.29 | 25.32 | 25.21 | 51.69 | 51.77 | 51.38 |
| 384 | 1 | 2.66 | 2.66 | 2.66 | 3.09 | 3.10 | 3.09 |
| 384 | 2 | 3.03 | 3.05 | 3.03 | 4.14 | 4.15 | 4.14 |
| 384 | 4 | 4.04 | 4.05 | 4.04 | 5.99 | 5.99 | 5.93 |
| 384 | 8 | 7.13 | 7.14 | 7.13 | 11.60 | 11.62 | 11.47 |
| 384 | 12 | 9.21 | 9.22 | 9.20 | 16.33 | 16.34 | 16.09 |
| 384 | 16 | 12.37 | 12.39 | 12.36 | 22.14 | 22.22 | 21.98 |
| 384 | 24 | 17.51 | 17.52 | 17.49 | 32.44 | 32.56 | 32.29 |
| 384 | 32 | 23.38 | 23.40 | 23.14 | 43.12 | 43.23 | 42.73 |
| 384 | 64 | 45.20 | 45.25 | 45.07 | 83.75 | 83.92 | 83.15 |
| 384 | 128 | 88.18 | 88.26 | 88.01 | 163.61 | 164.08 | 162.62 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.16 | 1.48 | 1.17 |
| 128 | 2 | 1.41 | 1.42 | 1.41 |
| 128 | 4 | 1.87 | 1.88 | 1.87 |
| 128 | 8 | 2.84 | 2.84 | 2.83 |
| 128 | 12 | 3.30 | 3.31 | 3.30 |
| 128 | 16 | 4.40 | 4.42 | 4.39 |
| 128 | 24 | 5.86 | 5.87 | 5.85 |
| 128 | 32 | 7.67 | 7.68 | 7.67 |
| 128 | 64 | 13.81 | 13.82 | 13.79 |
| 128 | 128 | 27.00 | 27.02 | 26.80 |
| 384 | 1 | 1.72 | 1.78 | 1.72 |
| 384 | 2 | 2.35 | 2.36 | 2.35 |
| 384 | 4 | 3.80 | 3.81 | 3.80 |
| 384 | 8 | 6.70 | 6.71 | 6.70 |
| 384 | 12 | 8.98 | 8.99 | 8.97 |
| 384 | 16 | 12.38 | 12.39 | 12.37 |
| 384 | 24 | 17.52 | 17.54 | 17.51 |
| 384 | 32 | 22.82 | 22.89 | 22.64 |
| 384 | 64 | 43.78 | 43.90 | 43.59 |
| 384 | 128 | 85.23 | 85.25 | 84.61 |


#### Inference performance: NVIDIA T4 (16GB)

Our results were obtained by running the `scripts/inference_benchmark.sh --gpu Turing` script in the container generated by the TensorRT OSS Dockerfile on NVIDIA T4 with (1x T4 16G) GPUs.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.64 | 1.68 | 1.37 | 2.00 | 2.15 | 1.96 |
| 128 | 2 | 1.88 | 2.11 | 1.82 | 2.77 | 2.78 | 2.70 |
| 128 | 4 | 2.70 | 2.70 | 2.63 | 4.55 | 4.57 | 4.48 |
| 128 | 8 | 4.73 | 4.97 | 4.64 | 9.33 | 10.22 | 8.85 |
| 128 | 12 | 6.63 | 6.73 | 6.55 | 12.82 | 13.19 | 12.39 |
| 128 | 16 | 9.45 | 9.77 | 9.31 | 18.08 | 18.63 | 17.35 |
| 128 | 24 | 14.07 | 14.35 | 13.63 | 27.77 | 28.77 | 26.88 |
| 128 | 32 | 19.75 | 20.59 | 19.12 | 37.42 | 37.79 | 36.66 |
| 128 | 64 | 37.78 | 38.34 | 37.02 | 72.84 | 72.88 | 71.84 |
| 128 | 128 | 74.62 | 75.10 | 73.61 | 147.01 | 147.83 | 145.46 |
| 384 | 1 | 2.59 | 2.63 | 2.51 | 4.12 | 4.16 | 4.03 |
| 384 | 2 | 4.11 | 4.13 | 3.98 | 6.85 | 7.38 | 6.62 |
| 384 | 4 | 7.43 | 7.48 | 7.32 | 13.43 | 13.80 | 12.93 |
| 384 | 8 | 14.94 | 15.08 | 14.73 | 28.62 | 29.65 | 27.84 |
| 384 | 12 | 22.51 | 22.86 | 22.05 | 42.67 | 43.18 | 41.88 |
| 384 | 16 | 30.07 | 30.77 | 29.10 | 57.08 | 57.56 | 56.18 |
| 384 | 24 | 45.34 | 45.90 | 44.62 | 87.62 | 88.20 | 85.69 |
| 384 | 32 | 60.10 | 60.50 | 58.77 | 118.02 | 118.76 | 115.03 |
| 384 | 64 | 121.20 | 121.69 | 118.76 | 235.94 | 237.30 | 230.79 |
| 384 | 128 | 243.66 | 244.15 | 242.68 | 447.69 | 448.97 | 445.64 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 3.72 | 3.81 | 3.67 | 5.69 | 5.76 | 5.57 |
| 128 | 2 | 5.08 | 5.27 | 5.00 | 8.52 | 8.64 | 8.40 |
| 128 | 4 | 8.51 | 8.54 | 8.32 | 14.93 | 15.23 | 14.53 |
| 128 | 8 | 14.77 | 14.89 | 14.58 | 28.77 | 29.22 | 28.25 |
| 128 | 12 | 23.07 | 23.22 | 22.74 | 46.08 | 46.10 | 45.28 |
| 128 | 16 | 30.29 | 30.56 | 29.57 | 60.23 | 60.93 | 58.35 |
| 128 | 24 | 48.58 | 48.70 | 47.77 | 90.67 | 91.77 | 89.92 |
| 128 | 32 | 64.13 | 64.80 | 63.15 | 117.89 | 118.47 | 116.12 |
| 128 | 64 | 127.74 | 128.46 | 125.80 | 243.10 | 243.52 | 241.59 |
| 128 | 128 | 242.26 | 242.86 | 240.10 | 465.64 | 466.77 | 463.31 |
| 384 | 1 | 7.50 | 7.54 | 7.31 | 12.56 | 12.67 | 12.37 |
| 384 | 2 | 12.46 | 12.58 | 12.23 | 23.09 | 23.11 | 22.57 |
| 384 | 4 | 24.93 | 25.10 | 24.58 | 47.41 | 47.43 | 46.68 |
| 384 | 8 | 50.73 | 50.95 | 49.83 | 93.40 | 94.03 | 92.25 |
| 384 | 12 | 72.95 | 73.36 | 71.97 | 140.44 | 141.25 | 138.23 |
| 384 | 16 | 95.85 | 96.26 | 94.21 | 186.44 | 187.16 | 184.91 |
| 384 | 24 | 145.08 | 145.57 | 143.04 | 281.55 | 282.20 | 279.78 |
| 384 | 32 | 188.62 | 189.24 | 187.12 | 375.30 | 375.91 | 372.80 |
| 384 | 64 | 376.59 | 377.52 | 374.39 | 760.16 | 760.96 | 757.81 |
| 384 | 128 | 758.68 | 759.85 | 754.89 | 1459.63 | 1460.42 | 1457.38 |
