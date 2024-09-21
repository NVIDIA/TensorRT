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
    * [Inference performance: NVIDIA L4](#inference-performance-nvidia-l4)
    * [Inference performance: NVIDIA L40S](#inference-performance-nvidia-l40s)


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

To run the BERT model in TensorRT, we construct the model using TensorRT APIs and import the weights from a pre-trained TensorFlow checkpoint from [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_tf_ckpt_large_qa_squad2_amp_128). Finally, a TensorRT engine is generated and serialized to the disk. The various inference scripts then load this engine for inference.

Lastly, the tokens predicted by the model are projected back to the original text to get a final result.

### Version Info

The following software version configuration has been tested:

|Software|Version|
|--------|-------|
|Python|>=3.8|
|TensorRT|10.4.0.26|
|CUDA|12.6|

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

**Warning:** In the event of encountering an error message stating, "Missing API key and missing Email Authentication. This command requires an API key or authentication via browser login", the recommended steps for resolution are as follows:
* Generate an API key by logging in https://ngc.nvidia.com/setup/api-key and copy the generated API key.
* Execute the command `ngc config set` in the docker and paste the copied API key into the prompt as directed.

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

##### BERT base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.69 | 0.69 | 0.55 | 0.79 | 0.79 | 0.63 |
| 128 | 2 | 0.60 | 0.76 | 0.60 | 0.72 | 0.91 | 0.72 |
| 128 | 4 | 0.73 | 0.93 | 0.73 | 1.09 | 1.09 | 0.94 |
| 128 | 8 | 1.21 | 1.21 | 0.95 | 1.31 | 1.31 | 1.30 |
| 128 | 12 | 1.40 | 1.40 | 1.21 | 1.72 | 1.72 | 1.72 |
| 128 | 16 | 1.34 | 1.71 | 1.34 | 2.08 | 2.08 | 2.06 |
| 128 | 24 | 1.82 | 1.83 | 1.82 | 3.05 | 3.06 | 3.03 |
| 128 | 32 | 2.23 | 2.24 | 2.23 | 3.95 | 3.99 | 3.91 |
| 128 | 64 | 4.19 | 4.20 | 4.14 | 7.82 | 7.83 | 7.69 |
| 128 | 128 | 8.14 | 8.19 | 8.09 | 15.37 | 15.42 | 15.32 |
| 384 | 1 | 1.13 | 1.45 | 1.14 | 1.25 | 1.60 | 1.26 |
| 384 | 2 | 1.32 | 1.69 | 1.32 | 1.55 | 1.98 | 1.55 |
| 384 | 4 | 1.66 | 2.12 | 1.66 | 2.12 | 2.13 | 2.12 |
| 384 | 8 | 2.21 | 2.21 | 2.20 | 3.37 | 3.40 | 3.33 |
| 384 | 12 | 3.31 | 3.31 | 3.31 | 4.82 | 4.83 | 4.78 |
| 384 | 16 | 4.00 | 4.00 | 4.00 | 6.38 | 6.43 | 6.37 |
| 384 | 24 | 5.70 | 5.75 | 5.70 | 9.44 | 9.49 | 9.35 |
| 384 | 32 | 7.72 | 7.74 | 7.66 | 13.02 | 13.02 | 12.91 |
| 384 | 64 | 14.88 | 14.90 | 14.84 | 25.17 | 25.25 | 24.88 |
| 384 | 128 | 29.00 | 29.01 | 28.83 | 49.03 | 49.22 | 48.77 |

##### BERT large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.24 | 1.24 | 1.24 | 1.54 | 1.55 | 1.54 |
| 128 | 2 | 1.42 | 1.79 | 1.42 | 1.82 | 1.82 | 1.82 |
| 128 | 4 | 1.78 | 1.79 | 1.78 | 2.53 | 2.53 | 2.52 |
| 128 | 8 | 2.64 | 2.64 | 2.64 | 4.07 | 4.10 | 4.06 |
| 128 | 12 | 3.11 | 3.12 | 3.11 | 5.08 | 5.10 | 5.03 |
| 128 | 16 | 4.03 | 4.03 | 4.03 | 6.95 | 6.95 | 6.90 |
| 128 | 24 | 5.32 | 5.34 | 5.30 | 9.80 | 9.90 | 9.72 |
| 128 | 32 | 7.07 | 7.07 | 7.00 | 13.08 | 13.08 | 12.93 |
| 128 | 64 | 12.94 | 13.01 | 12.82 | 24.83 | 24.99 | 24.69 |
| 128 | 128 | 25.29 | 25.29 | 25.09 | 49.70 | 49.72 | 49.06 |
| 384 | 1 | 2.55 | 2.56 | 2.55 | 2.96 | 2.96 | 2.96 |
| 384 | 2 | 3.04 | 3.04 | 3.03 | 3.90 | 3.90 | 3.90 |
| 384 | 4 | 4.01 | 4.01 | 4.01 | 5.74 | 5.79 | 5.71 |
| 384 | 8 | 7.16 | 7.16 | 7.15 | 11.15 | 11.24 | 11.09 |
| 384 | 12 | 9.15 | 9.23 | 9.14 | 15.46 | 15.47 | 15.40 |
| 384 | 16 | 12.40 | 12.40 | 12.29 | 21.17 | 21.18 | 21.05 |
| 384 | 24 | 17.72 | 17.85 | 17.64 | 31.09 | 31.36 | 30.81 |
| 384 | 32 | 23.29 | 23.31 | 23.15 | 41.32 | 41.34 | 40.86 |
| 384 | 64 | 45.38 | 45.40 | 45.02 | 79.95 | 80.27 | 79.31 |
| 384 | 128 | 87.97 | 87.99 | 87.89 | 156.97 | 157.56 | 155.84 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.24 | 1.56 | 1.24 |
| 128 | 2 | 1.42 | 1.42 | 1.42 |
| 128 | 4 | 1.78 | 1.79 | 1.78 |
| 128 | 8 | 2.64 | 2.65 | 2.64 |
| 128 | 12 | 3.11 | 3.12 | 3.11 |
| 128 | 16 | 4.03 | 4.03 | 4.02 |
| 128 | 24 | 5.32 | 5.34 | 5.31 |
| 128 | 32 | 7.07 | 7.09 | 7.02 |
| 128 | 64 | 12.98 | 13.01 | 12.86 |
| 128 | 128 | 25.40 | 25.55 | 25.17 |
| 384 | 1 | 2.55 | 2.55 | 2.55 |
| 384 | 2 | 3.03 | 3.04 | 3.03 |
| 384 | 4 | 4.01 | 4.01 | 4.01 |
| 384 | 8 | 7.16 | 7.16 | 7.16 |
| 384 | 12 | 9.14 | 9.23 | 9.14 |
| 384 | 16 | 12.31 | 12.41 | 12.29 |
| 384 | 24 | 17.85 | 17.90 | 17.68 |
| 384 | 32 | 23.41 | 23.51 | 23.23 |
| 384 | 64 | 45.39 | 45.40 | 45.09 |
| 384 | 128 | 88.73 | 88.79 | 88.11 |

### Inference Performance NVIDIA L4

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA L4.

##### BERT base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.62 | 0.62 | 0.60 | 1.03 | 1.03 | 1.01 |
| 128 | 2 | 0.79 | 0.80 | 0.77 | 1.31 | 1.35 | 1.30 |
| 128 | 4 | 1.14 | 1.15 | 1.12 | 2.23 | 2.23 | 2.15 |
| 128 | 8 | 1.97 | 1.97 | 1.92 | 3.68 | 3.69 | 3.63 |
| 128 | 12 | 2.66 | 2.67 | 2.61 | 5.34 | 5.35 | 5.27 |
| 128 | 16 | 3.39 | 3.39 | 3.34 | 6.62 | 6.69 | 6.58 |
| 128 | 24 | 4.84 | 4.85 | 4.76 | 10.49 | 10.55 | 10.32 |
| 128 | 32 | 6.20 | 6.29 | 6.14 | 13.92 | 13.92 | 13.75 |
| 128 | 64 | 13.42 | 13.42 | 13.26 | 31.28 | 31.48 | 31.07 |
| 128 | 128 | 28.48 | 28.64 | 28.19 | 66.10 | 66.23 | 65.36 |
| 384 | 1 | 1.29 | 1.30 | 1.29 | 2.08 | 2.09 | 2.08 |
| 384 | 2 | 1.83 | 1.84 | 1.82 | 3.15 | 3.19 | 3.11 |
| 384 | 4 | 2.99 | 2.99 | 2.92 | 5.75 | 5.81 | 5.68 |
| 384 | 8 | 5.53 | 5.54 | 5.42 | 11.28 | 11.33 | 11.08 |
| 384 | 12 | 8.26 | 8.29 | 8.09 | 17.19 | 17.22 | 16.99 |
| 384 | 16 | 11.00 | 11.08 | 10.85 | 23.38 | 23.38 | 22.90 |
| 384 | 24 | 16.79 | 16.89 | 16.60 | 37.90 | 38.29 | 37.18 |
| 384 | 32 | 23.08 | 23.31 | 22.74 | 50.70 | 50.94 | 50.27 |
| 384 | 64 | 49.43 | 49.86 | 48.56 | 103.88 | 104.19 | 102.89 |
| 384 | 128 | 104.55 | 104.97 | 103.74 | 211.09 | 211.67 | 209.85 |

##### BERT large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.78 | 1.79 | 1.76 | 3.11 | 3.11 | 3.10 |
| 128 | 2 | 2.50 | 2.51 | 2.44 | 4.35 | 4.45 | 4.31 |
| 128 | 4 | 3.60 | 3.63 | 3.54 | 6.83 | 6.86 | 6.77 |
| 128 | 8 | 6.27 | 6.31 | 6.25 | 12.98 | 13.01 | 12.80 |
| 128 | 12 | 8.40 | 8.41 | 8.27 | 18.45 | 18.66 | 18.22 |
| 128 | 16 | 11.22 | 11.23 | 11.12 | 25.18 | 25.19 | 25.14 |
| 128 | 24 | 15.95 | 16.10 | 15.82 | 35.67 | 35.68 | 35.59 |
| 128 | 32 | 21.30 | 21.35 | 20.90 | 49.02 | 49.26 | 48.33 |
| 128 | 64 | 44.08 | 44.32 | 43.93 | 107.89 | 108.30 | 107.11 |
| 128 | 128 | 93.69 | 94.36 | 92.69 | 215.00 | 215.46 | 213.84 |
| 384 | 1 | 3.43 | 3.44 | 3.41 | 6.58 | 6.66 | 6.40 |
| 384 | 2 | 5.55 | 5.55 | 5.49 | 10.56 | 10.59 | 10.44 |
| 384 | 4 | 9.80 | 9.88 | 9.58 | 20.55 | 20.94 | 19.93 |
| 384 | 8 | 18.04 | 18.11 | 17.86 | 38.87 | 39.47 | 37.69 |
| 384 | 12 | 26.44 | 26.61 | 26.14 | 59.28 | 59.85 | 56.90 |
| 384 | 16 | 36.37 | 36.48 | 36.04 | 82.93 | 83.33 | 81.95 |
| 384 | 24 | 53.60 | 53.73 | 53.15 | 122.78 | 123.06 | 122.05 |
| 384 | 32 | 75.52 | 75.84 | 74.45 | 164.55 | 164.98 | 163.68 |
| 384 | 64 | 157.71 | 158.27 | 155.68 | 345.90 | 346.53 | 344.57 |
| 384 | 128 | 331.37 | 332.44 | 329.06 | 663.75 | 664.69 | 661.89 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.78 | 1.79 | 1.76 |
| 128 | 2 | 2.50 | 2.51 | 2.44 |
| 128 | 4 | 3.56 | 3.57 | 3.54 |
| 128 | 8 | 6.27 | 6.31 | 6.26 |
| 128 | 12 | 8.40 | 8.41 | 8.29 |
| 128 | 16 | 11.23 | 11.23 | 11.16 |
| 128 | 24 | 16.06 | 16.12 | 15.90 |
| 128 | 32 | 21.31 | 21.34 | 20.98 |
| 128 | 64 | 44.15 | 44.66 | 43.88 |
| 128 | 128 | 94.19 | 94.93 | 92.81 |
| 384 | 1 | 3.39 | 3.43 | 3.37 |
| 384 | 2 | 5.56 | 5.56 | 5.48 |
| 384 | 4 | 9.81 | 9.90 | 9.61 |
| 384 | 8 | 18.07 | 18.25 | 17.94 |
| 384 | 12 | 26.47 | 26.57 | 26.27 |
| 384 | 16 | 36.78 | 37.14 | 36.37 |
| 384 | 24 | 54.16 | 54.53 | 53.65 |
| 384 | 32 | 75.33 | 75.62 | 74.69 |
| 384 | 64 | 158.72 | 159.55 | 156.72 |
| 384 | 128 | 333.24 | 334.26 | 330.67 |

### Inference Performance NVIDIA L40S

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA L40S.

##### BERT base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.34 | 0.34 | 0.34 | 0.48 | 0.48 | 0.48 |
| 128 | 2 | 0.41 | 0.41 | 0.41 | 0.56 | 0.56 | 0.55 |
| 128 | 4 | 0.50 | 0.50 | 0.50 | 0.77 | 0.77 | 0.77 |
| 128 | 8 | 0.67 | 0.67 | 0.67 | 1.30 | 1.30 | 1.29 |
| 128 | 12 | 0.91 | 0.91 | 0.91 | 1.68 | 1.68 | 1.67 |
| 128 | 16 | 1.09 | 1.10 | 1.09 | 2.22 | 2.23 | 2.22 |
| 128 | 24 | 1.50 | 1.50 | 1.48 | 3.23 | 3.24 | 3.20 |
| 128 | 32 | 1.82 | 1.83 | 1.82 | 3.94 | 3.94 | 3.93 |
| 128 | 64 | 3.47 | 3.47 | 3.45 | 8.24 | 8.26 | 8.14 |
| 128 | 128 | 7.74 | 7.91 | 7.66 | 17.73 | 17.86 | 17.56 |
| 384 | 1 | 0.73 | 0.73 | 0.73 | 1.02 | 1.02 | 1.02 |
| 384 | 2 | 0.88 | 0.89 | 0.88 | 1.38 | 1.38 | 1.37 |
| 384 | 4 | 1.17 | 1.17 | 1.16 | 2.16 | 2.17 | 2.15 |
| 384 | 8 | 1.72 | 1.73 | 1.72 | 3.45 | 3.46 | 3.45 |
| 384 | 12 | 2.73 | 2.73 | 2.72 | 5.07 | 5.07 | 5.05 |
| 384 | 16 | 3.28 | 3.28 | 3.27 | 7.41 | 7.44 | 7.37 |
| 384 | 24 | 4.93 | 4.94 | 4.90 | 10.16 | 10.19 | 10.09 |
| 384 | 32 | 6.33 | 6.34 | 6.29 | 14.07 | 14.11 | 13.96 |
| 384 | 64 | 13.74 | 13.76 | 13.57 | 30.65 | 30.82 | 30.14 |
| 384 | 128 | 28.25 | 28.41 | 27.87 | 62.48 | 62.67 | 61.70 |

##### BERT large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.89 | 0.89 | 0.88 | 1.30 | 1.30 | 1.30 |
| 128 | 2 | 0.98 | 0.98 | 0.98 | 1.47 | 1.47 | 1.47 |
| 128 | 4 | 1.34 | 1.35 | 1.33 | 2.29 | 2.29 | 2.28 |
| 128 | 8 | 1.92 | 1.92 | 1.91 | 3.82 | 3.83 | 3.79 |
| 128 | 12 | 2.77 | 2.78 | 2.75 | 5.73 | 5.73 | 5.71 |
| 128 | 16 | 3.22 | 3.22 | 3.19 | 6.72 | 6.73 | 6.67 |
| 128 | 24 | 4.54 | 4.55 | 4.52 | 10.38 | 10.39 | 10.31 |
| 128 | 32 | 5.67 | 5.68 | 5.63 | 12.87 | 12.90 | 12.74 |
| 128 | 64 | 11.92 | 11.96 | 11.77 | 28.21 | 28.40 | 27.89 |
| 128 | 128 | 25.44 | 25.49 | 25.12 | 61.85 | 62.01 | 61.20 |
| 384 | 1 | 1.68 | 1.68 | 1.68 | 2.74 | 2.75 | 2.74 |
| 384 | 2 | 2.32 | 2.32 | 2.30 | 3.87 | 3.87 | 3.85 |
| 384 | 4 | 3.27 | 3.28 | 3.27 | 6.32 | 6.35 | 6.29 |
| 384 | 8 | 5.09 | 5.09 | 5.06 | 10.76 | 10.77 | 10.60 |
| 384 | 12 | 8.06 | 8.07 | 8.02 | 18.73 | 18.77 | 18.64 |
| 384 | 16 | 9.70 | 9.75 | 9.61 | 22.15 | 22.26 | 21.95 |
| 384 | 24 | 15.02 | 15.04 | 14.88 | 35.43 | 35.48 | 35.15 |
| 384 | 32 | 20.37 | 20.49 | 20.00 | 46.36 | 46.37 | 45.86 |
| 384 | 64 | 43.50 | 43.65 | 43.02 | 105.84 | 106.08 | 104.92 |
| 384 | 128 | 86.30 | 86.48 | 85.58 | 195.18 | 195.98 | 192.90 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.89 | 0.89 | 0.88 |
| 128 | 2 | 0.98 | 0.98 | 0.98 |
| 128 | 4 | 1.34 | 1.36 | 1.33 |
| 128 | 8 | 1.93 | 1.95 | 1.91 |
| 128 | 12 | 2.79 | 2.82 | 2.77 |
| 128 | 16 | 3.24 | 3.24 | 3.22 |
| 128 | 24 | 4.59 | 4.59 | 4.57 |
| 128 | 32 | 5.68 | 5.68 | 5.65 |
| 128 | 64 | 11.81 | 11.87 | 11.71 |
| 128 | 128 | 26.21 | 26.24 | 25.86 |
| 384 | 1 | 1.68 | 1.68 | 1.68 |
| 384 | 2 | 2.31 | 2.32 | 2.31 |
| 384 | 4 | 3.29 | 3.29 | 3.28 |
| 384 | 8 | 5.14 | 5.15 | 5.10 |
| 384 | 12 | 8.05 | 8.06 | 8.01 |
| 384 | 16 | 9.78 | 9.80 | 9.66 |
| 384 | 24 | 15.14 | 15.15 | 15.01 |
| 384 | 32 | 20.34 | 20.42 | 19.99 |
| 384 | 64 | 43.81 | 43.97 | 43.39 |
| 384 | 128 | 88.37 | 88.64 | 87.38 |