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
|TensorRT|10.2.0.19|
|CUDA|12.5|

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

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.54 | 0.69 | 0.54 | 0.79 | 0.79 | 0.63 |
| 128 | 2 | 0.76 | 0.76 | 0.61 | 0.72 | 0.92 | 0.72 |
| 128 | 4 | 0.93 | 0.93 | 0.74 | 0.93 | 1.19 | 0.93 |
| 128 | 8 | 0.94 | 1.20 | 0.94 | 1.31 | 1.31 | 1.31 |
| 128 | 12 | 1.20 | 1.53 | 1.21 | 1.70 | 2.15 | 1.69 |
| 128 | 16 | 1.33 | 1.34 | 1.33 | 2.08 | 2.08 | 2.06 |
| 128 | 24 | 1.82 | 1.82 | 1.82 | 3.05 | 3.05 | 3.03 |
| 128 | 32 | 2.23 | 2.24 | 2.23 | 3.92 | 3.92 | 3.90 |
| 128 | 64 | 4.19 | 4.19 | 4.14 | 7.75 | 7.76 | 7.68 |
| 128 | 128 | 8.14 | 8.14 | 8.08 | 15.37 | 15.44 | 15.29 |
| 384 | 1 | 1.13 | 1.13 | 1.14 | 1.25 | 1.61 | 1.26 |
| 384 | 2 | 1.32 | 1.56 | 1.32 | 1.55 | 1.55 | 1.54 |
| 384 | 4 | 1.66 | 2.12 | 1.66 | 2.12 | 2.12 | 2.12 |
| 384 | 8 | 2.21 | 2.30 | 2.21 | 3.34 | 3.40 | 3.33 |
| 384 | 12 | 3.31 | 3.32 | 3.31 | 4.84 | 4.84 | 4.79 |
| 384 | 16 | 4.00 | 4.00 | 4.00 | 6.39 | 6.39 | 6.36 |
| 384 | 24 | 5.70 | 5.70 | 5.69 | 9.49 | 9.49 | 9.41 |
| 384 | 32 | 7.70 | 7.72 | 7.64 | 13.02 | 13.03 | 12.89 |
| 384 | 64 | 14.89 | 14.90 | 14.79 | 25.16 | 25.18 | 24.85 |
| 384 | 128 | 29.01 | 29.02 | 28.78 | 49.11 | 49.24 | 48.73 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.24 | 1.25 | 1.24 | 1.55 | 1.55 | 1.55 |
| 128 | 2 | 1.43 | 1.80 | 1.43 | 1.82 | 1.82 | 1.82 |
| 128 | 4 | 1.78 | 1.79 | 1.78 | 2.53 | 2.54 | 2.53 |
| 128 | 8 | 2.64 | 2.64 | 2.64 | 3.99 | 4.01 | 3.96 |
| 128 | 12 | 3.08 | 3.09 | 3.08 | 5.08 | 5.08 | 5.02 |
| 128 | 16 | 4.03 | 4.03 | 4.03 | 6.94 | 6.94 | 6.89 |
| 128 | 24 | 5.32 | 5.34 | 5.28 | 9.71 | 9.80 | 9.69 |
| 128 | 32 | 7.02 | 7.09 | 6.99 | 12.95 | 13.08 | 12.89 |
| 128 | 64 | 12.89 | 12.89 | 12.80 | 24.83 | 25.00 | 24.65 |
| 128 | 128 | 25.28 | 25.29 | 25.05 | 49.15 | 49.41 | 48.82 |
| 384 | 1 | 2.55 | 2.56 | 2.55 | 2.96 | 2.96 | 2.96 |
| 384 | 2 | 3.04 | 3.04 | 3.03 | 4.00 | 4.01 | 4.00 |
| 384 | 4 | 4.04 | 4.04 | 4.04 | 5.73 | 5.75 | 5.70 |
| 384 | 8 | 7.17 | 7.17 | 7.16 | 11.14 | 11.16 | 11.07 |
| 384 | 12 | 9.14 | 9.14 | 9.13 | 15.46 | 15.47 | 15.36 |
| 384 | 16 | 12.28 | 12.40 | 12.28 | 21.20 | 21.31 | 21.06 |
| 384 | 24 | 17.70 | 17.84 | 17.63 | 31.03 | 31.04 | 30.76 |
| 384 | 32 | 23.29 | 23.30 | 23.11 | 41.07 | 41.31 | 40.74 |
| 384 | 64 | 44.94 | 45.20 | 44.87 | 80.15 | 80.36 | 79.42 |
| 384 | 128 | 87.97 | 87.99 | 87.81 | 157.22 | 157.81 | 156.05 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.11 | 1.41 | 1.12 |
| 128 | 2 | 1.33 | 1.34 | 1.33 |
| 128 | 4 | 1.78 | 1.78 | 1.77 |
| 128 | 8 | 2.54 | 2.54 | 2.53 |
| 128 | 12 | 2.97 | 2.97 | 2.96 |
| 128 | 16 | 3.90 | 3.91 | 3.90 |
| 128 | 24 | 4.89 | 4.89 | 4.88 |
| 128 | 32 | 6.99 | 7.01 | 6.94 |
| 128 | 64 | 11.62 | 11.69 | 11.60 |
| 128 | 128 | 21.38 | 21.39 | 21.21 |
| 384 | 1 | 1.68 | 1.68 | 1.68 |
| 384 | 2 | 2.21 | 2.21 | 2.21 |
| 384 | 4 | 3.48 | 3.48 | 3.47 |
| 384 | 8 | 5.73 | 5.74 | 5.73 |
| 384 | 12 | 8.37 | 8.37 | 8.35 |
| 384 | 16 | 10.35 | 10.36 | 10.33 |
| 384 | 24 | 14.62 | 14.62 | 14.61 |
| 384 | 32 | 18.91 | 18.95 | 18.75 |
| 384 | 64 | 35.84 | 35.86 | 35.61 |
| 384 | 128 | 67.81 | 67.83 | 67.73 |

### Inference Performance NVIDIA L4

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA L4.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.62 | 0.62 | 0.61 | 1.03 | 1.03 | 1.01 |
| 128 | 2 | 0.81 | 0.81 | 0.78 | 1.35 | 1.37 | 1.33 |
| 128 | 4 | 1.16 | 1.16 | 1.14 | 2.17 | 2.18 | 2.14 |
| 128 | 8 | 1.95 | 2.00 | 1.92 | 3.68 | 3.68 | 3.60 |
| 128 | 12 | 2.70 | 2.71 | 2.64 | 5.24 | 5.26 | 5.14 |
| 128 | 16 | 3.44 | 3.44 | 3.34 | 6.77 | 6.77 | 6.64 |
| 128 | 24 | 4.91 | 4.94 | 4.80 | 10.19 | 10.42 | 10.15 |
| 128 | 32 | 6.31 | 6.40 | 6.23 | 13.57 | 13.72 | 13.41 |
| 128 | 64 | 13.69 | 13.85 | 13.46 | 30.35 | 30.72 | 29.58 |
| 128 | 128 | 28.90 | 29.15 | 28.61 | 66.75 | 67.06 | 66.09 |
| 384 | 1 | 1.30 | 1.30 | 1.30 | 2.10 | 2.10 | 2.09 |
| 384 | 2 | 1.85 | 1.86 | 1.84 | 3.18 | 3.20 | 3.17 |
| 384 | 4 | 3.02 | 3.02 | 2.96 | 5.49 | 5.53 | 5.48 |
| 384 | 8 | 5.60 | 5.64 | 5.50 | 11.10 | 11.11 | 10.90 |
| 384 | 12 | 8.37 | 8.39 | 8.20 | 16.61 | 16.76 | 16.51 |
| 384 | 16 | 11.18 | 11.19 | 11.02 | 23.24 | 23.56 | 23.16 |
| 384 | 24 | 17.09 | 17.29 | 16.96 | 35.94 | 35.95 | 35.39 |
| 384 | 32 | 23.38 | 23.57 | 23.17 | 50.65 | 50.92 | 50.51 |
| 384 | 64 | 49.52 | 49.54 | 49.01 | 104.52 | 104.94 | 103.73 |
| 384 | 128 | 104.93 | 105.33 | 103.94 | 197.12 | 197.56 | 196.03 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.81 | 1.82 | 1.79 | 3.15 | 3.16 | 3.12 |
| 128 | 2 | 2.50 | 2.55 | 2.47 | 4.49 | 4.58 | 4.44 |
| 128 | 4 | 3.60 | 3.62 | 3.59 | 6.94 | 6.95 | 6.90 |
| 128 | 8 | 6.44 | 6.50 | 6.34 | 12.93 | 12.99 | 12.79 |
| 128 | 12 | 8.53 | 8.53 | 8.35 | 18.26 | 18.27 | 18.08 |
| 128 | 16 | 11.37 | 11.37 | 11.23 | 25.17 | 25.40 | 25.04 |
| 128 | 24 | 16.13 | 16.14 | 16.09 | 35.45 | 35.45 | 35.26 |
| 128 | 32 | 21.66 | 21.66 | 21.56 | 47.66 | 47.66 | 47.63 |
| 128 | 64 | 47.07 | 47.08 | 46.65 | 102.00 | 102.24 | 101.29 |
| 128 | 128 | 91.60 | 92.23 | 91.19 | 219.24 | 219.55 | 218.06 |
| 384 | 1 | 3.47 | 3.48 | 3.47 | 6.53 | 6.63 | 6.36 |
| 384 | 2 | 5.58 | 5.58 | 5.53 | 10.51 | 10.62 | 10.44 |
| 384 | 4 | 9.91 | 10.01 | 9.73 | 20.58 | 20.80 | 20.10 |
| 384 | 8 | 18.45 | 18.47 | 18.23 | 38.06 | 38.24 | 37.60 |
| 384 | 12 | 27.03 | 27.03 | 26.72 | 58.94 | 59.27 | 58.09 |
| 384 | 16 | 37.47 | 37.51 | 36.77 | 79.40 | 79.70 | 78.36 |
| 384 | 24 | 55.02 | 55.25 | 54.56 | 123.06 | 123.32 | 121.71 |
| 384 | 32 | 77.22 | 77.54 | 76.48 | 167.99 | 168.34 | 167.10 |
| 384 | 64 | 157.21 | 157.53 | 155.69 | 335.31 | 335.96 | 333.65 |
| 384 | 128 | 337.82 | 338.55 | 335.23 | 640.65 | 641.04 | 639.38 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.51 | 1.51 | 1.49 |
| 128 | 2 | 2.05 | 2.06 | 2.01 |
| 128 | 4 | 3.00 | 3.01 | 2.94 |
| 128 | 8 | 5.06 | 5.08 | 5.05 |
| 128 | 12 | 6.71 | 6.78 | 6.63 |
| 128 | 16 | 8.83 | 8.84 | 8.75 |
| 128 | 24 | 13.38 | 13.39 | 13.16 |
| 128 | 32 | 17.61 | 17.63 | 17.50 |
| 128 | 64 | 36.49 | 36.55 | 36.16 |
| 128 | 128 | 80.34 | 80.39 | 79.62 |
| 384 | 1 | 2.81 | 2.82 | 2.77 |
| 384 | 2 | 4.20 | 4.23 | 4.12 |
| 384 | 4 | 7.62 | 7.66 | 7.53 |
| 384 | 8 | 15.13 | 15.15 | 14.97 |
| 384 | 12 | 21.74 | 21.87 | 21.56 |
| 384 | 16 | 28.83 | 29.00 | 28.70 |
| 384 | 24 | 47.51 | 47.58 | 47.12 |
| 384 | 32 | 61.31 | 61.50 | 60.79 |
| 384 | 64 | 126.97 | 127.06 | 126.69 |
| 384 | 128 | 256.27 | 256.61 | 255.09 |

### Inference Performance NVIDIA L40S

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA L40S.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.34 | 0.34 | 0.34 | 0.48 | 0.48 | 0.48 |
| 128 | 2 | 0.41 | 0.41 | 0.41 | 0.57 | 0.57 | 0.56 |
| 128 | 4 | 0.50 | 0.50 | 0.50 | 0.78 | 0.78 | 0.78 |
| 128 | 8 | 0.67 | 0.67 | 0.67 | 1.30 | 1.30 | 1.29 |
| 128 | 12 | 0.92 | 0.93 | 0.91 | 1.78 | 1.78 | 1.76 |
| 128 | 16 | 1.10 | 1.10 | 1.10 | 2.30 | 2.31 | 2.29 |
| 128 | 24 | 1.48 | 1.48 | 1.47 | 3.30 | 3.31 | 3.26 |
| 128 | 32 | 1.83 | 1.84 | 1.82 | 3.98 | 3.99 | 3.96 |
| 128 | 64 | 3.52 | 3.53 | 3.49 | 8.46 | 8.52 | 8.40 |
| 128 | 128 | 7.63 | 7.64 | 7.58 | 17.47 | 17.57 | 17.33 |
| 384 | 1 | 0.73 | 0.73 | 0.73 | 1.04 | 1.04 | 1.03 |
| 384 | 2 | 0.88 | 0.88 | 0.88 | 1.36 | 1.36 | 1.36 |
| 384 | 4 | 1.17 | 1.17 | 1.16 | 2.21 | 2.21 | 2.19 |
| 384 | 8 | 1.73 | 1.73 | 1.72 | 3.53 | 3.53 | 3.51 |
| 384 | 12 | 2.73 | 2.74 | 2.72 | 5.25 | 5.26 | 5.18 |
| 384 | 16 | 3.28 | 3.29 | 3.27 | 7.58 | 7.59 | 7.53 |
| 384 | 24 | 4.97 | 4.98 | 4.94 | 10.37 | 10.40 | 10.27 |
| 384 | 32 | 6.47 | 6.49 | 6.40 | 14.17 | 14.20 | 14.03 |
| 384 | 64 | 14.05 | 14.07 | 13.89 | 31.25 | 31.34 | 30.90 |
| 384 | 128 | 29.55 | 29.77 | 28.85 | 64.72 | 65.01 | 63.83 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.88 | 0.88 | 0.88 | 1.30 | 1.30 | 1.29 |
| 128 | 2 | 0.99 | 0.99 | 0.98 | 1.51 | 1.51 | 1.50 |
| 128 | 4 | 1.37 | 1.37 | 1.36 | 2.30 | 2.30 | 2.28 |
| 128 | 8 | 1.96 | 1.96 | 1.95 | 3.92 | 3.93 | 3.90 |
| 128 | 12 | 2.83 | 2.86 | 2.81 | 5.92 | 5.93 | 5.90 |
| 128 | 16 | 3.27 | 3.27 | 3.24 | 6.81 | 6.82 | 6.75 |
| 128 | 24 | 4.64 | 4.64 | 4.61 | 10.25 | 10.28 | 10.19 |
| 128 | 32 | 5.73 | 5.74 | 5.68 | 13.17 | 13.19 | 13.01 |
| 128 | 64 | 12.00 | 12.08 | 11.89 | 28.33 | 28.35 | 28.01 |
| 128 | 128 | 26.06 | 26.22 | 25.74 | 65.44 | 65.68 | 64.41 |
| 384 | 1 | 1.68 | 1.68 | 1.67 | 2.72 | 2.72 | 2.71 |
| 384 | 2 | 2.29 | 2.29 | 2.28 | 3.95 | 3.96 | 3.94 |
| 384 | 4 | 3.31 | 3.31 | 3.30 | 6.50 | 6.55 | 6.45 |
| 384 | 8 | 5.15 | 5.16 | 5.13 | 10.84 | 10.87 | 10.69 |
| 384 | 12 | 8.14 | 8.15 | 8.10 | 19.89 | 19.99 | 19.37 |
| 384 | 16 | 9.96 | 9.98 | 9.86 | 22.65 | 22.68 | 22.45 |
| 384 | 24 | 15.37 | 15.42 | 15.23 | 35.42 | 35.49 | 35.08 |
| 384 | 32 | 20.32 | 20.45 | 20.04 | 48.00 | 48.01 | 47.26 |
| 384 | 64 | 44.74 | 44.94 | 43.95 | 104.17 | 104.49 | 102.96 |
| 384 | 128 | 90.01 | 90.24 | 88.73 | 205.73 | 206.26 | 203.73 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.76 | 0.76 | 0.76 |
| 128 | 2 | 0.90 | 0.90 | 0.90 |
| 128 | 4 | 1.14 | 1.14 | 1.13 |
| 128 | 8 | 1.72 | 1.72 | 1.71 |
| 128 | 12 | 2.28 | 2.28 | 2.28 |
| 128 | 16 | 2.74 | 2.74 | 2.74 |
| 128 | 24 | 4.53 | 4.53 | 4.52 |
| 128 | 32 | 5.17 | 5.23 | 5.14 |
| 128 | 64 | 10.19 | 10.20 | 10.13 |
| 128 | 128 | 21.23 | 21.30 | 20.96 |
| 384 | 1 | 1.13 | 1.13 | 1.13 |
| 384 | 2 | 1.65 | 1.65 | 1.64 |
| 384 | 4 | 2.53 | 2.53 | 2.52 |
| 384 | 8 | 4.99 | 5.00 | 4.98 |
| 384 | 12 | 6.55 | 6.55 | 6.50 |
| 384 | 16 | 8.55 | 8.56 | 8.50 |
| 384 | 24 | 12.72 | 12.73 | 12.68 |
| 384 | 32 | 16.78 | 16.85 | 16.67 |
| 384 | 64 | 36.48 | 36.55 | 35.85 |
| 384 | 128 | 78.19 | 79.69 | 76.16 |

