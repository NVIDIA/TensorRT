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
| 128 | 1 | 0.53 | 0.68 | 0.54 | 0.79 | 0.79 | 0.64 |
| 128 | 2 | 0.76 | 0.76 | 0.60 | 0.72 | 0.91 | 0.72 |
| 128 | 4 | 0.73 | 0.92 | 0.73 | 1.03 | 1.04 | 0.93 |
| 128 | 8 | 0.94 | 1.20 | 0.95 | 1.31 | 1.31 | 1.31 |
| 128 | 12 | 1.19 | 1.20 | 1.19 | 1.72 | 1.73 | 1.72 |
| 128 | 16 | 1.33 | 1.71 | 1.34 | 2.07 | 2.08 | 2.05 |
| 128 | 24 | 1.82 | 1.82 | 1.81 | 3.04 | 3.07 | 3.01 |
| 128 | 32 | 2.23 | 2.24 | 2.23 | 3.90 | 3.93 | 3.86 |
| 128 | 64 | 4.15 | 4.17 | 4.12 | 7.62 | 7.70 | 7.57 |
| 128 | 128 | 8.11 | 8.12 | 8.03 | 15.34 | 15.35 | 15.20 |
| 384 | 1 | 1.13 | 1.45 | 1.13 | 1.24 | 1.25 | 1.24 |
| 384 | 2 | 1.31 | 1.31 | 1.31 | 1.54 | 1.98 | 1.55 |
| 384 | 4 | 1.66 | 1.66 | 1.66 | 2.12 | 2.12 | 2.12 |
| 384 | 8 | 2.21 | 2.21 | 2.20 | 3.34 | 3.36 | 3.32 |
| 384 | 12 | 3.32 | 3.32 | 3.31 | 4.78 | 4.82 | 4.77 |
| 384 | 16 | 4.01 | 4.01 | 4.00 | 6.37 | 6.44 | 6.35 |
| 384 | 24 | 5.71 | 5.71 | 5.70 | 9.47 | 9.49 | 9.39 |
| 384 | 32 | 7.64 | 7.64 | 7.63 | 13.00 | 13.04 | 12.85 |
| 384 | 64 | 14.87 | 14.88 | 14.73 | 25.12 | 25.14 | 24.78 |
| 384 | 128 | 28.96 | 28.97 | 28.70 | 48.93 | 49.13 | 48.57 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.22 | 1.23 | 1.22 | 1.54 | 1.91 | 1.55 |
| 128 | 2 | 1.42 | 1.42 | 1.41 | 1.82 | 1.82 | 1.82 |
| 128 | 4 | 1.78 | 2.06 | 1.79 | 2.50 | 2.50 | 2.50 |
| 128 | 8 | 2.64 | 2.64 | 2.64 | 3.98 | 3.98 | 3.98 |
| 128 | 12 | 3.09 | 3.09 | 3.08 | 5.02 | 5.07 | 4.99 |
| 128 | 16 | 4.09 | 4.09 | 4.08 | 6.93 | 6.94 | 6.86 |
| 128 | 24 | 5.28 | 5.28 | 5.27 | 9.64 | 9.68 | 9.56 |
| 128 | 32 | 7.01 | 7.01 | 6.95 | 12.92 | 13.07 | 12.85 |
| 128 | 64 | 12.86 | 12.86 | 12.73 | 24.79 | 25.07 | 24.59 |
| 128 | 128 | 25.03 | 25.26 | 24.99 | 49.12 | 49.28 | 48.83 |
| 384 | 1 | 2.55 | 2.55 | 2.55 | 2.96 | 2.96 | 2.95 |
| 384 | 2 | 3.04 | 3.04 | 3.03 | 3.90 | 3.90 | 3.90 |
| 384 | 4 | 4.01 | 4.02 | 4.01 | 5.68 | 5.74 | 5.67 |
| 384 | 8 | 7.18 | 7.18 | 7.17 | 11.13 | 11.13 | 11.01 |
| 384 | 12 | 9.14 | 9.15 | 9.13 | 15.43 | 15.44 | 15.32 |
| 384 | 16 | 12.28 | 12.28 | 12.27 | 21.14 | 21.15 | 20.90 |
| 384 | 24 | 17.68 | 17.68 | 17.54 | 30.98 | 31.02 | 30.68 |
| 384 | 32 | 23.24 | 23.24 | 23.02 | 41.11 | 41.20 | 40.58 |
| 384 | 64 | 44.86 | 45.13 | 44.78 | 79.25 | 79.68 | 79.10 |
| 384 | 128 | 87.82 | 87.84 | 87.69 | 156.70 | 157.02 | 155.61 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.11 | 1.40 | 1.11 |
| 128 | 2 | 1.33 | 1.33 | 1.33 |
| 128 | 4 | 1.78 | 1.78 | 1.78 |
| 128 | 8 | 2.54 | 2.54 | 2.53 |
| 128 | 12 | 2.97 | 2.97 | 2.97 |
| 128 | 16 | 3.99 | 3.99 | 3.98 |
| 128 | 24 | 4.91 | 4.91 | 4.90 |
| 128 | 32 | 7.13 | 7.13 | 7.12 |
| 128 | 64 | 11.61 | 11.62 | 11.60 |
| 128 | 128 | 21.22 | 21.32 | 21.09 |
| 384 | 1 | 1.71 | 2.15 | 1.71 |
| 384 | 2 | 2.21 | 2.21 | 2.21 |
| 384 | 4 | 3.47 | 3.48 | 3.47 |
| 384 | 8 | 5.74 | 5.74 | 5.74 |
| 384 | 12 | 8.21 | 8.21 | 8.20 |
| 384 | 16 | 10.33 | 10.34 | 10.32 |
| 384 | 24 | 14.68 | 14.69 | 14.67 |
| 384 | 32 | 18.73 | 18.74 | 18.72 |
| 384 | 64 | 35.77 | 35.78 | 35.49 |
| 384 | 128 | 67.78 | 67.95 | 67.63 |

### Inference Performance NVIDIA L4

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA L4.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.61 | 0.61 | 0.60 | 1.01 | 1.01 | 1.00 |
| 128 | 2 | 0.79 | 0.80 | 0.77 | 1.32 | 1.35 | 1.31 |
| 128 | 4 | 1.14 | 1.15 | 1.12 | 2.22 | 2.23 | 2.14 |
| 128 | 8 | 1.94 | 1.96 | 1.90 | 3.66 | 3.67 | 3.63 |
| 128 | 12 | 2.67 | 2.67 | 2.61 | 5.34 | 5.34 | 5.26 |
| 128 | 16 | 3.37 | 3.38 | 3.32 | 6.69 | 6.69 | 6.64 |
| 128 | 24 | 4.84 | 4.84 | 4.75 | 10.53 | 10.64 | 10.50 |
| 128 | 32 | 6.21 | 6.28 | 6.13 | 13.91 | 13.91 | 13.72 |
| 128 | 64 | 13.40 | 13.60 | 13.20 | 31.48 | 31.53 | 31.01 |
| 128 | 128 | 28.42 | 28.68 | 27.84 | 70.60 | 71.10 | 69.25 |
| 384 | 1 | 1.27 | 1.27 | 1.27 | 2.08 | 2.09 | 2.07 |
| 384 | 2 | 1.84 | 1.84 | 1.82 | 3.15 | 3.19 | 3.11 |
| 384 | 4 | 2.94 | 2.94 | 2.91 | 5.68 | 5.75 | 5.63 |
| 384 | 8 | 5.53 | 5.55 | 5.42 | 11.45 | 11.59 | 11.32 |
| 384 | 12 | 8.21 | 8.31 | 8.07 | 17.16 | 17.36 | 17.00 |
| 384 | 16 | 10.96 | 11.07 | 10.80 | 23.20 | 23.50 | 22.81 |
| 384 | 24 | 16.71 | 16.74 | 16.55 | 39.82 | 40.46 | 38.15 |
| 384 | 32 | 22.82 | 23.00 | 22.63 | 50.56 | 50.89 | 50.14 |
| 384 | 64 | 49.66 | 50.18 | 48.40 | 104.90 | 105.55 | 103.81 |
| 384 | 128 | 104.78 | 105.09 | 103.96 | 208.20 | 208.70 | 206.93 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.79 | 1.80 | 1.77 | 3.11 | 3.11 | 3.09 |
| 128 | 2 | 2.49 | 2.49 | 2.43 | 4.35 | 4.37 | 4.33 |
| 128 | 4 | 3.62 | 3.70 | 3.60 | 6.86 | 6.89 | 6.78 |
| 128 | 8 | 6.26 | 6.31 | 6.24 | 12.85 | 12.91 | 12.73 |
| 128 | 12 | 8.40 | 8.41 | 8.28 | 18.42 | 18.43 | 18.33 |
| 128 | 16 | 11.23 | 11.24 | 11.12 | 25.18 | 25.19 | 25.10 |
| 128 | 24 | 15.95 | 16.09 | 15.90 | 35.67 | 35.67 | 35.47 |
| 128 | 32 | 21.26 | 21.31 | 20.91 | 48.92 | 49.21 | 48.26 |
| 128 | 64 | 44.10 | 44.11 | 43.92 | 108.81 | 109.12 | 107.18 |
| 128 | 128 | 94.22 | 95.02 | 92.65 | 217.32 | 219.58 | 212.68 |
| 384 | 1 | 3.41 | 3.43 | 3.39 | 6.55 | 6.57 | 6.36 |
| 384 | 2 | 5.55 | 5.56 | 5.46 | 10.34 | 10.35 | 10.18 |
| 384 | 4 | 9.69 | 9.79 | 9.53 | 20.66 | 20.95 | 19.94 |
| 384 | 8 | 18.08 | 18.19 | 17.92 | 38.41 | 39.30 | 37.62 |
| 384 | 12 | 26.20 | 26.44 | 26.11 | 60.38 | 60.91 | 58.67 |
| 384 | 16 | 36.33 | 36.41 | 36.02 | 81.66 | 82.16 | 80.52 |
| 384 | 24 | 53.54 | 53.61 | 53.08 | 123.01 | 123.34 | 122.10 |
| 384 | 32 | 75.01 | 75.43 | 74.40 | 170.40 | 171.03 | 169.12 |
| 384 | 64 | 157.97 | 158.62 | 155.87 | 349.25 | 351.53 | 344.76 |
| 384 | 128 | 330.88 | 331.87 | 328.27 | 632.85 | 633.88 | 629.74 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.49 | 1.49 | 1.48 |
| 128 | 2 | 2.03 | 2.03 | 1.99 |
| 128 | 4 | 2.99 | 3.00 | 2.93 |
| 128 | 8 | 5.00 | 5.07 | 4.99 |
| 128 | 12 | 6.69 | 6.72 | 6.58 |
| 128 | 16 | 8.77 | 8.84 | 8.66 |
| 128 | 24 | 13.28 | 13.30 | 13.14 |
| 128 | 32 | 17.41 | 17.44 | 17.26 |
| 128 | 64 | 35.73 | 36.07 | 35.49 |
| 128 | 128 | 79.03 | 79.15 | 78.47 |
| 384 | 1 | 2.78 | 2.79 | 2.72 |
| 384 | 2 | 4.10 | 4.12 | 4.06 |
| 384 | 4 | 7.57 | 7.58 | 7.45 |
| 384 | 8 | 15.03 | 15.10 | 14.86 |
| 384 | 12 | 21.52 | 21.69 | 21.31 |
| 384 | 16 | 28.29 | 28.33 | 28.10 |
| 384 | 24 | 46.83 | 47.09 | 46.29 |
| 384 | 32 | 60.29 | 60.47 | 59.37 |
| 384 | 64 | 125.58 | 125.64 | 125.24 |
| 384 | 128 | 253.46 | 253.90 | 252.28 |

### Inference Performance NVIDIA L40S

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA L40S.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.33 | 0.33 | 0.33 | 0.48 | 0.48 | 0.48 |
| 128 | 2 | 0.41 | 0.41 | 0.41 | 0.57 | 0.57 | 0.57 |
| 128 | 4 | 0.50 | 0.51 | 0.50 | 0.78 | 0.78 | 0.78 |
| 128 | 8 | 0.67 | 0.67 | 0.67 | 1.33 | 1.33 | 1.32 |
| 128 | 12 | 0.91 | 0.91 | 0.91 | 1.75 | 1.76 | 1.73 |
| 128 | 16 | 1.10 | 1.10 | 1.09 | 2.29 | 2.29 | 2.28 |
| 128 | 24 | 1.48 | 1.49 | 1.47 | 3.30 | 3.31 | 3.27 |
| 128 | 32 | 1.84 | 1.84 | 1.83 | 3.98 | 3.99 | 3.97 |
| 128 | 64 | 3.61 | 3.66 | 3.56 | 8.64 | 8.70 | 8.51 |
| 128 | 128 | 7.92 | 7.99 | 7.82 | 18.78 | 18.82 | 18.45 |
| 384 | 1 | 0.73 | 0.73 | 0.73 | 1.11 | 1.12 | 1.10 |
| 384 | 2 | 0.88 | 0.88 | 0.88 | 1.39 | 1.39 | 1.38 |
| 384 | 4 | 1.17 | 1.17 | 1.17 | 2.19 | 2.20 | 2.19 |
| 384 | 8 | 1.74 | 1.74 | 1.73 | 3.53 | 3.53 | 3.50 |
| 384 | 12 | 2.75 | 2.75 | 2.73 | 5.32 | 5.33 | 5.29 |
| 384 | 16 | 3.33 | 3.33 | 3.31 | 7.62 | 7.64 | 7.57 |
| 384 | 24 | 4.97 | 4.98 | 4.95 | 10.53 | 10.57 | 10.40 |
| 384 | 32 | 6.55 | 6.57 | 6.48 | 14.36 | 14.47 | 14.20 |
| 384 | 64 | 14.27 | 14.37 | 14.07 | 33.31 | 33.51 | 32.65 |
| 384 | 128 | 30.38 | 30.52 | 29.73 | 67.34 | 68.04 | 66.06 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.89 | 0.89 | 0.88 | 1.30 | 1.30 | 1.29 |
| 128 | 2 | 0.97 | 0.98 | 0.97 | 1.45 | 1.45 | 1.44 |
| 128 | 4 | 1.36 | 1.36 | 1.35 | 2.30 | 2.30 | 2.29 |
| 128 | 8 | 1.94 | 1.96 | 1.93 | 3.89 | 3.90 | 3.88 |
| 128 | 12 | 2.82 | 2.82 | 2.80 | 5.89 | 5.90 | 5.85 |
| 128 | 16 | 3.26 | 3.27 | 3.24 | 6.85 | 6.86 | 6.80 |
| 128 | 24 | 4.62 | 4.63 | 4.59 | 10.72 | 10.73 | 10.64 |
| 128 | 32 | 5.74 | 5.76 | 5.70 | 13.22 | 13.23 | 13.04 |
| 128 | 64 | 12.18 | 12.20 | 11.97 | 29.42 | 29.59 | 28.89 |
| 128 | 128 | 26.68 | 26.86 | 26.23 | 68.72 | 69.05 | 67.12 |
| 384 | 1 | 1.68 | 1.68 | 1.68 | 2.78 | 2.78 | 2.77 |
| 384 | 2 | 2.31 | 2.31 | 2.30 | 3.95 | 3.95 | 3.94 |
| 384 | 4 | 3.29 | 3.30 | 3.29 | 6.57 | 6.58 | 6.50 |
| 384 | 8 | 5.16 | 5.17 | 5.13 | 10.89 | 10.90 | 10.79 |
| 384 | 12 | 8.16 | 8.17 | 8.10 | 19.81 | 19.91 | 19.31 |
| 384 | 16 | 9.90 | 9.93 | 9.80 | 23.34 | 23.51 | 23.10 |
| 384 | 24 | 15.60 | 15.62 | 15.39 | 37.37 | 37.48 | 36.93 |
| 384 | 32 | 20.66 | 20.73 | 20.33 | 50.13 | 50.34 | 49.52 |
| 384 | 64 | 46.31 | 46.53 | 45.39 | 111.74 | 111.98 | 110.14 |
| 384 | 128 | 93.80 | 94.04 | 92.33 | 213.05 | 214.15 | 210.25 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.76 | 0.76 | 0.76 |
| 128 | 2 | 0.91 | 0.91 | 0.91 |
| 128 | 4 | 1.13 | 1.13 | 1.13 |
| 128 | 8 | 1.70 | 1.70 | 1.70 |
| 128 | 12 | 2.26 | 2.26 | 2.25 |
| 128 | 16 | 2.72 | 2.72 | 2.71 |
| 128 | 24 | 4.54 | 4.55 | 4.52 |
| 128 | 32 | 5.14 | 5.16 | 5.10 |
| 128 | 64 | 10.07 | 10.08 | 10.01 |
| 128 | 128 | 21.57 | 21.67 | 21.21 |
| 384 | 1 | 1.13 | 1.13 | 1.13 |
| 384 | 2 | 1.64 | 1.65 | 1.62 |
| 384 | 4 | 2.51 | 2.51 | 2.50 |
| 384 | 8 | 5.02 | 5.03 | 4.99 |
| 384 | 12 | 6.43 | 6.43 | 6.41 |
| 384 | 16 | 8.47 | 8.49 | 8.41 |
| 384 | 24 | 12.62 | 12.65 | 12.54 |
| 384 | 32 | 16.88 | 16.91 | 16.74 |
| 384 | 64 | 36.62 | 36.71 | 36.12 |
| 384 | 128 | 79.88 | 80.18 | 77.33 |

