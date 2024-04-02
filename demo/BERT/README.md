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
|Python|>=3.6|
|TensorRT|8.5.1|
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
| 128 | 1 | 0.64 | 0.69 | 0.56 | 0.79 | 0.79 | 0.63 |
| 128 | 2 | 0.78 | 0.78 | 0.62 | 0.80 | 0.80 | 0.73 |
| 128 | 4 | 0.74 | 0.74 | 0.74 | 1.12 | 1.20 | 0.95 |
| 128 | 8 | 1.22 | 1.23 | 0.96 | 1.31 | 1.31 | 1.31 |
| 128 | 12 | 1.29 | 1.30 | 1.21 | 1.70 | 1.70 | 1.70 |
| 128 | 16 | 1.34 | 1.34 | 1.34 | 2.10 | 2.10 | 2.08 |
| 128 | 24 | 1.83 | 1.84 | 1.83 | 3.07 | 3.08 | 3.04 |
| 128 | 32 | 2.25 | 2.26 | 2.25 | 3.95 | 3.95 | 3.92 |
| 128 | 64 | 4.19 | 4.20 | 4.17 | 7.68 | 7.74 | 7.63 |
| 128 | 128 | 8.15 | 8.16 | 8.10 | 15.45 | 15.46 | 15.30 |
| 384 | 1 | 1.14 | 1.46 | 1.15 | 1.26 | 1.62 | 1.26 |
| 384 | 2 | 1.32 | 1.32 | 1.32 | 1.55 | 1.55 | 1.55 |
| 384 | 4 | 1.68 | 1.72 | 1.68 | 2.11 | 2.11 | 2.11 |
| 384 | 8 | 2.22 | 2.23 | 2.22 | 3.38 | 3.42 | 3.35 |
| 384 | 12 | 3.34 | 3.34 | 3.34 | 4.84 | 4.86 | 4.81 |
| 384 | 16 | 4.02 | 4.03 | 4.02 | 6.41 | 6.41 | 6.39 |
| 384 | 24 | 5.73 | 5.73 | 5.73 | 9.47 | 9.47 | 9.36 |
| 384 | 32 | 7.75 | 7.77 | 7.68 | 13.05 | 13.12 | 12.92 |
| 384 | 64 | 14.96 | 14.96 | 14.85 | 25.24 | 25.36 | 24.93 |
| 384 | 128 | 29.13 | 29.14 | 28.89 | 49.27 | 49.37 | 48.84 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.24 | 1.24 | 1.23 | 1.56 | 1.56 | 1.56 |
| 128 | 2 | 1.44 | 1.83 | 1.45 | 1.83 | 1.83 | 1.83 |
| 128 | 4 | 1.78 | 1.78 | 1.78 | 2.55 | 2.56 | 2.55 |
| 128 | 8 | 2.66 | 2.66 | 2.66 | 3.96 | 3.97 | 3.93 |
| 128 | 12 | 3.11 | 3.11 | 3.10 | 5.07 | 5.12 | 5.05 |
| 128 | 16 | 4.07 | 4.07 | 4.06 | 6.96 | 6.97 | 6.91 |
| 128 | 24 | 5.31 | 5.32 | 5.31 | 9.72 | 9.82 | 9.63 |
| 128 | 32 | 7.04 | 7.07 | 7.02 | 13.00 | 13.04 | 12.95 |
| 128 | 64 | 12.96 | 12.96 | 12.86 | 24.90 | 25.07 | 24.71 |
| 128 | 128 | 25.20 | 25.21 | 25.16 | 49.29 | 49.55 | 48.86 |
| 384 | 1 | 2.57 | 2.57 | 2.57 | 2.98 | 2.98 | 2.98 |
| 384 | 2 | 3.06 | 3.07 | 3.06 | 3.93 | 3.93 | 3.92 |
| 384 | 4 | 4.03 | 4.03 | 4.03 | 5.78 | 5.79 | 5.74 |
| 384 | 8 | 7.20 | 7.21 | 7.19 | 11.16 | 11.19 | 11.04 |
| 384 | 12 | 9.18 | 9.18 | 9.17 | 15.51 | 15.51 | 15.39 |
| 384 | 16 | 12.34 | 12.34 | 12.33 | 21.25 | 21.25 | 21.03 |
| 384 | 24 | 17.74 | 17.79 | 17.69 | 31.13 | 31.14 | 30.82 |
| 384 | 32 | 23.37 | 23.37 | 23.16 | 41.26 | 41.43 | 40.83 |
| 384 | 64 | 45.08 | 45.09 | 45.01 | 79.88 | 80.21 | 79.18 |
| 384 | 128 | 88.34 | 88.37 | 88.06 | 156.43 | 157.17 | 155.47 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.17 | 1.48 | 1.18 |
| 128 | 2 | 1.49 | 1.88 | 1.50 |
| 128 | 4 | 1.79 | 1.79 | 1.79 |
| 128 | 8 | 2.54 | 2.54 | 2.53 |
| 128 | 12 | 2.95 | 2.95 | 2.94 |
| 128 | 16 | 3.97 | 3.97 | 3.96 |
| 128 | 24 | 4.91 | 4.91 | 4.90 |
| 128 | 32 | 6.90 | 6.92 | 6.86 |
| 128 | 64 | 11.61 | 11.64 | 11.59 |
| 128 | 128 | 21.34 | 21.35 | 21.21 |
| 384 | 1 | 1.71 | 1.72 | 1.71 |
| 384 | 2 | 2.21 | 2.21 | 2.21 |
| 384 | 4 | 3.47 | 3.47 | 3.47 |
| 384 | 8 | 5.75 | 5.75 | 5.74 |
| 384 | 12 | 8.37 | 8.38 | 8.35 |
| 384 | 16 | 10.39 | 10.40 | 10.37 |
| 384 | 24 | 14.61 | 14.62 | 14.59 |
| 384 | 32 | 18.80 | 18.96 | 18.78 |
| 384 | 64 | 35.90 | 35.92 | 35.62 |
| 384 | 128 | 67.74 | 67.77 | 67.60 |

#### Inference performance: NVIDIA A30

Results were obtained by running `scripts/inference_benchmark.sh --gpu Ampere` on NVIDIA A30.

##### BERT Base

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 0.88 | 0.88 | 0.61 | 0.78 | 1.14 | 0.79 |
| 128 | 2 | 1.03 | 1.04 | 0.77 | 0.97 | 1.45 | 0.98 |
| 128 | 4 | 1.04 | 1.56 | 1.05 | 1.43 | 1.44 | 1.41 |
| 128 | 8 | 1.44 | 1.46 | 1.43 | 2.43 | 2.44 | 2.41 |
| 128 | 12 | 1.92 | 1.92 | 1.91 | 3.44 | 3.45 | 3.39 |
| 128 | 16 | 2.38 | 2.43 | 2.35 | 4.36 | 4.37 | 4.28 |
| 128 | 24 | 3.47 | 3.50 | 3.44 | 6.56 | 6.65 | 6.48 |
| 128 | 32 | 4.42 | 4.45 | 4.38 | 8.42 | 8.58 | 8.36 |
| 128 | 64 | 8.58 | 8.66 | 8.49 | 16.58 | 16.60 | 16.40 |
| 128 | 128 | 16.56 | 16.62 | 16.39 | 32.13 | 32.30 | 31.93 |
| 384 | 1 | 1.31 | 2.01 | 1.32 | 1.63 | 1.63 | 1.62 |
| 384 | 2 | 1.67 | 1.67 | 1.66 | 2.29 | 2.35 | 2.26 |
| 384 | 4 | 2.29 | 2.34 | 2.27 | 3.74 | 3.77 | 3.71 |
| 384 | 8 | 4.23 | 4.24 | 4.20 | 7.25 | 7.30 | 7.15 |
| 384 | 12 | 6.05 | 6.10 | 6.00 | 10.21 | 10.27 | 10.12 |
| 384 | 16 | 8.07 | 8.11 | 8.02 | 13.97 | 14.05 | 13.84 |
| 384 | 24 | 11.85 | 11.86 | 11.71 | 20.31 | 20.42 | 20.16 |
| 384 | 32 | 15.45 | 15.47 | 15.29 | 26.86 | 27.04 | 26.65 |
| 384 | 64 | 30.49 | 30.74 | 30.25 | 52.21 | 52.34 | 51.75 |
| 384 | 128 | 60.21 | 60.48 | 59.56 | 103.20 | 103.58 | 102.66 |

##### BERT Large

| Sequence Length | Batch Size | INT8 Latency (ms) |               |         | FP16 Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.46 | 1.46 | 1.45 | 2.01 | 2.01 | 2.01 |
| 128 | 2 | 1.83 | 1.85 | 1.83 | 2.80 | 2.83 | 2.75 |
| 128 | 4 | 2.71 | 2.71 | 2.69 | 4.34 | 4.36 | 4.29 |
| 128 | 8 | 4.33 | 4.35 | 4.28 | 8.12 | 8.20 | 8.03 |
| 128 | 12 | 5.71 | 5.72 | 5.61 | 10.65 | 10.65 | 10.51 |
| 128 | 16 | 7.62 | 7.64 | 7.55 | 14.57 | 14.66 | 14.55 |
| 128 | 24 | 10.58 | 10.62 | 10.46 | 20.64 | 20.79 | 20.45 |
| 128 | 32 | 14.18 | 14.26 | 13.99 | 28.17 | 28.31 | 28.01 |
| 128 | 64 | 26.87 | 27.00 | 26.61 | 53.44 | 53.71 | 53.31 |
| 128 | 128 | 52.36 | 52.71 | 51.90 | 105.42 | 105.95 | 104.96 |
| 384 | 1 | 3.33 | 3.33 | 3.33 | 4.23 | 4.24 | 4.19 |
| 384 | 2 | 4.26 | 4.26 | 4.23 | 6.63 | 6.65 | 6.57 |
| 384 | 4 | 7.26 | 7.26 | 7.25 | 12.00 | 12.06 | 11.88 |
| 384 | 8 | 12.91 | 12.99 | 12.83 | 22.61 | 22.69 | 22.45 |
| 384 | 12 | 18.73 | 18.85 | 18.53 | 33.43 | 33.64 | 33.28 |
| 384 | 16 | 24.06 | 24.22 | 24.02 | 44.35 | 44.64 | 44.06 |
| 384 | 24 | 35.83 | 35.95 | 35.49 | 64.84 | 64.90 | 64.78 |
| 384 | 32 | 47.05 | 47.27 | 46.73 | 85.89 | 86.17 | 85.11 |
| 384 | 64 | 92.09 | 92.32 | 91.34 | 168.09 | 168.48 | 167.24 |
| 384 | 128 | 180.47 | 180.90 | 179.75 | 330.71 | 331.31 | 329.53 |

##### Megatron Large with Sparsity

| Sequence Length | Batch Size | INT8 QAT Latency (ms) |               |         |
|-----------------|------------|-----------------|-----------------|---------|
|                 |            | 95th Percentile | 99th Percentile | Average |
| 128 | 1 | 1.44 | 1.45 | 1.44 |
| 128 | 2 | 1.84 | 1.84 | 1.84 |
| 128 | 4 | 2.76 | 2.76 | 2.75 |
| 128 | 8 | 4.12 | 4.12 | 4.11 |
| 128 | 12 | 5.26 | 5.28 | 5.22 |
| 128 | 16 | 7.52 | 7.52 | 7.51 |
| 128 | 24 | 9.97 | 9.99 | 9.89 |
| 128 | 32 | 12.84 | 12.85 | 12.80 |
| 128 | 64 | 24.35 | 24.46 | 24.15 |
| 128 | 128 | 46.38 | 46.60 | 45.96 |
| 384 | 1 | 2.37 | 2.37 | 2.36 |
| 384 | 2 | 3.88 | 3.88 | 3.87 |
| 384 | 4 | 6.10 | 6.11 | 6.05 |
| 384 | 8 | 11.60 | 11.63 | 11.49 |
| 384 | 12 | 15.73 | 15.78 | 15.64 |
| 384 | 16 | 20.95 | 21.01 | 20.90 |
| 384 | 24 | 29.83 | 29.93 | 29.71 |
| 384 | 32 | 40.01 | 40.09 | 39.75 |
| 384 | 64 | 76.46 | 76.67 | 76.28 |
| 384 | 128 | 148.96 | 149.23 | 148.11 |

