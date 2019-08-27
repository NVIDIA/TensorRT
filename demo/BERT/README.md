[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# BERT Example using the TensorRT C++ API

This example demonstrates how to write a TensorRT model with custom plugins to implement the BERT encoder with SQuAD output layer.

As input, the code expects tokenized sentences, input masks and segment ids, as in the [reference Tensorflow implementation](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT).

This code was tested on NVIDIA V100 and T4 GPUs, and will only compile to those architectures.


## Prerequisites

To build the TensorRT OSS components, ensure you meet the following package requirements:

**System Packages**

* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Recommended versions:
  * [cuda-10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base) + cuDNN-7.5

* [GNU Make](https://ftp.gnu.org/gnu/make/) >= v4.1

* [CMake](https://github.com/Kitware/CMake/releases) >= v3.8

* [Python](<https://www.python.org/downloads/>)
  * Recommended versions:
  * [Python2](https://www.python.org/downloads/release/python-2715/) >= v2.7.15
  * [Python3](https://www.python.org/downloads/release/python-365/) >= v3.6.5

* [PIP](https://pypi.org/project/pip/#history) >= v19.0

* Essential libraries and utilities
  * [Git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [Wget](https://www.gnu.org/software/wget/faq.html#download), [Zlib](https://zlib.net/)


**Optional Packages**

* Containerized builds
  * [Docker](https://docs.docker.com/install/) >= 1.12
  * [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) >= 2.0


**TensorRT Release**

* [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-5x-download) v5.1.5


## Example Workflow

The example provides scripts to convert fine-tuned Tensorflow model checkpoints into a simple binary format that can be read by sample binary.

The high-level workflow consists of the following steps:

1. Download a pre-trained BERT SQuAD checkpoint from NGC model registry (See optional section if you would like to train your own model)
2. Convert the fine-tuned checkpoint into our simple format, described in the appendix (the original weights are assumed to be float32 values)
3. Generate a test input/output pair (input sequences are assumed to be int32 values)
4. Build and run the sample

### 1. Download a pre-trained BERT SQuAD checkpoint from NGC model registry
```
wget -O bert-base-squad1.1.zip https://api.ngc.nvidia.com/v2/models/nvidia/bert_tf_v1_1_base_fp32_128/versions/2/zip
unzip bert-base-squad1.1.zip -d squad_output_path
```

Below, we will refer to the location `<squad output path>/model.ckpt-<number>` as shell variable `CHECKPOINT` and the path to the folder that contains the `bert_config.json` as `BERT_PATH`.


#### (Optional) Downloading the BERT reference code and pre-trained language model, and running SQuAD Fine-tuning

Please follow the instructions in the [DeepLearningExamples repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) for fine-tuning SQuAD, which involves downloading the pre-trained language model as well as the SQuAD training data.

Then, in the scripts folder, there is `run_squad.sh` script, that adds a SQuAD-specific task layer to BERT and performs the fine-tuning.

This will create three files prefixed with `model.ckpt-<number>` that contain the fine-tuned model parameters, in the specified output directory.


### 2. Convert the fine-tuned checkpoint into a simple format
Python scripts in step 2 and 3 require Tensorflow on the system. We tested using tensorflow:19.06-py3 NGC container image.

The SQuAD fine-tuned Tensorflow checkpoint can be converted using the following command:

```
python helpers/convert_weights.py -m $CHECKPOINT -o <weight path>/filename
```

This will generate a file `<weight path>/<filename>.weights`. The path that contains the weights file, will be referred to as `WEIGHT_PATH`.


### 3. Generate an input/output pair

To run the sample on random inputs and compare the output to the reference Tensorflow implementation, the following command produces test inputs and outputs:

```python helpers/generate_dbg.py -f $CHECKPOINT -p $BERT_PATH -o $OUTPUT_PATH -s <seq.len.> -b <batch size>```

Please refer to the help of `generate_dbg.py` for more options.


### 4. Build and run the example

The C++ example was tested using TensorRT OSS docker container image created by following the instruction [in this link](https://github.com/NVIDIA/TensorRT#setting-up-the-build-environment)

This example uses `cmake` and can be built with the following steps:
```
mkdir build
cd build
cmake ..
make -j
```

This will produce an executable `sample_bert` in the `build` folder.

The binary `sample_bert` requires as arguments the paths that contain `bert_config.json` (from the pre-trained BERT checkpoint), `bert.weights` and `test_inputs.weights_int32` and `test_outputs.weights` as generated by the steps above.

```build/sample_bert -d $WEIGHT_PATH -d $OUTPUT_PATH --fp16 --nheads <num_attention_heads>```

`<num_attention_heads>` refers to the number of attention heads and can be found in the `bert_config.json`.

# Appendix

## Description of the binary format

The example expects weights and inputs in a simple tensor dictionary format.
It consists of an integer in the first line `N` denoting the number of entries in the dictionary.
Then there are `N` lines, each line following the format
`[tensor name: String] [element type: DataType] [number of dimensions D: int] [dim1, dim2, ..., dimD] [binary tensor data]\n`
DataType is the `nvinfer1` enumeration, that encodes types as numbers. E.g. `DataType::kFLOAT = 0` (float32) and `DataType::kINT32 = 3`.
The binary tensor data is dim1 * dim2 * ... * dimD * sizeof(type) bytes followed by a line break.
Methods to read this format can be found in `dataUtils.hpp`
