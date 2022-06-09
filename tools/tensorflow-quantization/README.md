# <span style="color:green"> **NVIDIA TensorFlow 2.x Quantization** </span>

This TensorFlow 2.x Quantization toolkit quantizes (inserts Q/DQ nodes) TensorFlow 2.x Keras models for Quantization-Aware Training (QAT).
We follow NVIDIA's QAT recipe, which leads to optimal model acceleration with [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) on NVIDIA GPUs and hardware accelerators. 

### Features

- Implements [NVIDIA Quantization](https://arxiv.org/pdf/2004.09602.pdf) recipe.
- Supports fully automated or manual insertion of Quantization and DeQuantization (QDQ) nodes in the TensorFlow 2.x model with minimal code. 
- Can easily to add support for new layers.
- Quantization behavior can be set programmatically.
- Implements automatic tests for popular architecture blocks such as residual and inception.
- Offers utilities for TensorFlow 2.x to TensorRT conversion via ONNX.
- Includes [example workflows](examples).

## Dependencies

**Python** >= 3.8  
**TensorFlow** >= 2.8  
**tf2onnx** >= 1.10.0  
**onnx-graphsurgeon**  
**pytest**  
**pytest-html**  
**TensorRT** (optional) >= 8.4 GA

## Installation

### Docker

Latest TensorFlow 2.x [docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow/tags) from NGC is recommended.

```bash
$ cd ~/
$ git clone ssh://git@gitlab-master.nvidia.com:12051/TensorRT/Tools/tensorflow-quantization.git
$ docker pull nvcr.io/nvidia/tensorflow:22.03-tf2-py3
$ docker run -it --runtime=nvidia --gpus all -v ~/tensorflow-quantization:/home/tensorflow-quantization nvcr.io/nvidia/tensorflow:22.03-tf2-py3 /bin/bash 
```
After last command, you will be placed in `/workspace` directory inside the running docker container whereas `tensorflow-quantization` repo is mounted in `/home` directory.

```bash
$ cd ~/home/tensorflow-quantization
$ ./install.sh
$ cd tests
$ python3 -m pytest quantize_test.py -rP
```
If all tests pass, installation is successful.

### Local

```bash
$ cd ~/
$ git clone ssh://git@gitlab-master.nvidia.com:12051/TensorRT/Tools/tensorflow-quantization.git
$ cd tensorflow-quantization
$ ./install.sh
$ cd tests
$ python3 -m pytest quantize_test.py -rP
```

If all tests pass, installation is successful.

## Documentation

TensorFlow 2.x Quantization toolkit [userguide](http://tensorrt.gitlab-master-pages.nvidia.com/Tools/tensorflow-quantization).

## Known limitations

1. Only Quantization Aware Training (QAT) is supported as a quantization method.
2. Only Functional and Sequential Keras models are supported. Original Keras layers are wrapped into quantized layers using TensorFlow's [clone_model](https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model) method, which doesn't support subclassed models.
3. Saving quantized version of few layers may not be supported in `TensorFlow < 2.8`.
4. `TensorRT >= 8.2` is recommended since it supports engine visualization. 

## Resources

- [GTC 2022 talk](https://www.nvidia.com/gtc/session-catalog/?search=dheeraj%20&search=dheeraj+#/session/1636418253677001loTP)
- Quantization Basics [whitepaper](https://arxiv.org/abs/2004.09602)
