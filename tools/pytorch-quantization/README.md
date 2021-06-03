# Pytorch Quantization

PyTorch-Quantization is a toolkit for training and evaluating PyTorch models with simulated quantization. Quantization can be added to the model automatically, or manually, allowing the model to be tuned for accuracy and performance. Quantization is compatible with NVIDIAs high performance integer kernels which leverage integer Tensor Cores. The quantized model can be exported to ONNX and imported to an upcoming version of TensorRT.

## Install

#### Binaries

```bash
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

#### From Source

```bash
git clone https://github.com/NVIDIA/TensorRT.git
cd tools/pytorch-quantization
```

Install prerequisites
```bash
pip install -r requirements.txt
pip install torch
```

Build and install pytorch-quantization
```bash
python setup.py install
```

#### NGC Container

`pytorch-quantization` is preinstalled in NVIDIA NGC PyTorch container since 20.12, e.g. `nvcr.io/nvidian/pytorch:20.12-py3`

## Resources

* Pytorch Quantization Toolkit [userguide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)
* Quantization Basics [whitepaper](https://arxiv.org/abs/2004.09602)

