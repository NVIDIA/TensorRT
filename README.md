[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Documentation](https://img.shields.io/badge/TensorRT-documentation-brightgreen.svg)](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)

# Description
This repo is folked from Tensorrt: https://github.com/NVIDIA/TensorRT.
Will be clean up only for samples modifying, building, running and studying.

# Environment
## Developing
1. tbd -- conda env
## Deployment -- c++ compiling and running
1. WSL2 + Unbuntu22.04: 5.15.153.1-microsoft-standard-WSL2
2. CUDA: 12.4.99
3. Tensorrt: 10.5.0

# CMD
```shell 
clear && rm -rf build && mkdir build && cd build && cmake .. && make -j 20 && cd ..
```

# Excute
```shell
# e.g. sample_onnx_mnist  in root path
./build/sample_onnx_mnist -d ../data/tensorrt-sample-data/mnist/
# e.g. tbd

```
