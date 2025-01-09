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
# release
clear && rm -rf build && mkdir build && cd build && cmake .. && make -j 20 && cd ..
# debug 
clear && rm -rf build && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j 20 && cd ..
```

# Excute
```shell
# e.g. sample_onnx_mnist  in root path
./build/sample_onnx_mnist -d ../data/tensorrt-sample-data/mnist/
# e.g. tbd

```

# 注意事项
```shell
# 1. data从/usr/src/tensorrt/data/中copy到自己的文件夹：~/sbx/data
# 2. 需要注意使用原生的环境编译，conda环境需要deactivate
https://stackoverflow.com/questions/54429210/how-do-i-prevent-conda-from-activating-the-base-environment-by-default
# 3. 直接在目录下make 得到的结果在../bin/中
# 4. 执行如下命令：
./build/sample_onnx_mnist -d ../data/tensorrt-sample-data/mnist/ --fp16
# 5. 如何ssh遇到如下问题:
ssh: connect to host github.com port 22: Connection refused
Try this: https://stackoverflow.com/questions/7953806/github-ssh-via-public-wifi-port-22-blocked/45473512#45473512
$ vim ~/.ssh/config and Add
Host github.com
  Hostname ssh.github.com
  Port 443
```
