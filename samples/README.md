# Cross Compilation Guide

This guide shows how to cross compile TensorRT samples for AArch64 QNX, Linux and Android platform under x86_64 Linux.

### Common Prerequisites

* Install the CUDA cross-platform toolkit for the the corresponding target, and set the environment variable `CUDA_INSTALL_DIR`

  ```shell
  export CUDA_INSTALL_DIR="your cuda install dir"
  ```

  `CUDA_INSTALL_DIR` is set to `/usr/local/cuda` by default.

* Install the cuDNN cross-platform libraries for the corresponding target, and set the environment variable `CUDNN_INSTALL_DIR`

  ```shell
  export CUDNN_INSTALL_DIR="your cudnn install dir"
  ```

  `CUDNN_INSTALL_DIR` is set to `CUDA_INSTALL_DIR` by default.

* Install the TensorRT cross compilation debian packages for the corresponding target.
  * QNX AArch64: libnvinfer-dev-cross-qnx, libnvinfer5-cross-qnx
  * Linux AArch64: libnvinfer-dev-cross-aarch64, libnvinfer5-cross-aarch64
  * Android AArch64: No debian packages are available.

  If you are using the tar file released by the TensorRT team, you can safely skip this step. The tar file release already includes the cross compile libraries so no additional packages are required.

### Build Samples for QNX AArch64

Download the QNX toolchain and export the following environment variables.

```shell
export QNX_HOST=/path/to/your/qnx/toolchains/host/linux/x86_64
export QNX_TARGET=/path/to/your/qnx/toolchain/target/qnx7
```

Build samples via

```shell
cd /path/to/TensorRT/samples
make TARGET=qnx
```

### Build Samples for Linux AArch64

Sample compilation for Linux aarch64 needs the corresponding g++ compiler, `aarch64-linux-gnu-g++`. In Ubuntu, this can be installed via

```shell
sudo apt-get install g++-aarch64-linux-gnu
```

Build samples via

```shell
cd /path/to/TensorRT/samples
make TARGET=aarch64
```

### Build Samples for Android AArch64

Download Android NDK(r16b) from https://developer.android.com/ndk/.  After downloading the NDK, create a standalone toolchain, for example

```shell
$NDK/build/tools/make_standalone_toolchain.py \
  --arch arm64 \
  --api 26 \
  --install-dir=/path/to/my-toolchain
```

You can check the details on https://developer.android.com/ndk/guides/standalone_toolchain.

Build samples via

```shell
cd /path/to/TensorRT/samples
make TARGET=android64 ANDROID_CC=/path/to/my-toolchain/bin/aarch64-linux-android-clang++
```
