#!/bin/bash

set -e

TRT_VERSION="11.3.0.99"

usage() {
    echo "Usage: $0 [--x86 | --aarch64] [--cuda 13.4 | --cuda 12.9]"
    exit 1
}

ARCH=""
CUDA_VERSION="13.4"

while [ $# -gt 0 ]; do
    case "$1" in
        --x86)
            ARCH="x86_64"
            shift
            ;;
        --aarch64)
            ARCH="aarch64"
            shift
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [ -z "$ARCH" ]; then
    usage
fi

case "$CUDA_VERSION" in
    13.4|12.9)
        ;;
    *)
        usage
        ;;
esac

URL="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.3.0/tars/TensorRT-Enterprise-11.3.0.99-Linux-${ARCH}-cuda-${CUDA_VERSION}-Release-external.tar.zst"

echo "Downloading TensorRT package from: $URL"
cd /opt
wget "$URL" -O TensorRT-$TRT_VERSION.tar.zst
zstd -dc TensorRT-$TRT_VERSION.tar.zst | tar -x
rm -rf TensorRT-$TRT_VERSION.tar.zst
echo "TensorRT package downloaded and installed"
