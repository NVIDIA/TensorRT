#!/bin/bash

pip install nvidia-pyindex
pip install onnx-graphsurgeon
pip install git+https://github.com/onnx/tensorflow-onnx.git
pip install tensorflow-gpu==2.8.0 tensorflow-datasets
pip install numpy pytest pytest-html graphviz
pip install .
