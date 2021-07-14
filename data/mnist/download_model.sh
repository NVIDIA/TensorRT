#!/bin/bash
# Downloads the ONNX MNIST Model from the model zoo.
wget https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz
tar -xzvf mnist.tar.gz
# rename model to mnist.onnx
mv mnist/model.onnx mnist.onnx
# Delete MNIST data set and archive
rm -r mnist
rm -r mnist.tar*

