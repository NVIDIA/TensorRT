## TensorFlow ResNet QAT Example

This directory contains the code for quantizing a TensorFlow QAT ResNet50 model. The example uses TensorFlow and NVIDIA's [TensorFlow Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/tensorflow-quantization) which you can install with:
```
git clone https://github.com/NVIDIA/TensorRT.git
cd TensorRT/tools/tensorflow-quantization
./install.sh
```

Start by generating the example ResNet ONNX models:
```
python3 <path-to-trex>/examples/tensorflow/resnet/resnet_example.py
```

Then generate the JSON files:
```
<path-to-trex>/examples/tensorflow/resnet/process_resnet.sh
```

Finally, review the results in the TREx notebook [*example_qat_resnet.ipynb*](example_qat_resnet.ipynb).
