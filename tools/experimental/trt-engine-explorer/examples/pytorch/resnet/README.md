## TREx Blog ResNet QAT Example

This directory contains the code for the examples in the [TREx blog](https://developer.nvidia.com/blog/exploring-tensorrt-engines-with-trex/).<br>
Directory `A100` contains pre-generated JSON files which are provided as a shortcut so you can skip straight to using the `example_qat_resnet18.ipynb` notebook.
<br><br>
### Installation
The example uses PyTorch, TorchVision and Nvidia's [PyTorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization). PyTorch and TorchVision can be installed with:
```
python3 -m pip install torch torchvision
```

It is most reliable to install PyTorch Quantization Toolkit from source code so please follow [these](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization#from-source) instructions.


### Description
This example walks you through the process of optimizing a TensorRT engine created from QAT ResNet18.
File `resnet_example.py` contains code to generate several variations of QAT ResNet18, each one performing better than the previous.

* To generate the example ResNet ONNX models:
    ```
    python3 <path-to-trex>/examples/pytorch/resnet/resnet_example.py
    ```

    The script generates several ONNX files and writes them to the `<path-to-script>/generated/` directory.

* Then generate the JSON files:
    ```
    <path-to-trex>/examples/pytorch/resnet/process_resnet.sh
    ```

* Finally, review the results using the `example_qat_resnet18.ipynb` notebook.

