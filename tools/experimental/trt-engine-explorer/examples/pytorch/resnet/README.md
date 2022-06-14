## TREx Blog Example

This directory contains the code for the examples in the TREx blog.<br>
Directory `A100` contains pre-generated JSON files which are provided as a shortcut so you can skip straight to using the `example_qat_resnet18.ipynb` notebook.
<br><br>
### Installation
The example uses PyTorch,  TorchVision and Nvidia's [PyTorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization) which you can install with:
```
python3 -m pip install -r requirements.txt
```

### Description
This example walks you through the process of optimizing a TensorRT engine created from QAT ResNet18.
File `resnet_example.py` contains code to generate several variations of QAT ResNet18, each one performing better than the previous.

* To generate the example ResNet ONNX models:
    ```
    python3 trt-engine-explorer-OSS/examples/pytorch/resnet/resnet_example.py
    ```

    The script generates several ONNX files and writes them to the `<path-to-script>/generated/` directory.

* Then generate the collateral JSON files:
    ```
    ./trt-engine-explorer-OSS/examples/pytorch/resnet/process_resnet.sh
    ```

* Finally, review the results using the `example_qat_resnet18.ipynb` notebook.
