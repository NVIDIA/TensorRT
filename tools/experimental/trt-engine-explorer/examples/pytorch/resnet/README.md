## TREx Blog ResNet QAT Example

This directory contains the code for the examples in the [TREx blog](https://developer.nvidia.com/blog/exploring-tensorrt-engines-with-trex/).<br>
Directory `A100` contains pre-generated JSON files which are provided as a shortcut so you can skip straight to using the `example_qat_resnet18.ipynb` notebook.
<br><br>
### Installation
The example uses PyTorch, TorchVision and Nvidia's [PyTorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization) which you can install with:
```
python3 -m pip install -r requirements.txt
```

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

<br>

### Troubleshooting

When exporting PyTorch to ONNX and then using that ONNX file with TensorRT, you might see this error:
```
[07/15/2022-05:26:40] [V] [TRT] Parsing node: Identity_0 [Identity]
[07/15/2022-05:26:40] [V] [TRT] Searching for input: onnx::QuantizeLinear_429
[07/15/2022-05:26:40] [V] [TRT] Identity_0 [Identity] inputs: [onnx::QuantizeLinear_429 -> ()[INT8]],
[07/15/2022-05:26:40] [V] [TRT] Registering layer: onnx::QuantizeLinear_429 for ONNX node: onnx::QuantizeLinear_429
[07/15/2022-05:26:40] [V] [TRT] Registering layer: Identity_0 for ONNX node: Identity_0
[07/15/2022-05:26:40] [E] Error[3]: onnx::QuantizeLinear_429: invalid weights type of Int8
[07/15/2022-05:26:40] [E] [TRT] parsers/onnx/ModelImporter.cpp:791: While parsing node number 0 [Identity -> "onnx::QuantizeLinear_491"]:
[07/15/2022-05:26:40] [E] [TRT] parsers/onnx/ModelImporter.cpp:792: --- Begin node ---
[07/15/2022-05:26:40] [E] [TRT] parsers/onnx/ModelImporter.cpp:793: input: "onnx::QuantizeLinear_429"
output: "onnx::QuantizeLinear_491"
name: "Identity_0"
op_type: "Identity"

[07/15/2022-05:26:40] [E] [TRT] parsers/onnx/ModelImporter.cpp:794: --- End node ---
[07/15/2022-05:26:40] [E] [TRT] parsers/onnx/ModelImporter.cpp:796: ERROR: parsers/onnx/ModelImporter.cpp:179 In function parseGraph:
[6] Invalid Node - Identity_0
onnx::QuantizeLinear_429: invalid weights type of Int8
[07/15/2022-05:26:40] [E] Failed to parse onnx file
[07/15/2022-05:26:40] [I] Finish parsing network model
[07/15/2022-05:26:40] [E] Parsing model failed
[07/15/2022-05:26:40] [E] Failed to create engine from model.
[07/15/2022-05:26:40] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v8205] # trtexec --verbose --nvtxMode=verbose --buildOnly --workspace=1024 --onnx=/tmp/resnet/resnet-qat.onnx --saveEngine=/tmp/resnet/qat/resnet-qat.onnx.engine --timingCacheFile=./timing.cache --int8 --fp16 --shapes=input.1:32x3x224x224

```
The solution is disable constant folding when exporting to ONNX (`do_constant_folding = False`):
```
torch.onnx.export(model, dummy_input, onnx_filename, do_constant_folding=False)
```
<br>

Explanation: When do_constant_folding == True then the ONNX exporter folds `contant(FP32) + QuantizeLinear => constant(INT8)`. TensorRT does not support INT8 weights for explicit quantization.
