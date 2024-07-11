#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for ONNX export."""
import io
import onnxruntime
import pytest
import torch

# ORT output correctness tests sometimes fails due to random seed.
# It needs to be investigated closer
torch.manual_seed(0)

import tests.utils as test_utils
import torch.nn as nn
import pytorch_quantization
from pytorch_quantization.nn import QuantLinear
from pytorch_quantization.tensor_quant import QuantDescriptor


class MyModel(nn.Module):
    """Test model for ONNX export."""

    def __init__(self, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            QuantLinear(16, 32, **kwargs),
            nn.ReLU(),
            QuantLinear(32, 64, **kwargs),
            nn.ReLU(),
            QuantLinear(64, 16, **kwargs),
        )

    def forward(self, x):
        return self.net(x)


@pytest.mark.parametrize("num_bits, per_channel_quantization, constant_folding, dtype",
                         [(8, True, True, torch.float32), (8, False, True, torch.float32),
                          (8, True, False, torch.float32), (8, False, False, torch.float32),
                          (8, False, False, torch.float16), (8, False, False, torch.bfloat16),
                          ((4, 3), False, True, torch.float32), ((4, 3), False, False, torch.float32),
                          ((4, 3), False, False, torch.float16), ((4, 3), False, False, torch.bfloat16)])
def test_onnx_export(num_bits, per_channel_quantization, constant_folding, dtype, onnx_file_path=None):
    quant_desc_input = QuantDescriptor(num_bits=num_bits, axis=None)
    quant_desc_weight = QuantDescriptor(num_bits=num_bits, axis=0 if per_channel_quantization else None)

    model = MyModel(quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight).cuda()
    model.eval()

    OPSET = 17
    dummy_input = torch.randn(16, 16).cuda()
    input_names = ["input"]
    output_names = ["output"]

    model = model.to(dtype)
    dummy_input = dummy_input.to(dtype)

    # Calibrate model
    for name, module in model.named_modules():
        if name.endswith('_quantizer'):
            module.enable_calib()
            module.disable_quant()
    _ = model(dummy_input)
    for name, module in model.named_modules():
        if name.endswith('_quantizer'):
            module.disable_calib()
            module.load_calib_amax()
            module.enable_quant()

    f = io.BytesIO() if onnx_file_path is None else None

    with pytorch_quantization.enable_onnx_export():
        torch.onnx.export(
            model,
            dummy_input,
            f=f if onnx_file_path is None else onnx_file_path,
            opset_version=OPSET,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=constant_folding,
        )

    # TODO: ort output correctness check for fp8
    # ONNXRuntime does not seem to be supporting bf16 gemms
    if num_bits == 8 and dtype != torch.bfloat16:
        if f is not None:
            f.seek(0)
        ort_session = onnxruntime.InferenceSession(f.read() if onnx_file_path is None else onnx_file_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        ort_result = ort_session.run([], {"input": dummy_input.cpu().numpy()})
        ort_result = torch.tensor(ort_result[0]).cuda()
        torch_result = model(dummy_input)
        test_utils.compare(ort_result, torch_result, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_onnx_export(8, False, False, torch.float16, "/tmp/test_fp16.onnx")
    test_onnx_export(8, False, False, torch.bfloat16, "/tmp/test_bf16.onnx")
