#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
This script demonstrates how to use PyTorch tensors with the TensorRT runner and calibrator.
"""

import torch

from polygraphy.backend.trt import (
    Calibrator,
    CreateConfig,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
)

# If your PyTorch installation has GPU support, then we'll allocate the tensors
# directly in GPU memory. This will mean that the calibrator and runner can skip the
# host-to-device copy we would otherwise incur with NumPy arrays.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calib_data():
    for _ in range(4):
        yield {"x": torch.ones((1, 1, 2, 2), dtype=torch.float32, device=DEVICE)}


def main():
    calibrator = Calibrator(data_loader=calib_data())

    engine = engine_from_network(
        network_from_onnx_path("identity.onnx"),
        config=CreateConfig(int8=True, calibrator=calibrator),
    )

    with TrtRunner(engine) as runner:
        inp_data = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=DEVICE)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        #
        # When you provide PyTorch tensors in the feed_dict, the runner will try to use
        # PyTorch tensors for the outputs. Specifically:
        # - If the `copy_outputs_to_host` argument to `infer()` is set to `True` (the default),
        #       it will return PyTorch tensors in CPU memory.
        # - If `copy_outputs_to_host` is `False`, it will return:
        #       - PyTorch tensors in GPU memory if you have a GPU-enabled PyTorch installation.
        #       - Polygraphy `DeviceView`s otherwise.
        #
        outputs = runner.infer({"x": inp_data})

        # `copy_outputs_to_host` defaults to True, so the outputs should be PyTorch
        # tensors in CPU memory.
        assert isinstance(outputs["y"], torch.Tensor)
        assert outputs["y"].device.type == "cpu"

        assert torch.equal(outputs["y"], inp_data.to("cpu"))  # It's an identity model!


if __name__ == "__main__":
    main()
