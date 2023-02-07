/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <torch/extension.h>

void fake_tensor_quant_cuda_inplace(at::Tensor, at::Tensor, int, bool);
at::Tensor fake_tensor_quant_cuda(at::Tensor, at::Tensor, int, bool);
at::Tensor fake_tensor_quant_with_axis_cuda(at::Tensor, at::Tensor, int, int, bool);
float bits_to_bound(int, int);

void fake_tensor_quant_inplace(at::Tensor inputs, at::Tensor amax, int num_bits=8, bool is_unsigned=false) {
  TORCH_CHECK(amax.numel(), 1);

  float bound = bits_to_bound(num_bits, is_unsigned);
  float scale = bound / amax.data_ptr<float>()[0];
  for (int i = 0; i < inputs.numel(); ++i) {
    float output = round(inputs.data_ptr<float>()[i] * scale);
    output = output > bound ? bound : output;
    output = output < -bound ? -bound : output;

    inputs.data_ptr<float>()[i] = output / scale;
  }
}

void fake_tensor_quant_(at::Tensor inputs, at::Tensor amax, int num_bits=8, bool is_unsigned=false) {
  TORCH_CHECK(amax.numel(), 1);
  if (inputs.type().is_cuda()) {
    fake_tensor_quant_cuda_inplace(inputs, amax, num_bits, is_unsigned);
  } else {
    fake_tensor_quant_inplace(inputs, amax, num_bits, is_unsigned);
  }
}

at::Tensor fake_tensor_quant(at::Tensor inputs, at::Tensor amax, int num_bits=8, bool is_unsigned=false) {
  TORCH_CHECK(amax.numel(), 1);
  if (inputs.type().is_cuda()) {
    return fake_tensor_quant_cuda(inputs, amax, num_bits, is_unsigned);
  } else {
    auto outputs = torch::clone(inputs);
    fake_tensor_quant_inplace(outputs, amax, num_bits, is_unsigned);
    return outputs;
  }
}

at::Tensor fake_tensor_quant_with_axis(
    at::Tensor inputs, at::Tensor amax, int axis, int num_bits=8, bool is_unsigned=false) {
  TORCH_CHECK(amax.numel(), inputs.size(axis));
  if (inputs.type().is_cuda()) {
    return fake_tensor_quant_with_axis_cuda(
        inputs, amax, axis, num_bits, is_unsigned);
  } else {
    throw std::runtime_error("axis is only supported on GPU.");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_tensor_quant_", &fake_tensor_quant_, "Fake Tensor Quant Inplace", py::arg("inputs"), py::arg("amax"),
        py::arg("num_bits")=8, py::arg("unsigned")=false);
  m.def("fake_tensor_quant", &fake_tensor_quant, "Fake Tensor Quant", py::arg("inputs"), py::arg("amax"),
        py::arg("num_bits")=8, py::arg("unsigned")=false);
  m.def("fake_tensor_quant_with_axis", &fake_tensor_quant_with_axis,
        "Fake Tensor Quant with axis", py::arg("inputs"), py::arg("amax"),
        py::arg("axis"), py::arg("num_bits")=8, py::arg("unsigned")=false);
}
