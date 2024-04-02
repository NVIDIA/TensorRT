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


"""Quantized Pooling
Base code is from nn.pooling, details of Module and original argument can be found there.
Module names are intentionally kept same as unquantized version so that they can be dropped into preexisting model
easily, and load pretrained weight. Aliases with Quant prefix are defined and are encouraged to be used explicitly
when start scratch.
"""

from torch.nn.modules import pooling

from . import _utils

__all__ = [
    "MaxPool1d", "QuantMaxPool1d", "MaxPool2d", "QuantMaxPool2d", "MaxPool3d", "QuantMaxPool3d",
    "AvgPool1d", "QuantAvgPool1d", "AvgPool2d", "QuantAvgPool2d", "AvgPool3d", "QuantAvgPool3d",
    "AdaptiveAvgPool1d", "QuantAdaptiveAvgPool1d", "AdaptiveAvgPool2d", "QuantAdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "QuantAdaptiveAvgPool3d"
]

class QuantMaxPool1d(pooling.MaxPool1d, _utils.QuantInputMixin):
    """Quantized 1D maxpool"""
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, **kwargs):
        super(QuantMaxPool1d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantMaxPool1d, self).forward(quant_input)

class QuantMaxPool2d(pooling.MaxPool2d, _utils.QuantInputMixin):
    """Quantized 2D maxpool"""
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, **kwargs):
        super(QuantMaxPool2d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantMaxPool2d, self).forward(quant_input)

class QuantMaxPool3d(pooling.MaxPool3d, _utils.QuantInputMixin):
    """Quantized 3D maxpool"""
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, **kwargs):
        super(QuantMaxPool3d, self).__init__(kernel_size, stride, padding, dilation,
                                             return_indices, ceil_mode)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantMaxPool3d, self).forward(quant_input)


class QuantAvgPool1d(pooling.AvgPool1d, _utils.QuantInputMixin):
    """Quantized 1D average pool"""
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, **kwargs):
        super(QuantAvgPool1d, self).__init__(kernel_size, stride, padding, ceil_mode,
                                             count_include_pad)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantAvgPool1d, self).forward(quant_input)

class QuantAvgPool2d(pooling.AvgPool2d, _utils.QuantInputMixin):
    """Quantized 2D average pool"""
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None, **kwargs):
        super(QuantAvgPool2d, self).__init__(kernel_size, stride, padding, ceil_mode,
                                             count_include_pad, divisor_override)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantAvgPool2d, self).forward(quant_input)


class QuantAvgPool3d(pooling.AvgPool3d, _utils.QuantInputMixin):
    """Quantized 3D average pool"""
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None, **kwargs):
        super(QuantAvgPool3d, self).__init__(kernel_size, stride, padding, ceil_mode,
                                             count_include_pad, divisor_override)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantAvgPool3d, self).forward(quant_input)


class QuantAdaptiveAvgPool1d(pooling.AdaptiveAvgPool1d, _utils.QuantInputMixin):
    """Quantized 1D adaptive average pool"""
    def __init__(self, output_size, **kwargs):
        super(QuantAdaptiveAvgPool1d, self).__init__(output_size)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantAdaptiveAvgPool1d, self).forward(quant_input)


class QuantAdaptiveAvgPool2d(pooling.AdaptiveAvgPool2d, _utils.QuantInputMixin):
    """Quantized 2D adaptive average pool"""
    def __init__(self, output_size, **kwargs):
        super(QuantAdaptiveAvgPool2d, self).__init__(output_size)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantAdaptiveAvgPool2d, self).forward(quant_input)


class QuantAdaptiveAvgPool3d(pooling.AdaptiveAvgPool3d, _utils.QuantInputMixin):
    """Quantized 3D adaptive average pool"""
    def __init__(self, output_size, **kwargs):
        super(QuantAdaptiveAvgPool3d, self).__init__(output_size)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantAdaptiveAvgPool3d, self).forward(quant_input)


# Define alias with Quant prefix
MaxPool1d = QuantMaxPool1d
MaxPool2d = QuantMaxPool2d
MaxPool3d = QuantMaxPool3d
AvgPool1d = QuantAvgPool1d
AvgPool2d = QuantAvgPool2d
AvgPool3d = QuantAvgPool3d
AdaptiveAvgPool1d = QuantAdaptiveAvgPool1d
AdaptiveAvgPool2d = QuantAdaptiveAvgPool2d
AdaptiveAvgPool3d = QuantAdaptiveAvgPool3d
