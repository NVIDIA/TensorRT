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


"""Quantized convolution
Base code is from nn.Conv, details of Module and original argument can be found there.
Module names are intentionally kept same as unquantized version so that they can be dropped into preexisting model
easily, and load pretrained weight. Aliases with Quant prefix are defined and are encouraged to be used explicitly
when start scratch.
"""

import inspect
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvTransposeNd

from pytorch_quantization import tensor_quant

from . import _utils

__all__ = [
    "Conv2d", "QuantConv2d", "Conv3d", "QuantConv3d", "Conv1d", "QuantConv1d", "ConvTranspose1d", "ConvTranspose2d",
    "ConvTranspose3d", "QuantConvTranspose1d", "QuantConvTranspose2d", "QuantConvTranspose3d"
]


class _QuantConvNd(torch.nn.modules.conv._ConvNd, _utils.QuantMixin):
    """base class of quantized Conv inherited from _ConvNd

    Comments of original arguments can be found in torch.nn.modules.conv

    Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, quant_desc_input, quant_desc_weight):
        super(_QuantConvNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                           transposed, output_padding, groups, bias, padding_mode)
        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def _quant(self, input):
        """Apply quantization on input and weight

        Function called by the classes lower in the hierarchy, which actually performs the quantization before forward
        in the derivate class the particular Function.

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: (quant_in_feature, quant_weight)
        """
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        return (quant_input, quant_weight)


class QuantConv2d(_QuantConvNd):
    """Quantized 2D conv"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode,
                                          quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(quant_input, expanded_padding, mode='circular'),
                              quant_weight, self.bias, self.stride,
                              _pair(0), self.dilation, self.groups)
        else:
            output = F.conv2d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                              self.groups)

        return output


class QuantConv3d(_QuantConvNd):
    """Quantized 3D Conv"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV3D_WEIGHT_PER_CHANNEL

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        super(QuantConv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _triple(0), groups, bias, padding_mode,
                                          quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv3d(F.pad(quant_input, expanded_padding, mode='circular'),
                              quant_weight, self.bias, self.stride, _triple(0),
                              self.dilation, self.groups)
        else:
            output = F.conv3d(quant_input, quant_weight, self.bias, self.stride, self.padding, self.dilation,
                              self.groups)

        return output


class QuantConv1d(_QuantConvNd):
    """Quantized 1D Conv"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONV1D_WEIGHT_PER_CHANNEL

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 **kwargs):

        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        super(QuantConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _single(0), groups, bias, padding_mode,
                                          quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)

    def forward(self, input):
        # the actual quantization happens in the next level of the class hierarchy
        quant_input, quant_weight = self._quant(input)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv1d(F.pad(quant_input, expanded_padding, mode='circular'),
                              quant_weight, self.bias, self.stride,
                              _single(0), self.dilation, self.groups)
        else:
            output = F.conv1d(quant_input, quant_weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        return output


class _QuantConvTransposeNd(torch.nn.modules.conv._ConvTransposeNd, _utils.QuantMixin):
    """base class of quantized Transposed Conv inherited from _ConvTransposeNd

    Comments of original arguments can be found in torch.nn.modules.conv

    Arguments:
        quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of input.
        quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
            Quantization descriptor of weight.

    Raises:
        ValueError: If unsupported arguments are passed in.

    Readonly properties:
        - input_quantizer:
        - weight_quantizer:

    Static methods:
        - set_default_quant_desc_input: Set default_quant_desc_input
        - set_default_quant_desc_weight: Set default_quant_desc_weight
    """

    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, quant_desc_input, quant_desc_weight):
        super(_QuantConvTransposeNd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                    transposed, output_padding, groups, bias, padding_mode)
        self.init_quantizer(quant_desc_input, quant_desc_weight)

    def _quant(self, input):
        """Apply quantization on input and weight

        Function called by the classes lower in the hierarchy, which actually performs the quantization before forward
        in the derivate class the particular Function.

        Arguments:
            input: in_features to quantize
        Returns:
            A tuple: (quant_in_feature, quant_weight)
        """
        quant_input = self._input_quantizer(input)
        quant_weight = self._weight_quantizer(self.weight)

        return (quant_input, quant_weight)

    def _output_padding_nd(self,
                           input,
                           output_size,
                           stride,
                           padding,
                           kernel_size,
                           num_spatial_dims,
                           dilation=None):
        if "num_spatial_dims" in inspect.signature(self._output_padding).parameters:
            return self._output_padding(input, output_size, stride, padding, kernel_size, num_spatial_dims)
        else:
            return self._output_padding(input, output_size, stride, padding, kernel_size)


class QuantConvTranspose1d(_QuantConvTransposeNd):
    """Quantized ConvTranspose1d"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE1D_WEIGHT_PER_CHANNEL

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 **kwargs):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        super(QuantConvTranspose1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode,
            quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for QuantConvTranspose1d')

        num_spatial_dims = 1
        output_padding = self._output_padding_nd(input, output_size, self.stride, self.padding, self.kernel_size,
                                                 num_spatial_dims)

        quant_input, quant_weight = self._quant(input)
        output = F.conv_transpose1d(quant_input, quant_weight, self.bias, self.stride, self.padding, output_padding,
                                    self.groups, self.dilation)
        return output


class QuantConvTranspose2d(_QuantConvTransposeNd):
    """Quantized ConvTranspose2d"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 **kwargs):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        super(QuantConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode,
            quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for QuantConvTranspose2d')

        num_spatial_dims = 2
        output_padding = self._output_padding_nd(input, output_size, self.stride, self.padding, self.kernel_size,
                                                 num_spatial_dims)

        quant_input, quant_weight = self._quant(input)
        output = F.conv_transpose2d(quant_input, quant_weight, self.bias, self.stride, self.padding, output_padding,
                                    self.groups, self.dilation)

        return output


class QuantConvTranspose3d(_QuantConvTransposeNd):
    """Quantized ConvTranspose3d"""

    default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE3D_WEIGHT_PER_CHANNEL

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros',
                 **kwargs):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        quant_desc_input, quant_desc_weight = _utils.pop_quant_desc_in_kwargs(self.__class__, **kwargs)
        super(QuantConvTranspose3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode,
            quant_desc_input=quant_desc_input, quant_desc_weight=quant_desc_weight)

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for QuantConvTranspose3d')

        num_spatial_dims = 3
        output_padding = self._output_padding_nd(input, output_size, self.stride, self.padding, self.kernel_size,
                                                 num_spatial_dims)

        quant_input, quant_weight = self._quant(input)
        output = F.conv_transpose3d(quant_input, quant_weight, self.bias, self.stride, self.padding, output_padding,
                                    self.groups, self.dilation)

        return output


# Define alias with Quant prefix
_ConvNd = _QuantConvNd
Conv1d = QuantConv1d
Conv2d = QuantConv2d
Conv3d = QuantConv3d
ConvTranspose1d = QuantConvTranspose1d
ConvTranspose2d = QuantConvTranspose2d
ConvTranspose3d = QuantConvTranspose3d
