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


"""Some helper functions for implementing quantized modules"""
import copy
import inspect

from absl import logging

from torch import nn

from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor, QUANT_DESC_8BIT_PER_TENSOR


class QuantMixin():
    """Mixin class for adding basic quantization logic to quantized modules"""

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    default_quant_desc_weight = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    @classmethod
    def set_default_quant_desc_weight(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_weight = copy.deepcopy(value)

    def init_quantizer(self, quant_desc_input, quant_desc_weight, num_layers=None):
        """Helper function for __init__ of quantized module

        Create input and weight quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
            num_layers: An integer. Default None. If not None, create a list of quantizers.
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        self._fake_quant = True
        if (not quant_desc_input.fake_quant) or (not quant_desc_weight.fake_quant):
            raise ValueError("Only fake quantization is supported!")

        logging.info("Input is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_input.fake_quant else "fake ",
                     quant_desc_input.num_bits, self.__class__.__name__, quant_desc_input.axis)
        logging.info("Weight is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_weight.fake_quant else "fake ",
                     quant_desc_weight.num_bits, self.__class__.__name__, quant_desc_weight.axis)

        if num_layers is None:
            self._input_quantizer = TensorQuantizer(quant_desc_input)
            self._weight_quantizer = TensorQuantizer(quant_desc_weight)
        else:
            self._input_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_input) for _ in range(num_layers)])
            self._weight_quantizers = nn.ModuleList([TensorQuantizer(quant_desc_weight) for _ in range(num_layers)])

    # pylint:disable=missing-docstring
    @property
    def input_quantizer(self):
        return self._input_quantizer

    @property
    def weight_quantizer(self):
        return self._weight_quantizer
    # pylint:enable=missing-docstring


class QuantInputMixin():
    """Mixin class for adding basic quantization logic to quantized modules"""

    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR

    @classmethod
    def set_default_quant_desc_input(cls, value):
        """
        Args:
            value: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not isinstance(value, QuantDescriptor):
            raise ValueError("{} is not an instance of QuantDescriptor!")
        cls.default_quant_desc_input = copy.deepcopy(value)

    def init_quantizer(self, quant_desc_input):
        """Helper function for __init__ of simple quantized module

        Create input quantizer based on quant_desc passed by kwargs, or default of the class.

        Args:
            quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`
        """
        if not inspect.stack()[1].function == "__init__":
            raise TypeError("{} should be only called by __init__ of quantized module.".format(__name__))
        self._fake_quant = True
        if not quant_desc_input.fake_quant:
            raise ValueError("Only fake quantization is supported!")

        logging.info("Input is %squantized to %d bits in %s with axis %s!", ""
                     if not quant_desc_input.fake_quant else "fake ",
                     quant_desc_input.num_bits, self.__class__.__name__, quant_desc_input.axis)

        self._input_quantizer = TensorQuantizer(quant_desc_input)

    # pylint:disable=missing-docstring
    @property
    def input_quantizer(self):
        return self._input_quantizer
    # pylint:enable=missing-docstring


def pop_quant_desc_in_kwargs(quant_cls, input_only=False, **kwargs):
    """Pop quant descriptors in kwargs

    If there is no descriptor in kwargs, the default one in quant_cls will be used

    Arguments:
       quant_cls: A class that has default quantization descriptors
       input_only: A boolean. If True, pop quant_desc_input only, not quant_desc_weight. Default false.

    Keyword Arguments:
       quant_desc_input: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of input.
       quant_desc_weight: An instance of :class:`QuantDescriptor <pytorch_quantization.tensor_quant.QuantDescriptor>`.
           Quantization descriptor of weight.
    """
    quant_desc_input = kwargs.pop('quant_desc_input', quant_cls.default_quant_desc_input)
    if not input_only:
        quant_desc_weight = kwargs.pop('quant_desc_weight', quant_cls.default_quant_desc_weight)

    # Check if anything is left in **kwargs
    if kwargs:
        raise TypeError("Unused keys: {}".format(kwargs.keys()))

    if input_only:
        return quant_desc_input
    return quant_desc_input, quant_desc_weight
