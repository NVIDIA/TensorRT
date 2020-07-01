#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.util import misc

from typing import Set, Sequence, Union
import numpy as np


class Tensor(object):
    DYNAMIC = -1

    def __init__(self):
        raise NotImplementedError("Tensor is an abstract class")


    def __setattr__(self, name, value):
        if name in ["inputs", "outputs"]:
            try:
                getattr(self, name).clear()
                getattr(self, name).extend(value)
            except AttributeError:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


    def is_empty(self):
        """
        Returns whether this tensor is empty.

        Returns:
            bool: Whether the tensor is empty.
        """
        return self.name == ""


    def to_constant(self, values: np.ndarray):
        """
        Modifies this tensor in-place to convert it to a Constant. This means that all consumers/producers of the tensor will see the update.

        Args:
            values (np.ndarray): The values in this tensor

        Returns:
            self
        """
        self.__class__ = Constant
        self.values = values
        return self


    def to_variable(self, dtype: np.dtype=None, shape: Sequence[Union[int, str]]=[]):
        """
        Modifies this tensor in-place to convert it to a Variable. This means that all consumers/producers of the tensor will see the update.

        Optional Args:
            dtype (np.dtype): The data type of the tensor.
            shape (Sequence[int]): The shape of the tensor.

        Returns:
            self
        """
        self.__class__ = Variable
        self.dtype = dtype
        self.shape = shape
        return self


    def i(self, tensor_idx=0, producer_idx=0):
        """
        Convenience function to get an input tensor of one of this tensor's input nodes.
        Note that the parameters are swapped compared to the o() function; this is because tensors are likely to have only a single producer

        Args:
            tensor_idx (int): The index of the input tensor of the input node. Defaults to 0.
            producer_idx (int): The index of the producer node of the input tensor, if the tensor has multiple producers. Defaults to 0.

        Example:
            assert tensor.i() == tensor.inputs[0].inputs[0]
            assert tensor.i(1, 2) == tensor.inputs[2].inputs[1]

        Returns:
            Node: The specified producer (input) tensor.
        """
        return self.inputs[producer_idx].inputs[tensor_idx]


    def o(self, consumer_idx=0, tensor_idx=0):
        """
        Convenience function to get an output tensor of one of this tensor's output nodes.

        Args:
            consumer_idx (int): The index of the consumer of the input tensor. Defaults to 0.
            tensor_idx (int): The index of the output tensor of the node, if the node has multiple outputs. Defaults to 0.

        Example:
            assert tensor.o() == tensor.outputs[0].outputs[0]
            assert tensor.o(2, 1) == tensor.outputs[2].outputs[1]

        Returns:
            Node: The specified consumer (output) tensor
        """
        return self.outputs[consumer_idx].outputs[tensor_idx]


    def __str__(self):
        return "{:} ({:}): (shape={:}, dtype={:})".format(type(self).__name__, self.name, self.shape, self.dtype)


    def __repr__(self):
        return self.__str__()


    def __eq__(self, other):
        """
        Perform a check to see if two tensors are equal.

        Tensors are considered equal if they share the same name. A Graph must not include Tensors with duplicate names.
        """
        return self.name == other.name


class Variable(Tensor):
    @staticmethod
    def empty():
        return Variable(name="")


    def __init__(self, name: str, dtype: np.dtype=None, shape: Sequence[Union[int, str]]=None):
        """
        Represents a Tensor whose value is not known until inference-time.

        Args:
            name (str): The name of the tensor.
            dtype (np.dtype): The data type of the tensor.
            shape (Sequence[Union[int, str]]): The shape of the tensor. This may contain strings if the model uses dimension parameters.
        """
        self.name = name
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=[])
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=[])
        self.dtype = dtype
        self.shape = misc.default_value(shape, None)


    def to_constant(self, values: np.ndarray):
        del self.dtype
        del self.shape
        return super().to_constant(values)


    def copy(self):
        """
        Makes a shallow copy of this tensor, omitting input and output information.

        Note: Generally, you should only ever make a deep copy of a Graph.
        """
        return Variable(self.name, self.dtype, self.shape)


class Constant(Tensor):
    def __init__(self, name: str, values: np.ndarray):
        """
        Represents a Tensor whose value is known.

        Args:
            name (str): The name of the tensor.
            values (np.ndarray): The values in this tensor, in the form of a NumPy array.
            dtype (np.dtype): The data type of the tensor.
            shape (Sequence[Union[int, str]]): The shape of the tensor.
        """
        self.name = name
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=[])
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=[])
        if not isinstance(values, np.ndarray):
            G_LOGGER.critical("Provided `values` argument is not a NumPy array (please provide a NumPy array to construct a Constant): {:}".format(values))
        self.values = np.array(values)


    def to_variable(self, dtype: np.dtype=None, shape: Sequence[Union[int, str]]=[]):
        del self.values
        return super().to_variable(dtype, shape)


    def copy(self):
        """
        Makes a shallow copy of this tensor, omitting input and output information.

        Note: Generally, you should only ever make a deep copy of a Graph.
        """
        return Constant(self.name, self.values)


    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype.type

    def __repr__(self):
        ret = self.__str__()
        ret += "\n{:}".format(self.values)
        return ret
