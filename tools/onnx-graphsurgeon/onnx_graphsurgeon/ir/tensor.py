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

from onnx_graphsurgeon.logger import G_LOGGER
from onnx_graphsurgeon.util import misc

from typing import Set, Sequence, Union
import numpy as np


class Tensor(object):
    """Abstract base class for tensors in a graph"""

    DYNAMIC = -1

    def __init__(self):
        """
        **This class is abstract and cannot be constructed directly.**
        """
        raise NotImplementedError("Tensor is an abstract class")

    def __setattr__(self, name, value):
        if name in ["inputs", "outputs"]:
            try:
                attr = getattr(self, name)
                if value is attr:
                    # This can happen when using things like +=
                    # The __iadd__ is executed followed by an assignment
                    return

                attr.clear()
                attr.extend(value)
            except AttributeError:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)

    def is_empty(self):
        """
        Returns whether this tensor is considered empty in the graph.

        *Note: 'Empty' here refers to the name of the tensor, which is omitted for
        optional tensors, NOT the shape of the tensor*

        Returns:
            bool: Whether the tensor is empty, meaning that it is used for an omitted optional input or output.
        """
        return self.name == ""

    def to_constant(
        self,
        values: np.ndarray,
        data_location: int = None,
        export_dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None,
    ):
        """
        Modifies this tensor in-place to convert it to a Constant. This means that all consumers/producers of the tensor will see the update.

        Args:
            values (np.ndarray): The values in this tensor

            data_location (int):
                    An enum value indicating the location where the tensor data is stored.
                    Generally, this will come from onnx.TensorProto.DataLocation.

            dtype (Union[numpy.dtype, onnx.TensorProto.DataType]): The data type of the tensor.
        Returns:
            self
        """
        self.__class__ = Constant
        self._values = values
        self.data_location = data_location
        self.export_dtype = export_dtype

        return self

    def to_variable(
        self,
        dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None,
        shape: Sequence[Union[int, str]] = [],
    ):
        """
        Modifies this tensor in-place to convert it to a Variable. This means that all consumers/producers of the tensor will see the update.

        Args:
            dtype (Union[numpy.dtype, onnx.TensorProto.DataType]): The data type of the tensor.
            shape (Sequence[int]): The shape of the tensor.

        Returns:
            self
        """

        variable_dtype = dtype if dtype is not None else self.export_dtype

        self.__class__ = Variable
        self.shape = shape
        self.dtype = variable_dtype

        return self

    def i(self, tensor_idx=0, producer_idx=0):
        """
        Convenience function to get an input tensor of one of this tensor's input nodes.
        Note that the parameters are swapped compared to the o() function; this is because tensors are likely to have only a single producer

        For example:
        ::

            assert tensor.i() == tensor.inputs[0].inputs[0]
            assert tensor.i(1, 2) == tensor.inputs[2].inputs[1]

        Args:
            tensor_idx (int): The index of the input tensor of the input node. Defaults to 0.
            producer_idx (int): The index of the producer node of the input tensor, if the tensor has multiple producers. Defaults to 0.

        Returns:
            Tensor: The specified producer (input) tensor.
        """
        return self.inputs[producer_idx].inputs[tensor_idx]

    def o(self, consumer_idx=0, tensor_idx=0):
        """
        Convenience function to get an output tensor of one of this tensor's output nodes.

        For example:
        ::

            assert tensor.o() == tensor.outputs[0].outputs[0]
            assert tensor.o(2, 1) == tensor.outputs[2].outputs[1]

        Args:
            consumer_idx (int): The index of the consumer of the input tensor. Defaults to 0.
            tensor_idx (int): The index of the output tensor of the node, if the node has multiple outputs. Defaults to 0.

        Returns:
            Tensor: The specified consumer (output) tensor
        """
        return self.outputs[consumer_idx].outputs[tensor_idx]

    def __str__(self):
        return "{:} ({:}): (shape={:}, dtype={:})".format(
            type(self).__name__, self.name, self.shape, self.dtype
        )

    def __repr__(self):  # Hack to make logging output pretty.
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
        """
        Creates a variable tensor with no name.
        This can be used to represent an omitted optional input of a node.
        """
        return Variable(name="")

    def __init__(
        self,
        name: str,
        dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None,
        shape: Sequence[Union[int, str]] = None,
        type: str = "tensor_type",
    ):
        """
        Represents a Tensor whose value is not known until inference-time.

        Args:
            name (str): The name of the tensor.
            dtype (Union[numpy.dtype, onnx.TensorProto.DataType]): The data type of the tensor.
            shape (Sequence[Union[int, str]]): The shape of the tensor. This may contain strings if the model uses dimension parameters.
            type (str): The type of the tensor.
        """
        self.name = name
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=[])
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=[])
        self.dtype = dtype
        self.shape = misc.default_value(shape, None)
        self.type = type

    def to_constant(
        self,
        values: np.ndarray,
        export_dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None,
    ):
        del self.dtype
        del self.shape

        return super().to_constant(values, export_dtype=export_dtype)

    def copy(self):
        """
        Makes a shallow copy of this tensor, omitting input and output information.

        Note: Generally, you should only ever make a copy of a Graph.
        """
        return Variable(self.name, self.dtype, self.shape)


class LazyValues(object):
    """
    A special object that represents constant tensor values that should be lazily loaded.
    """

    def __init__(self, tensor):
        """
        Args:
            tensor (onnx.TensorProto, onnx.SparseTensorProto): The ONNX tensor that this instance should lazily load.
        """
        from onnx_graphsurgeon.importers.onnx_importer import (
            get_onnx_tensor_shape,
            get_onnx_tensor_dtype,
            get_itemsize,
        )

        self.tensor = tensor
        self.shape = get_onnx_tensor_shape(self.tensor)
        self.dtype = get_onnx_tensor_dtype(self.tensor)
        self.nbytes = misc.volume(self.shape) * get_itemsize(self.dtype)

    def load(self):
        """
        Load a numpy array from the underlying tensor values.

        Returns:
            np.array: A numpy array containing the values of the tensor.
        """
        import onnx
        import onnx.numpy_helper
        from onnx_graphsurgeon.importers.onnx_importer import (
            get_dtype_name,
            get_numpy_type,
        )

        if get_numpy_type(self.dtype) is None:
            G_LOGGER.warning(
                f"Datatype: {get_dtype_name(self.dtype)} could not be converted to a NumPy type.\n"
                f"Accessing the values of this constant tensor ({self.tensor.name}) will cause them to be casted to a supported data type. "
                f"This means that the weights will have a different type than the original model when they are exported again!\n"
                f"If this is not what you intended, please avoid accessing the values of this constant tensor."
            )

        return np.array(onnx.numpy_helper.to_array(self.tensor))

    def __str__(self):
        return "LazyValues (shape={:}, dtype={:})".format(self.shape, self.dtype)

    def __repr__(self):  # Hack to make logging output pretty.
        return self.__str__()


class SparseValues(LazyValues):
    """
    A special object that represents constant tensor values that is sparse
    """

    def load(self):
        """
        Load a numpy array from the sparse structure.

        Returns:
            np.array: A numpy array containing the values of the tensor.
        """
        import onnx
        import onnx.numpy_helper
        from onnx_graphsurgeon.importers.onnx_importer import (
            get_dtype_name,
            get_numpy_type,
        )

        supported_index_type = [onnx.TensorProto.INT64]
        if self.tensor.indices.data_type not in supported_index_type:
            G_LOGGER.critical(
                f"Unsupported index data type {self.tensor.indices.data_type} in {self.tensor.values.name}"
            )

        if self.tensor.values.data_type == onnx.TensorProto.FLOAT16:
            values_data = np.asarray(
                self.tensor.values.int32_data, dtype=np.uint16
            ).view(np.float16)
        else:
            field_name = onnx.helper.tensor_dtype_to_field(self.tensor.values.data_type)
            values = getattr(self.tensor.values, field_name)
            dtype = onnx.helper.tensor_dtype_to_np_dtype(self.tensor.values.data_type)
            values_data = np.asarray(values, dtype)
        indices_data = self.tensor.indices.int64_data

        if len(self.tensor.indices.dims) == 1:
            values = np.zeros(np.prod(self.tensor.dims))
            # [NNZ] layout, in which case the i-th value must be the linearized-index of the i-th value.
            values[indices_data] = values_data
            values = values.reshape(self.tensor.dims)
        elif len(self.tensor.indices.dims) == 2:
            # [NNZ, rank] with the [i,j]-th value corresponding to the j-th index of the i-th value
            values = np.zeros(self.tensor.dims)
            indices_data = np.asarray(indices_data).reshape(self.tensor.indices.dims)

            for i in range(len(values_data)):
                values[tuple(indices_data[i])] = values_data[i]
        else:
            G_LOGGER.critical(
                f"Unsupported index data dims {self.tensor.indices.dims} in {self.tensor.values.name}"
            )

        return values

    def __str__(self):
        return "SparseValues (shape={:}, dtype={:})".format(self.shape, self.dtype)


class Constant(Tensor):
    def __init__(
        self,
        name: str,
        values: Union[np.ndarray, LazyValues],
        data_location: int = None,
        export_dtype: Union[np.dtype, "onnx.TensorProto.DataType"] = None,
    ):
        """
        Represents a Tensor whose value is known.

        Args:
            name (str): The name of the tensor.
            values (numpy.ndarray): The values in this tensor, in the form of a NumPy array.

            data_location (int):
                    An enum value indicating the location where the tensor data is stored.
                    Generally, this will come from onnx.TensorProto.DataLocation.


            export_dtype (Union[np.dtype, onnx.TensorProto.DataType]):
                    The data type of the tensor when exported to onnx. If not specified, then
                    the data type of values will be used.
        """
        self.name = name
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=[])
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=[])
        if (
            not isinstance(values, np.ndarray)
            and not isinstance(values, LazyValues)
            and not isinstance(values, SparseValues)
        ):
            G_LOGGER.critical(
                "Provided `values` argument is not a NumPy array, a LazyValues instance or a"
                "SparseValues instance. Please provide a NumPy array or LazyValues instance "
                "to construct a Constant. Note: Provided `values` parameter was: {:}".format(
                    values
                )
            )
        self._values = values
        self.data_location = data_location
        self._export_dtype = export_dtype

    def to_variable(
        self, dtype: np.dtype = None, shape: Sequence[Union[int, str]] = []
    ):
        var_dtype = self.export_dtype

        del self._export_dtype
        del self._values

        if dtype is not None:
            return super().to_variable(dtype, shape)

        return super().to_variable(var_dtype, shape)

    def copy(self):
        """
        Makes a shallow copy of this tensor, omitting input and output information.

        Note: Generally, you should only ever make a copy of a Graph.
        """
        return Constant(self.name, self._values, export_dtype=self.export_dtype)

    @property
    def values(self):
        # Load values when they are first accesed
        if isinstance(self._values, LazyValues):
            self._values = self._values.load()
        return self._values

    @values.setter
    def values(self, values: Union[np.ndarray, LazyValues]):
        self._values = values

    @property
    def shape(self):
        return self._values.shape

    @property
    def dtype(self):
        return self._values.dtype

    @property
    def export_dtype(self):
        if self._export_dtype is not None:
            return self._export_dtype

        return self.dtype

    @export_dtype.setter
    def export_dtype(self, export_dtype):
        self._export_dtype = export_dtype

    def __repr__(self):  # Hack to make logging output pretty.
        ret = self.__str__()
        ret += "\n{:}".format(self._values)
        return ret
