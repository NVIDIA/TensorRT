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

from polygraphy import mod, util
from polygraphy.common.interface import TypedDict
from polygraphy.datatype import DataType
from polygraphy.json import Decoder, Encoder, add_json_methods


class BoundedShape(list):
    """
    Represents a shape with min/max bounds.
    """

    def __init__(self, shape, min=None, max=None):
        super().__init__(shape)
        self.min = min
        self.max = max

    def __repr__(self):
        return f"BoundedShape({list(self)}, min={self.min}, max={self.max})"


class MetadataTuple:
    def __init__(self, dtype, shape, docstring):
        self.dtype = dtype
        self.shape = shape
        self.docstring = docstring

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, new):
        self._dtype = DataType.from_dtype(new) if new is not None else None

    def __iter__(self):
        yield from [self.dtype, self.shape]

    def __repr__(self):
        return f"MetadataTuple({self.dtype}, {self.shape}, {self.docstring})"

    def __str__(self):
        ret = ""
        meta_items = []
        if self.dtype is not None:
            meta_items.append(f"dtype={self.dtype}")
        if self.shape is not None:
            meta_items.append(f"shape={tuple(self.shape)}")
        if self.docstring:
            meta_items.append(self.docstring)
        if meta_items:
            ret += "[" + ", ".join(meta_items) + "]"
        return ret

    def __eq__(self, other):
        return self.shape == other.shape and self.dtype == other.dtype


@mod.export()
class TensorMetadata(TypedDict(lambda: str, lambda: MetadataTuple)):
    """
    An OrderedDict[str, MetadataTuple] that maps input names to their data types and shapes.

    Shapes may include negative values, ``None``, or strings to indicate dynamic dimensions.

    Example:
    ::

        shape = tensor_meta["input0"].shape
        dtype = tensor_meta["input0"].dtype
    """

    @staticmethod
    def from_feed_dict(feed_dict):
        """
        Constructs a new TensorMetadata using information from the provided feed_dict.

        Args:
            feed_dict (OrderedDict[str, Union[numpy.ndarray, torch.tensor]]):
                    A mapping of input tensor names to corresponding input arrays.

        Returns:
            TensorMetadata
        """
        meta = TensorMetadata()
        for name, arr in feed_dict.items():
            meta.add(name, util.array.dtype(arr), util.array.shape(arr))
        return meta

    def add(self, name, dtype, shape, min_shape=None, max_shape=None, docstring=None):
        """
        Convenience function for adding entries.

        Args:
            name (str): The name of the input.
            dtype (Any):
                    The data type of the input.
                    This can be any type that can be converted to a Polygraphy DataType.
            shape (Sequence[Union[int, str]]]):
                    The shape of the input. Dynamic dimensions may
                    be indicated by negative values, ``None``, or a string.

            min_shape (Sequence[int]):
                    The minimum valid shape for the input.
                    If provided, this shape should not include any dynamic dimensions.
            max_shape (Sequence[int]):
                    The maximum valid shape for the input.
                    If provided, this shape should not include any dynamic dimensions.
            docstring (str):
                    Any additional information associated with a tensor.

        Returns:
            The newly added entry.
        """
        self[name] = MetadataTuple(
            dtype,
            (
                BoundedShape(shape, min=min_shape, max=max_shape)
                if shape is not None
                else None
            ),
            docstring,
        )
        return self

    def __repr__(self):
        ret = "TensorMetadata()"
        for name, metadata_tuple in self.items():
            (dtype, shape) = metadata_tuple
            ret += util.make_repr(
                ".add",
                name,
                dtype,
                list(shape),
                min_shape=shape.min,
                max_shape=shape.max,
                docstring=metadata_tuple.docstring,
            )[0]
        return ret

    def __str__(self):
        sep = ",\n "
        elems = [f"{name} {meta_tuple}".strip() for name, meta_tuple in self.items()]
        return "{" + sep.join(elems) + "}"


@mod.export()
@add_json_methods("formatted array")
class FormattedArray:
    """
    [EXPERIMENTAL, UNTESTED] This API is experimental and untested and may be significantly
    modified in future releases. Use with caution!

    Representes an array whose semantic shape differs from its physical size in memory.

    For example, consider an ``NCHW`` tensor of shape ``(1, 3, 28, 28)``. If we use a vectorized format
    like ``N(C/4)HW4``, then the physical size of the array would be ``(1, 1, 28, 28 * 4)`` since
    the channel dimension would be padded to a multiple of 4. However, we still need a way to keep
    track of the semantic shape for things like shape inference.

    This class provides a mechanism to specify the shape of an array independently of
    the underlying array.
    """

    def __init__(self, array, shape):
        """
        Args:
            array (Union[np.ndarray, polygraphy.cuda.DeviceView]):
                    The array. In most cases, this will be a raw byte-array.
            shape (Sequence[int]):
                    The semantic shape of the data.
        """
        self.array = array
        self.shape = shape


@Encoder.register(FormattedArray)
def encode(farray):
    return {
        "array": farray.array,
        "shape": farray.shape,
    }


@Decoder.register(FormattedArray)
def decode(dct):
    return FormattedArray(dct["array"], dct["shape"])
