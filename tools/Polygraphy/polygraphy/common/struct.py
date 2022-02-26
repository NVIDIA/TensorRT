#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from polygraphy import mod
from polygraphy.common.interface import TypedDict

np = mod.lazy_import("numpy")


class MetadataTuple(object):
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape

    def __iter__(self):
        yield from [self.dtype, self.shape]

    def __repr__(self):
        return "MetadataTuple({:}, {:})".format(self.dtype, self.shape)

    def __str__(self):
        ret = ""
        meta_items = []
        if self.dtype is not None:
            meta_items.append("dtype={:}".format(np.dtype(self.dtype).name))
        if self.shape is not None:
            meta_items.append("shape={:}".format(tuple(self.shape)))
        if meta_items:
            ret += "[" + ", ".join(meta_items) + "]"
        return ret


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
            feed_dict (OrderedDict[str, numpy.ndarray]):
                    A mapping of input tensor names to corresponding input NumPy arrays.

        Returns:
            TensorMetadata
        """
        meta = TensorMetadata()
        for name, arr in feed_dict.items():
            meta.add(name, arr.dtype, arr.shape)
        return meta

    def add(self, name, dtype, shape):
        """
        Convenience function for adding entries.

        Args:
            name (str): The name of the input.
            dtype (numpy.dtype): The data type of the input.
            shape (Sequence[Union[int, str]]]):
                    The shape of the input. Dynamic dimensions may
                    be indicated by negative values, ``None``, or a string.

        Returns:
            The newly added entry.
        """
        self[name] = MetadataTuple(dtype, shape)
        return self

    def __repr__(self):
        ret = "TensorMetadata()"
        for name, (dtype, shape) in self.items():
            ret += ".add('{:}', {:}, {:})".format(name, dtype, shape)
        return ret

    def __str__(self):
        sep = ",\n "
        elems = ["{:} {:}".format(name, meta_tuple).strip() for name, meta_tuple in self.items()]
        return "{" + sep.join(elems) + "}"
