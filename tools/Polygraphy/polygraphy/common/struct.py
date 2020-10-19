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
from collections import OrderedDict, namedtuple

import numpy as np

MetadataTuple = namedtuple("MetadataTuple", ["dtype", "shape"]) # Metadata for single tensor

class TensorMetadata(OrderedDict):
    """
    An OrderedDict[str, Tuple[np.dtype, Tuple[int]]] that maps input names to their data types and shapes.
    """
    def add(self, name, dtype, shape):
        """
        Convenience function for adding entries.

        Args:
            name (str): The name of the input.
            dtype (np.dtype): The data type of the input.
            shape (Tuple[int]):
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
        def str_from_single_meta(name, dtype, shape):
            ret = "{:}".format(name)
            meta_items = []
            if dtype is not None:
                meta_items.append("dtype={:}".format(np.dtype(dtype).name))
            if shape is not None:
                meta_items.append("shape={:}".format(tuple(shape)))
            if meta_items:
                ret += " [" + ", ".join(meta_items) + "]"
            return ret

        sep = ", "
        elems = [str_from_single_meta(name, dtype, shape) for name, (dtype, shape) in self.items()]
        return "{" + sep.join(elems) + "}"
