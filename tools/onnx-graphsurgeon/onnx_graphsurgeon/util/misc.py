#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import OrderedDict
from typing import List, Sequence

# default_value exists to solve issues that might result from Python's normal default argument behavior.
# Specifically, consider the following class:
#
# class MyClass(object):
#     def __init__(self, value=[]):
#         self.value = value
#
# This leads to unwanted behavior when the default value is used:
#
# >>> x = MyClass()
# >>> x.value.append("SHOULD NOT BE IN Y")
# >>> y = MyClass()
# >>> y.value
# ['SHOULD NOT BE IN Y']
#
# If we rewrite the class using default value:
#
# class MyClass(object):
#     def __init__(self, value=None):
#         self.value = default_value(value, [])
#
# Then we get the desired behavior:
#
# >>> x = MyClass()
# >>> x.value.append("SHOULD NOT BE IN Y")
# >>> y = MyClass()
# >>> y.value
# []
def default_value(value, default):
    return value if value is not None else default


def combine_dicts(dict0, dict1):
    """
    Combine two dictionaries. Values in the second will overwrite values in the first.
    """
    combined = OrderedDict()
    combined.update(dict0)
    combined.update(dict1)
    return combined


def is_dynamic_dimension(dim):
    return not isinstance(dim, int) or dim < 0


def is_dynamic_shape(shape):
    return any(is_dynamic_dimension(dim) for dim in shape)


def volume(obj):
    vol = 1
    for elem in obj:
        vol *= elem
    return vol


# Special type of list that synchronizes contents with another list.
# Concrete example: Assume some node, n, contains an input tensor, t. If we remove t from n.inputs,
# we also need to remove n from t.outputs. To avoid having to do this manually, we use SynchronizedList,
# which takes an attribute name as a parameter, and then synchronizes to that attribute of each of its elements.
# So, in the example above, we can make n.inputs a synchronized list whose field_name is set to "outputs".
# See test_ir.TestNodeIO for functional tests
class SynchronizedList(list):
    def __init__(self, parent_obj, field_name, initial):
        self.parent_obj = parent_obj
        self.field_name = field_name
        self.extend(initial)

    def _add_to_elem(self, elem):
        # Explicitly avoid SynchronizedList overrides to prevent infinite recursion
        list.append(getattr(elem, self.field_name), self.parent_obj)

    def _remove_from_elem(self, elem):
        # Explicitly avoid SynchronizedList overrides to prevent infinite recursion
        list.remove(getattr(elem, self.field_name), self.parent_obj)

    def __delitem__(self, index):
        self._remove_from_elem(self[index])
        super().__delitem__(index)

    def __setitem__(self, index, elem):
        self._remove_from_elem(self[index])
        super().__setitem__(index, elem)
        self._add_to_elem(elem)

    def append(self, x):
        super().append(x)
        self._add_to_elem(x)

    def extend(self, iterable: Sequence[object]):
        super().extend(iterable)
        for elem in iterable:
            self._add_to_elem(elem)

    def insert(self, i, x):
        super().insert(i, x)
        self._add_to_elem(x)

    def remove(self, x):
        super().remove(x)
        self._remove_from_elem(x)

    def pop(self, i=-1):
        elem = super().pop(i)
        self._remove_from_elem(elem)
        return elem

    def clear(self):
        for elem in self:
            self._remove_from_elem(elem)
        super().clear()

    def __add__(self, other_list: List[object]):
        return list(self) + list(other_list)

    def __iadd__(self, other_list: List[object]):
        self.extend(other_list)
        return self
