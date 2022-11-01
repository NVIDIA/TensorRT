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

import copy
from collections import OrderedDict

from polygraphy import util
from polygraphy.logger import G_LOGGER

#
# NOTE: These classes intentionally don't inherit from the built-in collections (dict, list, etc.)
# because doing so prevents us from providing custom JSON serialization methods, since the default
# encoder implementation can handle most of the built-in collections and therefore doesn't dispatch
# to custom implementations.
#


def TypedDict(key_type_func, value_type_func):
    """
    Returns a class (not an instance) that will provide a dictionary-like
    interface with runtime type checks.

    Note: The types are provided lazily via a callable to avoid unnecessary dependencies
    on types from external packages at import-time.

    Args:
        key_type_func (Callable() -> type):
                A callable that returns the expected key type.
        value_type_func (Callable() -> type):
                A callable that returns the expected value type.
    """

    class Interface:
        def __init__(self, dct=None):
            self.dct = OrderedDict(util.default(dct, {}))
            self.key_type = key_type_func()
            self.value_type = value_type_func()

        def _check_types(self, key, val):
            if not isinstance(key, self.key_type):
                G_LOGGER.critical(
                    f"Unsupported key type in {self}. Key: {repr(key)} is type `{type(key).__name__}` but {type(self).__name__} expects type `{self.key_type.__name__}`"
                )
            if not isinstance(val, self.value_type):
                G_LOGGER.critical(
                    f"Unsupported value type in {self}. Value: {repr(val)} for key: {repr(key)} is type `{type(val).__name__}` but {type(self).__name__} expects type `{self.value_type.__name__}`"
                )

        def keys(self):
            return self.dct.keys()

        def values(self):
            return self.dct.values()

        def items(self):
            return self.dct.items()

        def update(self, other):
            for key, val in other.items():
                self._check_types(key, val)
            return self.dct.update(other)

        def __contains__(self, key):
            return key in self.dct

        def __getitem__(self, key):
            return self.dct[key]

        def __setitem__(self, key, val):
            self._check_types(key, val)
            self.dct[key] = val

        def __str__(self):
            return str(self.dct)

        def __repr__(self):
            return repr(self.dct)

        def __len__(self):
            return len(self.dct)

        def __eq__(self, other):
            return self.dct == other.dct

        def __iter__(self):
            return self.dct.__iter__()

        def __copy__(self):
            new_dict = type(self)()
            new_dict.__dict__.update(self.__dict__)
            new_dict.dct = copy.copy(self.dct)
            return new_dict

        def __deepcopy__(self, memo):
            new_dict = type(self)()
            new_dict.__dict__.update(self.__dict__)
            new_dict.dct = copy.deepcopy(self.dct)
            return new_dict

    return Interface


def TypedList(elem_type_func):
    """
    Returns a class (not an instance) that will provide a list-like
    interface with runtime type checks.

    Note: The types are provided lazily via a callable to avoid unnecessary dependencies
    on types from external packages at import-time.

    Args:
        elem_type_func (Callable() -> type):
                A callable that returns the expected list-element type.
    """

    class Interface:
        def __init__(self, lst=None):
            self.lst = util.default(lst, [])
            self.elem_type = elem_type_func()

        def _check_type(self, elem):
            if not isinstance(elem, self.elem_type):
                G_LOGGER.critical(
                    f"Unsupported element type type in {type(self).__name__}. Element: {repr(elem)} is type: {type(elem).__name__} but type: {self.elem_type.__name__} was expected"
                )

        def __contains__(self, key):
            return key in self.lst

        def __getitem__(self, index):
            return self.lst[index]

        def __setitem__(self, index, elem):
            self._check_type(elem)
            self.lst[index] = elem

        def __str__(self):
            return str(self.lst)

        def __repr__(self):
            return repr(self.lst)

        def append(self, elem):
            self._check_type(elem)
            return self.lst.append(elem)

        def extend(self, elems):
            for elem in elems:
                self._check_type(elem)
            return self.lst.extend(elems)

        def __iadd__(self, elems):
            for elem in elems:
                self._check_type(elem)
            self.lst += elems

        def __len__(self):
            return len(self.lst)

        def __eq__(self, other):
            return self.lst == other.lst

        def __iter__(self):
            return self.lst.__iter__()

        def __copy__(self):
            new_list = type(self)()
            new_list.__dict__.update(self.__dict__)
            new_list.lst = copy.copy(self.lst)
            return new_list

        def __deepcopy__(self, memo):
            new_list = type(self)()
            new_list.__dict__.update(self.__dict__)
            new_list.lst = copy.deepcopy(self.lst)
            return new_list

    return Interface
