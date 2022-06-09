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
from polygraphy import mod, util
from polygraphy.backend.base import BaseLoader


@mod.export(funcify=True)
class BytesFromPath(BaseLoader):
    """
    Functor that can load a file in binary mode ('rb').
    """

    def __init__(self, path):
        """
        Loads a file in binary mode ('rb').

        Args:
            path (str): The file path.
        """
        self._path = path

    def call_impl(self):
        """
        Returns:
            bytes: The contents of the file.
        """
        return util.load_file(self._path, description="bytes")


@mod.export(funcify=True)
class SaveBytes(BaseLoader):
    """
    Functor that can save bytes to a file.
    """

    def __init__(self, obj, path):
        """
        Saves bytes to a file.

        Args:
            obj (Union[bytes, Callable() -> bytes]):
                    The bytes to save or a callable that returns them.
            path (str): The file path.
        """
        self._bytes = obj
        self._path = path

    def call_impl(self):
        """
        Returns:
            bytes: The bytes saved.
        """
        obj, _ = util.invoke_if_callable(self._bytes)
        util.save_file(obj, self._path)
        return obj


@mod.export(funcify=True)
class InvokeFromScript(BaseLoader):
    """
    Functor that invokes a function from a Python script.
    """

    def __init__(self, path, name):
        """
        Invokes the specified function from the specified Python script.

        If you intend to use the function more than once, you should import
        the function using ``polygraphy.mod.import_from_script`` instead.

        Args:
            path (str): The path to the Python script. The path must include a '.py' extension.
            name (str): The name of the function to import and invoke.
        """
        self._path = path
        self._name = name

    def call_impl(self, *args, **kwargs):
        """
        Returns:
            object:
                    The return value of the imported function.
        """
        return mod.import_from_script(self._path, self._name)(*args, **kwargs)
