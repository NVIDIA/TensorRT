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
import functools
from textwrap import dedent

from polygraphy import mod
from polygraphy.exception import DataTypeConversionException
from polygraphy.logger import G_LOGGER, LogMode


import enum


class _SkipImporterException(Exception):
    pass


class _DataTypeKind(enum.Enum):
    FLOATING_POINT = 0
    INTEGRAL = 1
    _OTHER = 3


@mod.export()
class DataTypeEntry:
    """
    Represents a data type.
    Can be transformed to and from data type classes of external modules, like NumPy.

    Do *not* construct objects of this type directly. Instead, use the predefined data types
    provided in the ``DataType`` class.
    """

    def __init__(self, name, itemsize, type: _DataTypeKind):
        self.name = name
        """The human-readable name of the data type"""
        self.itemsize = itemsize
        """The size in bytes of a single element of this data type"""

        # self._type describes the basic kind of the type we have.
        # For example, this can
        self._type = type

    @property
    def is_floating(self):
        return self._type == _DataTypeKind.FLOATING_POINT

    @property
    def is_integral(self):
        return self._type == _DataTypeKind.INTEGRAL

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"DataType.{self.name.upper()}"


@mod.export()
class DataType:
    # Docstring will be populated by the loop below
    """
    Aggregates supported Polygraphy data types. Each data type is accessible
    via this class as a class member of type ``DataTypeEntry``.

    Members:
    """
    _IMPORTER_FUNCS = {}
    _EXPORTER_FUNCS = {}

    __members__ = {
        "FLOAT64": DataTypeEntry("float64", 8, _DataTypeKind.FLOATING_POINT),
        "FLOAT32": DataTypeEntry("float32", 4, _DataTypeKind.FLOATING_POINT),
        "FLOAT16": DataTypeEntry("float16", 2, _DataTypeKind.FLOATING_POINT),
        "FLOAT4": DataTypeEntry("float4", 0.5, _DataTypeKind.FLOATING_POINT),
        "INT16": DataTypeEntry("int16", 2, _DataTypeKind.INTEGRAL),
        "INT32": DataTypeEntry("int32", 4, _DataTypeKind.INTEGRAL),
        "INT64": DataTypeEntry("int64", 8, _DataTypeKind.INTEGRAL),
        "INT8": DataTypeEntry("int8", 1, _DataTypeKind.INTEGRAL),
        "INT4": DataTypeEntry("int4", 0.5, _DataTypeKind.INTEGRAL),
        "UINT16": DataTypeEntry("uint16", 2, _DataTypeKind.INTEGRAL),
        "UINT32": DataTypeEntry("uint32", 4, _DataTypeKind.INTEGRAL),
        "UINT64": DataTypeEntry("uint64", 8, _DataTypeKind.INTEGRAL),
        "UINT8": DataTypeEntry("uint8", 1, _DataTypeKind.INTEGRAL),
        "BOOL": DataTypeEntry("bool", 1, _DataTypeKind._OTHER),
        "STRING": DataTypeEntry("string", 0, _DataTypeKind._OTHER),
        "BFLOAT16": DataTypeEntry("bfloat16", 2, _DataTypeKind.FLOATING_POINT),
        "FLOAT8E4M3FN": DataTypeEntry("float8e4m3fn", 1, _DataTypeKind.FLOATING_POINT),
        "FLOAT8E4M3FNUZ": DataTypeEntry(
            "float8e4m3fnuz", 1, _DataTypeKind.FLOATING_POINT
        ),
        "FLOAT8E5M2": DataTypeEntry("float8e5m2", 1, _DataTypeKind.FLOATING_POINT),
        "FLOAT8E5M2FNUZ": DataTypeEntry(
            "float8e5m2fnuz", 1, _DataTypeKind.FLOATING_POINT
        ),
    }

    @staticmethod
    def from_dtype(dtype, source_module=None):
        """
        Converts a data type from any known external libraries to a corresponding
        Polygraphy data type.

        Args:
            dtype (Any): A data type from an external library.
            source_module (str):
                    The name of the module from where the provided `dtype` originates.
                    If this is not provided, Polygraphy will attempt to guess the module
                    in order to convert the data type.

        Returns:
            DataTypeEntry: The corresponding Polygraphy data type.

        Raises:
            PolygraphyException: If the data type could not be converted.
        """
        if dtype is None:
            G_LOGGER.critical(f"Could not convert: {dtype} to a Polygraphy data type")

        if isinstance(dtype, DataTypeEntry):
            return dtype

        if source_module is not None:
            if source_module not in DataType._IMPORTER_FUNCS:
                G_LOGGER.critical(
                    f"Could not find source module: {source_module} in known importers. "
                    f"Note: Importer functions have been registered for the following modules: {list(DataType._IMPORTER_FUNCS.keys())}"
                )
            try:
                return DataType._IMPORTER_FUNCS[source_module](dtype)
            except _SkipImporterException:
                pass
        else:
            for func in DataType._IMPORTER_FUNCS.values():
                try:
                    ret = func(dtype)
                except _SkipImporterException:
                    pass
                else:
                    return ret

        msg = f"Could not convert: {dtype} to a corresponding Polygraphy data type. Leaving this type in its source format."
        G_LOGGER.warning(msg, mode=LogMode.ONCE)
        G_LOGGER.internal_error(msg)
        return dtype

    @staticmethod
    def to_dtype(dtype, target_module):
        """
        Converts a Polygraphy data type to one from any known external libraries.

        Args:
            dtype (DataType):
                    A Polygraphy data type. If something other than a Polygraphy data type is provided,
                    then this function will return it without modifying it.
            target_module (str):
                    The name of the module whose data type class to convert this data type to.

        Returns:
            Any: The corresponding data type from the target module.

        Raises:
            PolygraphyException: If the data type could not be converted.
        """
        if not isinstance(dtype, DataTypeEntry):
            G_LOGGER.internal_error(
                f"Received input of type other than DataType: {dtype}"
            )
            return dtype

        if target_module not in DataType._EXPORTER_FUNCS:
            G_LOGGER.critical(
                f"Could not find target module: {target_module} in known exporters. "
                f"Note: Exporter functions have been registered for the following modules: {list(DataType._EXPORTER_FUNCS.keys())}"
            )
        return DataType._EXPORTER_FUNCS[target_module](dtype)


DataType.__doc__ = dedent(DataType.__doc__)
for name, value in DataType.__members__.items():
    setattr(DataType, name, value)
    DataType.__doc__ += f"\t- {name}\n"


def register_dtype_importer(source_module):
    """
    Registers an importer function with the DataType class.

    IMPORTANT: You *must* ensure that the importer function does not attempt to automatically install
    or import modules which are not already installed.
    With a lazily imported module, `module.is_installed()/is_importable()` is an easy way to guard the code against this.
    We do not want to automatically install heavy modules like PyTorch or TensorRT just for the sake of DataType.

    For example:
    ::

        @register_dtype_importer("numpy")
        def func(dtype):
            ...

    The importer function should return `None` if no corresponding data type could be found
    or if the input type did not match what was expected.

    The newly registered function is then usable via `from_dtype`:
    ::

        dtype = DataType.from_dtype(np.int64)
    """

    def register_importer_impl(func):
        @functools.wraps(func)
        def new_func(dtype):
            val = func(dtype)
            if val is None:
                # We raise an exception to indicate that `from_dtype` should skip this importer and try a different one.
                # We have to do it this way since we don't necessarily know which importer is the right one to use.
                raise _SkipImporterException()
            return val

        DataType._IMPORTER_FUNCS[source_module] = new_func
        return new_func

    return register_importer_impl


def register_dtype_exporter(target_module):
    """
    Registers an exporter function with the DataType class.

    For example:
    ::

        @register_dtype_exporter("numpy")
        def func(dtype):
            ...

    The newly registered function is then accessible with, for example:
    ::

        np_dtype = DataType.FLOAT32.numpy()
    """

    def register_exporter_impl(func):
        @functools.wraps(func)
        def new_func(dtype):
            val = func(dtype)
            if val is None:
                G_LOGGER.critical(
                    f"Could not convert Polygraphy data type: {dtype} to a corresponding {target_module} data type. ",
                    ExceptionType=DataTypeConversionException,
                )
            return val

        new_func.__name__ = target_module
        setattr(DataTypeEntry, new_func.__name__, new_func)
        DataType._EXPORTER_FUNCS[target_module] = new_func
        return new_func

    return register_exporter_impl
