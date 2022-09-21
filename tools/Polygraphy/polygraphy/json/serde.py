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

import base64
import functools
import io
import json
from collections import OrderedDict

from polygraphy import config, constants, mod, util
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")

TYPE_STRING_PREFIX = "__polygraphy_encoded_"


def legacy_str_from_type(typ):
    return TYPE_STRING_PREFIX + typ.__name__


def str_from_type(typ):
    return typ.__name__


class BaseCustomImpl:
    """
    Base class for Polygraphy's JSON encoder/decoder.
    """

    @classmethod
    def register(cls, typ):
        """
        Decorator that registers JSON encoding/decoding functions for types.

        For the documentation that follows, assume we have a class:
        ::

            class Dummy:
                def __init__(self, x):
                    self.x = x

        ========
        Encoders
        ========

        Encoder functions should accept instances of the specified type and
        return dictionaries.

        For example:
        ::

            @Encoder.register(Dummy)
            def encode(dummy):
                return {"x": dummy.x}


        To use the custom encoder, use the `to_json` helper:
        ::

            d = Dummy(x=1)
            d_json = to_json(d)


        ========
        Decoders
        ========

        Decoder functions should accept dictionaries, and return instances of the
        type.

        For example:
        ::

            @Decoder.register(Dummy)
            def decode(dct):
                return Dummy(x=dct["x"])


        To use the custom decoder, use the `from_json` helper:
        ::

            from_json(d_json)


        Args:
            typ (type): The type of the class for which to register the function.
        """

        def register_impl(func):
            def add(key, val):
                if key in cls.polygraphy_registered:
                    G_LOGGER.critical(
                        f"Duplicate serialization function for type: {key}.\nNote: Existing function: {cls.polygraphy_registered[key]}, New function: {func}"
                    )
                cls.polygraphy_registered[key] = val

            if cls == Encoder:

                def wrapped(obj):
                    dct = func(obj)
                    dct[constants.TYPE_MARKER] = str_from_type(typ)
                    return dct

                add(typ, wrapped)
                return wrapped
            elif cls == Decoder:

                def wrapped(dct):
                    if constants.TYPE_MARKER in dct:
                        del dct[constants.TYPE_MARKER]

                    type_name = legacy_str_from_type(typ)
                    if type_name in dct:
                        del dct[type_name]

                    return func(dct)

                add(legacy_str_from_type(typ), wrapped)
                add(str_from_type(typ), wrapped)
            else:
                G_LOGGER.critical("Cannot register for unrecognized class type: ")

        return register_impl


@mod.export()
class Encoder(BaseCustomImpl, json.JSONEncoder):
    """
    Polygraphy's custom JSON Encoder implementation.
    """

    polygraphy_registered = {}

    def default(self, o):
        if type(o) in self.polygraphy_registered:
            return self.polygraphy_registered[type(o)](o)
        return super().default(o)


@mod.export()
class Decoder(BaseCustomImpl):
    """
    Polygraphy's custom JSON Decoder implementation.
    """

    polygraphy_registered = {}

    def __call__(self, pairs):
        dct = OrderedDict(pairs)

        if config.INTERNAL_CORRECTNESS_CHECKS:
            custom_type_keys = [key for key in dct if key.startswith(TYPE_STRING_PREFIX)]
            if custom_type_keys and custom_type_keys[0] not in self.polygraphy_registered:
                G_LOGGER.internal_error(
                    f"Custom type has no decode function registered! Note: Encoded object is:\n{dct}"
                )

        # The encoder will insert special key-value pairs into dictionaries encoded from
        # custom types. If we find one, then we know to decode using the corresponding custom
        # type function.
        type_name = dct.get(constants.TYPE_MARKER)
        func = self.polygraphy_registered.get(type_name)
        if func:
            return func(dct)

        for type_str, func in self.polygraphy_registered.items():
            if type_str in dct and dct[type_str] == constants.LEGACY_TYPE_MARKER:  # Found a custom type!
                return func(dct)
        return dct


NUMPY_REGISTRATION_SUCCESS = False
COMMON_REGISTRATION_SUCCESS = False


def try_register_common_json(func):
    """
    Decorator that attempts to register common JSON encode/decode methods
    if the methods have not already been registered.

    This needs to be attempted multiple times because dependencies may become available in the
    middle of execution - for example, if using dependency auto-installation.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        global NUMPY_REGISTRATION_SUCCESS
        if not NUMPY_REGISTRATION_SUCCESS and mod.has_mod("numpy"):
            # We define this alongside load_json/save_json so that it is guaranteed to be
            # imported before we need to encode/decode NumPy arrays.
            @Encoder.register(np.ndarray)
            def encode(array):
                outfile = io.BytesIO()
                np.save(outfile, array, allow_pickle=False)
                outfile.seek(0)
                data = base64.b64encode(outfile.read()).decode()
                return {"array": data}

            @Decoder.register(np.ndarray)
            def decode(dct):
                def load(mode="base64"):
                    if mode == "base64":
                        data = base64.b64decode(dct["array"].encode(), validate=True)
                    elif mode == "latin-1":
                        data = dct["array"].encode(mode)
                    else:
                        assert False, f"Unsupported mode: {mode}"
                    infile = io.BytesIO(data)
                    return np.load(infile, allow_pickle=False)

                try:
                    arr = load()
                except:
                    arr = load("latin-1")  # For backwards compatibility
                if isinstance(arr, np.ndarray):
                    return arr
                return list(arr.values())[0]  # For backwards compatibility

            NUMPY_REGISTRATION_SUCCESS = True

        global COMMON_REGISTRATION_SUCCESS
        if not COMMON_REGISTRATION_SUCCESS:
            # Pull in some common types so that we can get their associated serialization/deserialization
            # functions. This allows the user to avoid importing these manually.
            # Note: We can only do this here for submodules with no external dependencies.
            # That means, for example, nothing from `backend/` can be imported here.
            from polygraphy.common import FormattedArray
            from polygraphy.comparator import RunResults

            COMMON_REGISTRATION_SUCCESS = True
        return func(*args, **kwargs)

    return wrapped


@mod.export()
@try_register_common_json
def to_json(obj):
    """
    Encode an object to JSON.

    NOTE: For Polygraphy objects, you should use the ``to_json()`` method instead.

    Returns:
        str: A JSON representation of the object.
    """
    return json.dumps(obj, cls=Encoder, indent=constants.TAB)


@mod.export()
@try_register_common_json
def from_json(src):
    """
    Decode a JSON string to an object.

    NOTE: For Polygraphy objects, you should use the ``from_json()`` method instead.

    Args:
        src (str):
                The JSON representation of the object

    Returns:
        object: The decoded instance
    """
    return json.loads(src, object_pairs_hook=Decoder())


@mod.export()
@try_register_common_json
def save_json(obj, dest, description=None):
    """
    Encode an object as JSON and save it to a file.

    NOTE: For Polygraphy objects, you should use the ``save()`` method instead.

    Args:
        obj : The object to save.
        src (Union[str, file-like]): The path or file-like object to save to.
    """
    util.save_file(to_json(obj), dest, mode="w", description=description)


@mod.export()
@try_register_common_json
def load_json(src, description=None):
    """
    Loads a file and decodes the JSON contents.

    NOTE: For Polygraphy objects, you should use the ``load()`` method instead.

    Args:
        src (Union[str, file-like]): The path or file-like object to load from.

    Returns:
        object: The object, or `None` if nothing could be read.
    """
    return from_json(util.load_file(src, mode="r", description=description))


@mod.export()
def add_json_methods(description=None):
    """
    Decorator that adds 4 JSON helper methods to a class:

    - to_json(): Convert to JSON string
    - from_json(): Convert from JSON string
    - save(): Convert to JSON and save to file
    - load(): Load from file and convert from JSON

    Args:
        description (str):
                A description of what is being saved or loaded.
    """

    def add_json_methods_impl(cls):
        # JSON methods

        def check_decoded(obj):
            if not isinstance(obj, cls):
                G_LOGGER.critical(
                    f"Provided JSON cannot be decoded into a {cls.__name__}.\nNote: JSON was decoded into a {type(obj)}:\n{obj}"
                )
            return obj

        def _to_json_method(self):
            """
            Encode this instance as a JSON object.

            Returns:
                str: A JSON representation of this instance.
            """
            return to_json(self)

        def _from_json_method(src):
            return check_decoded(from_json(src))

        _from_json_method.__doc__ = f"""
            Decode a JSON object and create an instance of this class.

            Args:
                src (str):
                        The JSON representation of the object

            Returns:
                {cls.__name__}: The decoded instance

            Raises:
                PolygraphyException:
                        If the JSON cannot be decoded to an instance of {cls.__name__}
            """

        cls.to_json = _to_json_method
        cls.from_json = staticmethod(_from_json_method)

        # Save/Load methods

        def _save_method(self, dest):
            """
            Encode this instance as a JSON object and save it to the specified path
            or file-like object.

            Args:
                dest (Union[str, file-like]):
                      The path or file-like object to write to.

            """
            save_json(self, dest, description=description)

        def _load_method(src):
            return check_decoded(load_json(src, description=description))

        _load_method.__doc__ = f"""
            Loads an instance of this class from a JSON file.

            Args:
                src (Union[str, file-like]): The path or file-like object to read from.

            Returns:
                {cls.__name__}: The decoded instance

            Raises:
                PolygraphyException:
                        If the JSON cannot be decoded to an instance of {cls.__name__}
            """

        cls.save = _save_method
        cls.load = staticmethod(_load_method)

        return cls

    return add_json_methods_impl
