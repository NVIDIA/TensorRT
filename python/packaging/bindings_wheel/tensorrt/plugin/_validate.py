#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import inspect
import numpy as np
import typing
import types

from ._utils import _is_numpy_array, _join_with, _infer_numpy_type, _is_npt_ndarray
from ._tensor import TensorDesc, Tensor, SymExprs
from ._export import IS_AOT_ENABLED
if IS_AOT_ENABLED:
    from ._tensor import KernelLaunchParams
from ._autotune import AutoTuneCombination

SERIALIZABLE_BUILTIN_TYPES = (int, float, bytes, bool, str)
SERIALIZABLE_NP_DTYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    bool,
    np.bool_,
)

# Reserve some namespaces for future use/avoid confusion
RESERVED_NAMESPACES = {
    "",
    "trt",
    "tensorrt",
    "std",
}

DISALLOWED_ATTR_NAMES = {
    "outputs",
    "stream",
    "tactic",
}

def _validate_name_and_namespace(ns: str, name: str):
    if "." in ns:
        raise ValueError(
            f"Provided namespace {ns} cannot have any '.' in trt.plugin.register(\"{ns}::{name}\", ...)"
        )

    if "." in name:
        raise ValueError(
            f"Provided name {name} cannot have any '.' in trt.plugin.register(\"{ns}::{name}\", ...)"
        )

    if ns in RESERVED_NAMESPACES:
        raise ValueError(
            f"Provided namespace {ns} is a reserved namespace"
        )


# Parse `tensorrt.plugin.register` schema
def _parse_register_inputs(register_func, lazy_register):
    tensor_names = []
    input_attrs = (
        dict()
    )  # order is important here but for Python >= 3.7, dict respects key order

    schema_chunks = []

    # TensorDescs and attribute args cannot be interspersed, so remember when we saw the first attribute arg
    saw_first_attr = False

    # Map of (attr_name: str) -> (is_builtin_type?: bool, type annotation: str)
    attrs_types = {}

    sig = inspect.signature(register_func)

    for idx, (name, param) in enumerate(sig.parameters.items()):

        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise ValueError(
                f"Argument {name} is not a positional-or-keyword or keyword-only arg"
            )

        # Type annotations are manadatory for `tensorrt.plugin.register` args
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Argument {name} does not have a type annotation. Please mark as TensorDesc or one of the serializable attribute types."
            )

        # Presently, we do not support default values for attributes
        if param.default is not inspect.Parameter.empty:
            raise ValueError(
                f"Argument {name} has a default value. Default values are not supported yet."
            )


        if issubclass(param.annotation, TensorDesc):
            if saw_first_attr:
                raise ValueError(
                    f"TensorDescs args and attribute args cannot be interspersed. Received function with signature {sig}."
                )

            tensor_names.append(name)
            schema_chunks.append(f"TensorDesc {name}")
        # At this point, we don't validate attribute types since we only care about the types of serializable attributes
        # However, we memorize name and type so that we may validate that the autotune function maintains consistency
        else:
            if idx == 0:
                raise ValueError(
                    f"TensorDescs args should come first, followed by attributes. Received function with signature {sig}."
                )

            if name in DISALLOWED_ATTR_NAMES:
                raise ValueError(
                    f"'{name}' is not allowed as a plugin attribute name."
                )

            if param.annotation not in SERIALIZABLE_BUILTIN_TYPES:
                if _is_numpy_array(param.annotation):
                    if not lazy_register:
                        if param.annotation == np.ndarray:
                            raise ValueError(
                                "If using non-lazy registration, annotate numpy array attributes using 'numpy.typing.NDArray[dtype]', where 'dtype' is the expected numpy dtype of the array."
                            )

                        if _is_npt_ndarray(param.annotation):
                            np_dtype = _infer_numpy_type(param.annotation)
                            if np_dtype not in SERIALIZABLE_NP_DTYPES:
                                raise ValueError(
                                    f"Attribute '{name}' is not a supported numpy array type. Supported numpy arrays type are {SERIALIZABLE_NP_DTYPES}."
                                )
                            attrs_types[name] = (False, np_dtype)

                else:
                    raise ValueError(
                        f"Attribute '{name}' of type {param.annotation} is not a supported serializable type. Supported types are {SERIALIZABLE_BUILTIN_TYPES} or numpy arrays of type {SERIALIZABLE_NP_DTYPES}."
                    )
            else:
                attrs_types[name] = (True, param.annotation)

            saw_first_attr = True

            schema_chunks.append(f"{param.annotation} {name}")
            input_attrs[name] = param.annotation

    return (
        tensor_names,
        input_attrs,
        f"({_join_with(schema_chunks)})",
        attrs_types,
    )


def _parse_register_return(register_func):
    sig = inspect.signature(register_func)

    ret_annotation = sig.return_annotation

    if ret_annotation == inspect.Parameter.empty:
        raise ValueError(
            f"No return annotation found for register function. Received signature {sig}."
        )

    if typing.get_origin(ret_annotation) is not tuple:
        if not inspect.isclass(ret_annotation) or not issubclass(
            ret_annotation, TensorDesc
        ):
            raise ValueError(
                f"Return argument is of type {ret_annotation}. Return types can only be TensorDesc or Tuple[TensorDesc]."
            )

        num_outputs = 1
    else:
        args = typing.get_args(ret_annotation)

        for arg in args:
            if not issubclass(arg, TensorDesc):
                raise ValueError(
                    f"Return argument is of type {ret_annotation}. Return types can only be TensorDesc or Tuple[TensorDesc]."
                )

        num_outputs = len(args)

    return num_outputs


def _validate_impl(impl_func, plugin_def):
    impl_attr_names = []
    found_tactic = False

    sig = inspect.signature(impl_func)
    registered_attr_names = plugin_def.input_attrs.keys()

    # input arg annotations are optional, but we will validate if provided
    for name, param in sig.parameters.items():
        # tactic arg is optional in impl function. If specified, remember so that we can pass it during enqueue.
        if name == "tactic":
            found_tactic = True
        if param.annotation != inspect.Parameter.empty:
            if name == "outputs":
                if typing.get_origin(param.annotation) is not tuple:
                    raise ValueError(
                        f"'outputs' should be of type Tuple[Tensor]. Received {param.annotation}."
                    )
                args = typing.get_args(param.annotation)
                for arg in args:
                    if not issubclass(arg, Tensor):
                        raise ValueError(
                            f"Argument for receiving output Tensor, '{name}' contains a {param.annotation}. '{name}' should be a Tuple[Tensor]."
                        )
            elif name == "stream":
                if not issubclass(param.annotation, int):
                    raise ValueError("'stream' input argument should be an int")
            elif name == "tactic":
                if not issubclass(param.annotation, int):
                    raise ValueError("'tactic' input argument should be an int")
            elif issubclass(param.annotation, Tensor):
                if name not in plugin_def.input_tensor_names:
                    raise ValueError(
                        f"Unexpected tensor '{name}' specified in autotune function. Expected one of {plugin_def.input_tensor_names}."
                    )
            else:
                if name not in plugin_def.input_attrs:
                    raise ValueError(
                        f"Unexpected attribute '{name}' specified in impl function. Expected one of {list(registered_attr_names)}."
                    )

                if param.annotation != plugin_def.input_attrs[name]:
                    raise ValueError(
                        f"Attribute '{name}' has a type annotation different from the one specified at registration. Expected '{plugin_def.input_attrs[name]}'."
                    )

                impl_attr_names.append(name)
        else:
            if name in plugin_def.input_attrs:
                impl_attr_names.append(name)

    # Expected attribute schema should be constructed in the order they appeared in the register function
    expected_attr_schema_chunks = [
        n for n in registered_attr_names if n in impl_attr_names
    ]

    expected_schema = (
        "("
        + _join_with(plugin_def.input_tensor_names)
        + _join_with(expected_attr_schema_chunks, True)
        + ", outputs, stream"
    )
    if found_tactic:
        expected_schema += ", tactic)"
    else:
        expected_schema += ")"

    if f"({', '.join(sig.parameters.keys())})" != expected_schema:
        raise ValueError(
            f"Signature of the impl function '{sig}' does not match the expected input arg schema: {expected_schema}"
        )

    # Return annotation is optional, but we will validate if one is specified
    if sig.return_annotation != inspect.Parameter.empty and sig.return_annotation is not None:
        raise ValueError("Return annotation should be None.")

    return impl_attr_names, found_tactic

def _validate_aot_impl(aot_impl_func, plugin_def):
    aot_impl_attr_names = []

    sig = inspect.signature(aot_impl_func)
    registered_attr_names = plugin_def.input_attrs.keys()

    # input arg annotations are optional, but we will validate if provided
    for name, param in sig.parameters.items():
        if param.annotation != inspect.Parameter.empty:
            if name == "outputs":
                if typing.get_origin(param.annotation) is not tuple:
                    raise ValueError(
                        f"'outputs' should be of type Tuple[TensorDesc]. Received {param.annotation}."
                    )
                args = typing.get_args(param.annotation)
                for arg in args:
                    if not issubclass(arg, TensorDesc):
                        raise ValueError(
                            f"Argument for receiving output TensorDesc, '{name}' contains a {param.annotation}. '{name}' should be a Tuple[TensorDesc]."
                        )
            elif name == "tactic":
                if not issubclass(param.annotation, int):
                    raise ValueError("'tactic' input argument should be an int")
            elif issubclass(param.annotation, TensorDesc):
                if name not in plugin_def.input_tensor_names:
                    raise ValueError(
                        f"Unexpected tensor '{name}' specified in autotune function. Expected one of {plugin_def.input_tensor_names}."
                    )
            else:
                if name not in plugin_def.input_attrs:
                    raise ValueError(
                        f"Unexpected attribute '{name}' specified in aot_impl function. Expected one of {list(registered_attr_names)}."
                    )

                if param.annotation != plugin_def.input_attrs[name]:
                    raise ValueError(
                        f"Attribute '{name}' has a type annotation different from the one specified at registration. Expected '{plugin_def.input_attrs[name]}'."
                    )

                aot_impl_attr_names.append(name)
        else:
            if name in plugin_def.input_attrs:
                aot_impl_attr_names.append(name)

    # Expected attribute schema should be constructed in the order they appeared in the register function
    expected_attr_schema_chunks = [
        n for n in registered_attr_names if n in aot_impl_attr_names
    ]

    expected_schema = (
        "("
        + _join_with(plugin_def.input_tensor_names)
        + _join_with(expected_attr_schema_chunks, True)
        + ", outputs, tactic)"
    )

    if f"({', '.join(sig.parameters.keys())})" != expected_schema:
        raise ValueError(
            f"Signature of the aot_impl function '{sig}' does not match the expected input arg schema: {expected_schema}"
        )

    ret_annotation = sig.return_annotation

    if ret_annotation == inspect.Parameter.empty:
        raise ValueError(
            f"No return annotation found for aot_impl function. Received signature {sig}."
        )

    expected_return_schema = "tuple[str | bytes, str | bytes, tensorrt.plugin.KernelLaunchParams, tensorrt.plugin.SymIntExprs]"

    # Return annotation is optional, but we will validate if one is specified
    if ret_annotation != inspect.Parameter.empty:
        if typing.get_origin(ret_annotation) is not tuple:
            raise ValueError(
                f"Return annotation is {ret_annotation}. Expected {expected_return_schema}."
            )
        else:
            args = typing.get_args(ret_annotation)

            if len(args) != 4:
                raise ValueError(
                    f"Return annotation is {ret_annotation}. Expected {expected_return_schema}."
                )

            def validate_union_str_or_bytes(index):
                def validate_str_or_bytes(arg_):
                    if (arg_ is not str) and (arg_ is not bytes):
                        raise ValueError(
                            f"Return annotation for argument at {index} is '{arg_}'. Expected 'str' or 'bytes'."
                        )

                orig = typing.get_origin(args[index])
                # orig is `typing.Union` when annotation uses typing module (e.g, Union[str, bytes])
                # orig is `types.UnionType` when annotation is of the new (3.10+) native syntax (e.g, str | bytes)
                if orig is typing.Union or orig is types.UnionType:
                    for a in typing.get_args(args[index]):
                        validate_str_or_bytes(a)
                else:
                # when annoted with `str` or `bytes`
                    validate_str_or_bytes(args[index])

            # kernel name should be str or bytes encoding
            validate_union_str_or_bytes(0)
            # kernel PTX should be str or bytes encoding
            validate_union_str_or_bytes(1)

            if not issubclass(args[2], KernelLaunchParams):
                raise ValueError(f"Argument at index 2 of return annotation is '{args[2]}'. Expected 'tensorrt.plugin.KernelLaunchParams'.")

            if not issubclass(args[3], SymExprs):
                raise ValueError(f"Argument at index 3 of return annotation is '{args[3]}'. Expected a descendent of tensorrt.plugin.SymExprs.")

    return aot_impl_attr_names


def _validate_autotune(autotune_func, plugin_def):

    sig = inspect.signature(autotune_func)
    registered_attr_names = plugin_def.input_attrs.keys()

    autotune_attr_names = []

    # input arg annotations are optional, but we will validate if provided
    for name, param in sig.parameters.items():
        if param.annotation != inspect.Parameter.empty:
            if name == "outputs":
                if typing.get_origin(param.annotation) is not tuple:
                    raise ValueError(
                        f"'outputs' should be of type Tuple[TensorDesc]. Received {param.annotation}."
                    )
                args = typing.get_args(param.annotation)
                for arg in args:
                    if not issubclass(arg, TensorDesc):
                        raise ValueError(
                            f"Argument for receiving output TensorDescs, '{name}' contains a {param.annotation}. '{name}' should be a Tuple[TensorDesc]."
                        )
            elif issubclass(param.annotation, TensorDesc):
                if name not in plugin_def.input_tensor_names:
                    raise ValueError(
                        f"Unexpected tensor '{name}' specified in autotune function. Expected one of {plugin_def.input_tensor_names}."
                    )
            else:
                if name not in plugin_def.input_attrs:
                    raise ValueError(
                        f"Unexpected attribute '{name}' specified in autotune function. Expected one of {list(registered_attr_names)}."
                    )
                if param.annotation != plugin_def.input_attrs[name]:
                    raise ValueError(
                        f"Attribute '{name}' has a type annotation different from the one specified at registration. Expected '{plugin_def.input_attrs[name]}'."
                    )

                autotune_attr_names.append(name)
        else:
            if name in plugin_def.input_attrs:
                autotune_attr_names.append(name)

    # Expected attribute schema should be constructed in the order they appeared in the register function
    expected_attr_schema_chunks = [
        n for n in registered_attr_names if n in autotune_attr_names
    ]

    expected_schema = (
        "("
        + _join_with(plugin_def.input_tensor_names)
        + _join_with(expected_attr_schema_chunks, True)
        + ", outputs)"
    )

    if f"({', '.join(sig.parameters.keys())})" != expected_schema:
        raise ValueError(
            f"Specified autotune function signature {sig} is not consistent with the expected input arg schema {expected_schema}."
        )

    ret_annotation = sig.return_annotation

    # Return annotation is optional, but we will validate if one is specified
    if ret_annotation != inspect.Parameter.empty:
        if typing.get_origin(ret_annotation) is not list:
            if not inspect.isclass(ret_annotation) or not issubclass(
                ret_annotation, AutoTuneCombination
            ):
                raise ValueError(
                    f"Return argument is of type {ret_annotation}. Return types can only be AutoTuneCombination or List[AutoTuneCombination]."
                )
        else:
            args = typing.get_args(ret_annotation)

            for arg in args:
                if not issubclass(arg, AutoTuneCombination):
                    raise ValueError(
                        f"Return argument is of type {ret_annotation}. Return types can only be AutoTuneCombination or List[AutoTuneCombination]."
                    )

    return autotune_attr_names
