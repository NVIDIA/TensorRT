#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from polygraphy import constants, mod, util
from polygraphy.common import TensorMetadata
from polygraphy.datatype import DataType
from polygraphy.logger import G_LOGGER
from polygraphy.tools.script import Script, ensure_safe, inline, safe


@mod.export()
def cast(val):
    """
    Cast a value from a string to one of:
    [int, float, str, List[int], List[float], List[str]]

    Args:
        val (str): The value to cast.

    Returns:
        object: The casted value.
    """
    val = str(val.strip())

    if val.strip("[]") != val:
        return [cast(elem) for elem in val.strip("[]").split(",")]

    try:
        return int(val)  # This fails for float strings like '0.0'
    except:
        pass

    try:
        return float(val)  # This fails for non-numerical strings like 'isildur'
    except:
        pass
    return val.strip("\"'")


@mod.export()
def run_script(script_func, *args):
    """
    Populates a script using the provided callable, then returns
    the variable indicated by the return value of the callable.

    Args:
        script_func (Callable(Script, *args) -> str):
                A callable that populates a Script and then returns
                the name of an object defined within the script to retrieve.
        args:
                Additional positional argruments to pass to script_func.
                The script_func must accept these by variable name instead
                of taking the values themselves. Values of ``None`` will be
                passed directly instead of by variable name.

    Returns:
        object:
                An object defined within the script, or ``None`` if it is not
                defined by the script.
    """
    script = Script()

    arg_names = []
    for index, arg in enumerate(args):
        if arg is not None:
            arg_name = safe("__arg{:}", index)
            locals()[arg_name.unwrap()] = arg
            arg_names.append(inline(arg_name))
        else:
            arg_names.append(None)

    safe_ret_name = script_func(script, *arg_names)
    exec(str(script), globals(), locals())

    if safe_ret_name is not None:
        ret_name = ensure_safe(safe_ret_name).unwrap()
        if ret_name in locals():
            return locals()[ret_name]
    return None


@mod.export()
def get(args, attr, default=None):
    """
    Gets a command-line argument if it exists, otherwise returns a default value.

    Args:
        args: The command-line arguments.
        attr (str): The name of the command-line argument.
        default (obj): The default value to return if the argument is not found. Defaults to None.
    """
    return util.try_getattr(args, attr, default)


@mod.export()
def get_outputs(args, name):
    outputs = get(args, name)
    if outputs is not None and len(outputs) == 2 and outputs == ["mark", "all"]:
        outputs = constants.MARK_ALL
    return outputs


@mod.export()
def get_outputs_for_script(script, outputs):
    if outputs == constants.MARK_ALL:
        script.add_import(["constants"], frm="polygraphy")
        outputs = inline(safe("constants.MARK_ALL"))
    return outputs


def datatype_from_str(dt_str):
    """
    Converts a string representation of a data type to a Polygraphy data type.

    Args:
        dt_str (str): The string representation of the data type.

    Returns:
        DataType: The Polygraphy data type.

    Raises:
        KeyError: If the provided string does not correspond to a NumPy data type.
    """
    try:
        return DataType.__members__[dt_str.upper()]
    except KeyError:
        G_LOGGER.error(
            f"Could not understand data type: {dt_str}. Did you forget to specify a data type? "
            f"Please use one of: {list(DataType.__members__.keys())} or `auto`."
        )
        raise


@mod.export()
def parse_arglist_to_tuple_list(
    arg_lst, cast_to=None, sep=None, allow_empty_key=None, treat_missing_sep_as_val=None, treat_unspecified_as_none=None
):
    """
    Generate a list of (key, value) pairs from a list of arguments of the form:
    ``<key><sep><val>``.

    If the argument is missing a separator,

    - If `treat_missing_sep_as_val` is True, then the argument is treated as a
      value with empty key, i.e. it is parsed as ``<sep><val>``.
    - If `treat_missing_sep_as_val` is False, then the argument is treated as a
      key with empty value, i.e. it is parsed as ``<key><sep>``.

    If `allow_empty_key` is False, then this function will log a critical error if
    any empty keys are detected.

    Args:
        arg_lst (List[str]):
                The arguments to map.

        cast_to (Callable):
                A callable to cast types before adding them to the map.
                Defaults to `cast()`.
        sep (str):
                The separator between the key and value strings.
                Defaults to ":".
        allow_empty_key (bool):
                Whether empty keys should be allowed.
                Defaults to True.
        treat_missing_sep_as_val (bool):
                Whether the argument should be treated as a value with empty key
                when separator is missing (see above).
                Defaults to True.
        treat_unspecified_as_none (bool):
                Whether to treat unspecified keys and values as `None`s instead of
                empty strings.
                Defaults to False.

    Returns:
        Optional[List[Tuple[str, obj]]]:
                The parsed list, or None if arg_lst is None (indicating the flag
                was not specified).
    """
    sep = util.default(sep, ":")
    cast_to = util.default(cast_to, cast)
    allow_empty_key = util.default(allow_empty_key, True)
    treat_missing_sep_as_val = util.default(treat_missing_sep_as_val, True)
    treat_unspecified_as_none = util.default(treat_unspecified_as_none, False)

    if arg_lst is None:
        return None

    ret = []
    for arg in arg_lst:
        key, parsed_sep, val = arg.rpartition(sep)

        if parsed_sep == "" and not treat_missing_sep_as_val:
            key, val = val, key

        if not key and not allow_empty_key:
            G_LOGGER.critical(
                f"Could not parse argument: {arg}. Expected an argument in the format: `key{sep}value`.\n"
            )

        val = cast_to(val)

        if treat_unspecified_as_none:
            key = key or None
            val = val or None

        ret.append((key, val))

    return ret


@mod.export()
def parse_arg_to_tuple(
    arg, cast_to=None, sep=None, allow_empty_key=None, treat_missing_sep_as_val=None, treat_unspecified_as_none=None
):
    """
    Similar to `parse_arglist_to_tuple_list` but operates on a single argument and returns a single tuple
    instead of a list of tuples.

    Args:
        arg (str): The argument.

    Returns:
        Optional[Tuple[str, obj]]:
                The parser key-value pair, or None if `arg` is None (indicating the flag
                was not specified).
    """
    if arg is None:
        return None

    tuple_list = parse_arglist_to_tuple_list(
        [arg], cast_to, sep, allow_empty_key, treat_missing_sep_as_val, treat_unspecified_as_none
    )
    if tuple_list is None:
        return None

    if len(tuple_list) != 1:
        G_LOGGER.critical(
            f"Failed to parse argument: {arg}. Expected an argument of the form: "
            f"`key{sep}value`{f' or `value`' if allow_empty_key else ''}."
        )
    return tuple_list[0]


@mod.export()
def parse_arglist_to_dict(
    arg_lst, cast_to=None, sep=None, allow_empty_key=None, treat_missing_sep_as_val=None, treat_unspecified_as_none=None
):
    """
    Similar to `parse_arglist_to_tuple_list` but returns a dictionary instead of a list of tuples.

    Returns:
        Optional[Dict[str, obj]]:
                The parsed key-value map, or None if arg_lst is None (indicating the flag
                was not specified).
    """
    tuple_list = parse_arglist_to_tuple_list(
        arg_lst, cast_to, sep, allow_empty_key, treat_missing_sep_as_val, treat_unspecified_as_none
    )
    if tuple_list is None:
        return None
    return dict(tuple_list)


@mod.export()
def parse_script_and_func_name(arg, default_func_name=None):
    if arg is None:
        return None, None

    # On Windows we need to split the drive letter (e.g. 'C:') so it's not confused with the script/function separator.
    drive_letter, arg = os.path.splitdrive(arg)
    script_and_func_name = parse_arg_to_tuple(arg, treat_missing_sep_as_val=False, treat_unspecified_as_none=True)
    if script_and_func_name is not None:
        script, func_name = script_and_func_name
        func_name = util.default(func_name, default_func_name)
    else:
        script, func_name = None, None

    script = drive_letter + script
    return script, func_name


@mod.export()
def parse_meta(meta_args, includes_shape=True, includes_dtype=True):
    """
    Parses a list of tensor metadata arguments of the form "<name>:<shape>:<dtype>"
    `shape` and `dtype` are optional, but `dtype` must always come after `shape` if they are both enabled.

    Args:
        meta_args (List[str]): A list of tensor metadata arguments from the command-line.
        includes_shape (bool): Whether the arguments include shape information.
        includes_dtype (bool): Whether the arguments include dtype information.

    Returns:
        TensorMetadata: The parsed tensor metadata.
    """
    SEP = ":"
    meta = TensorMetadata()
    for meta_arg in meta_args:
        name, shape, dtype = None, None, None

        def pop_meta(func):
            nonlocal meta_arg
            meta_arg, _, val = meta_arg.rpartition(SEP)
            val = cast(val.strip())
            if isinstance(val, str) and val.lower() == "auto":
                return None
            return func(val)

        if includes_dtype:
            dtype = pop_meta(func=datatype_from_str)

        if includes_shape:
            shape = pop_meta(func=lambda s: tuple(e for e in s if e != ""))

        name = meta_arg

        meta.add(name, dtype=dtype, shape=shape)
    return meta


@mod.export()
def parse_num_bytes(num_bytes_arg):
    """
    Parses an argument that indicates a number of bytes. The argument may use scientific notation,
    or contain a `K`, `M`, or `G` suffix (case-insensitive), indicating `KiB`, `MiB`, or `GiB` respectively.
    If the number is fractional, it will be truncated to the nearest integer value.

    If the provided argument is `None`, `None` is returned.

    Args:
        num_bytes_arg (str): The argument indicating the number of bytes.

    Returns:
        int: The number of bytes.
    """
    if num_bytes_arg is None:
        return None

    num_component = num_bytes_arg  # Numerical component of the argument
    multiplier = 1

    suffix_mulitplier = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    for suffix, mult in suffix_mulitplier.items():
        if num_bytes_arg.upper().endswith(suffix):
            num_component = num_bytes_arg.upper().rstrip(suffix)
            multiplier = mult
            break

    try:
        return int(float(num_component) * multiplier)
    except:
        G_LOGGER.critical(
            f"Could not convert {num_bytes_arg} to a number of bytes. "
            "Please use either an integer (e.g. 16000000), scientific notation (e.g. 16e6), "
            "or a number with a valid suffix: K, M, or G (e.g. 16M)."
        )


@mod.export()
def parse_path(path, name):
    """
    Parses a path from a command-line argument.

    If the path does not exist, emits a message using the specified logging function.

    Args:
        path (str): The path.
        name (str): Name of what the path refers to.

    Returns:
        str: The path, converted to an absolute path if it exists.
    """
    if path is None:
        return None

    if os.path.exists(path):
        path = os.path.abspath(path)
    else:
        G_LOGGER.warning(f"{name} path does not exist: {path}")
    return path
