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
import contextlib
import glob
import os
import sys
import tempfile
import zlib
from collections import OrderedDict

from polygraphy import constants, mod
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")


@mod.export()
def check(cond, msg=None):
    """
    Like assert, but applies even when optimizations are enabled (i.e. __debug__ is False).

    Args:
        cond (bool): The condition to check.
        msg (str): The error message in case condition is False.

    Raises:
        AssertionError: If the condition is False.
    """
    if not cond:
        raise AssertionError(msg)


@mod.export()
def find_in_dict(name, mapping, index=None):
    """
    Attempts to partially match keys in a dictionary. Checks for exact matches and
    substring matches, falling back to index based matching.

    Args:
        name (str): The key to search for.
        mapping (dict): The dictionary to search in.
        index (int): An index to fall back to if the key could not be found by name.

    Returns:
        str: The key found in the dict, or None if it could not be found.
    """
    G_LOGGER.ultra_verbose("Searching for key: {:}. Fallback index is set to {:}".format(name, index))
    if name in mapping:
        return name
    for key in mapping.keys():
        if name.lower() in key.lower() or key.lower() in name.lower():
            return key
    if index is not None and index >= 0 and index < len(mapping.keys()):
        return list(mapping.keys())[index]
    return None


@mod.export()
def check_dict_contains(dct, keys, check_missing=True, dict_name=None, log_func=None):
    """
    Checks that a dictionary contains the provided keys and also
    that it does not contain any extra items and issues warnings
    otherwise.

    Args:
        dct (Dict[Any, Any]):
                The dictionary to check.
        keys (Sequence[Any]):
                The keys that should be in the dictionary.

        check_missing (bool):
                Whether to check for missing keys in the dictionary.
                Defaults to True.
        dict_name (str):
                The name to use instead of "the dictionary" when
                displaying warnings.
        log_func (Logger.method):
                The logging method to use to display warnings/errors.
                Defaults to G_LOGGER.warning.

    Returns:
        bool: Whether the dictionary contains exactly the specified keys.
    """
    log_func = default(log_func, G_LOGGER.warning)
    dict_name = default(dict_name, "the dictionary")

    feed_names = set(dct.keys())
    keys = set(keys)
    missing_in_dct = (keys - feed_names) if check_missing else False
    extra_in_dct = feed_names - keys

    if missing_in_dct:
        log_func(
            "Some keys are missing in {:}: {:}.\n"
            "Note: Expected keys are: {:}, but keys provided were: {:}".format(
                dict_name, missing_in_dct, keys, feed_names
            )
        )

    if extra_in_dct:
        log_func(
            "Extra keys in {:}: {:}.\n"
            "Note: Expected keys are: {:}, but keys provided were: {:}".format(
                dict_name, extra_in_dct, keys, feed_names
            )
        )

    return not extra_in_dct and not missing_in_dct


@mod.export()
def value_or_from_dict(obj, key, default=None):
    """
    Many Polygraphy APIs can accept a `Union[obj, Dict[str, obj]]` to allow
    for specifying either a global value, or a per-key (e.g. input, output, etc.) value.

    When a dictionary is provided, the `""` key indiciates a default value to use for keys
    not otherwise found.

    For example, Polygraphy allows for providing per-output tolerances. Thus, all of the
    following are valid arguments:
    ::

        # Value directly
        atol = 1.0

        # Per-output values
        atol = {"out1": 1.0, "out2": 2.0}

        # Per-output values with default
        atol = {"out1": 1.0, "": 2.0}

    Args:
        obj (Union[obj, Dict[str, obj]]): The value, or per-key values.
        key (str): The key to use when per-key values are provided.
        default (obj): The default value to use if it is not found in the dictionary.

    Returns:
        obj: The value.
    """
    if not isinstance(obj, dict):
        return obj

    if key in obj:
        return obj[key]
    elif "" in obj:
        return obj[""]
    return default


@mod.export()
def unique_list(sequence):
    """
    Creates a list without duplicate elements, preserving order.

    Args:
        sequence (Sequence): The sequence to make unique

    Returns:
        list: A list containing the same elements as sequence, in the same order, but without duplicates.
    """
    return list(OrderedDict.fromkeys(sequence))


# default exists to solve issues that might result from Python's normal default arguments.
# Specifically, consider the following class:
#
# class MyClass(object):
#     def __init__(self, value=[]):
#         self.value = value
#
# This leads to unexpected behavior when the default value is used:
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
#         self.value = default(value, [])
#
# Then we get the expected behavior:
#
# >>> x = MyClass()
# >>> x.value.append("SHOULD NOT BE IN Y")
# >>> y = MyClass()
# >>> y.value
# []
@mod.export()
def default(value, default):
    """
    Returns a specified default value if the provided value is None.

    Args:
        value (object): The value.
        default (object): The default value to use if value is None.

    Returns:
        object: Either value, or the default.
    """
    return value if value is not None else default


@mod.export()
def is_sequence(obj):
    return hasattr(obj, "__iter__") and not isinstance(obj, dict) and not isinstance(obj, set)


@mod.export()
def unpack_args(args, num):
    """
    Extracts the specified number of arguments from a tuple, padding with
    `None` if the tuple length is insufficient.

    Args:
        args (Tuple[object]): The tuple of arguments
        num (int): The number of elements desired.

    Returns:
        Tuple[object]: A tuple containing `num` arguments, padded with `None` if `len(args) < num`
    """
    args = args if is_sequence(args) else (args,)
    args += (None,) * (num - len(args))
    return args[0:num]


##
## File I/O
##


@mod.export()
class NamedTemporaryFile(object):
    """
    Cross-platform temporary file implementation. Unlike tempfile.NamedTemporaryFile,
    it can be opened multiple times without error on Windows.
    """

    def __init__(self, mode=None, prefix=None, suffix=None):
        """
        Args:
            mode (str): The mode to use when opening the file.
            prefix (str): The prefix to use for the file path.
            suffix (str): The suffix to use for the file path.
        """
        self.mode = default(mode, "wb+")
        prefix = default(prefix, "")
        suffix = default(suffix, "")

        def rand_path():
            return os.path.join(tempfile.gettempdir(), "{:}{:}{:}".format(prefix, os.urandom(24).hex(), suffix))

        # In the unlikely event the path exists, generate a new one. Only try 100 times so
        # we don't end up in an infinite loop.
        path = rand_path()
        for _ in range(100):
            if not os.path.exists(path):
                break
            path = rand_path()
        else:
            G_LOGGER.critical("Could not create a temporary file under: {:}".format(tempfile.gettempdir()))

        self.name = path  # Use 'name' to be compatible with tempfile.NamedTemporaryFile
        open(self.name, "x").close()
        self._fhandle = None

    def __enter__(self):
        """
        Opens the temporary file using the mode specified in the constructor.

        Returns:
            file-like: The open file object.
        """
        self._fhandle = open(self.name, self.mode)
        return self._fhandle

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the file handle.
        """
        self._fhandle.close()


@mod.export()
def find_in_dirs(name_glob, dirs):
    """
    Finds a file, optionally including a glob expression, in the specified directories.

    Args:
        name_glob (str):
                The name of the file, optionally including a glob expression.
                Only the first match will be returned.
        dirs (Sequence[str]):
                The directories in which to search.

    Returns:
        List[str]: The paths found, or an empty list if it could not be found.
    """
    for dir_name in dirs:
        paths = glob.glob(os.path.join(dir_name, name_glob))
        if paths:
            return paths
    return []


@mod.export()
def get_file_size(src):
    """
    Gets the size of a file or file-like object.

    Args:
        src (Union[str, file-like]):
                The path or file-like object to read from.

    Returns:
        int: The size of the file if it exists, otherwise 0.
    """
    try:
        src.fileno
    except AttributeError:
        path = src
        if not os.path.exists(path):
            return 0
    else:
        path = src.fileno()

    return os.stat(path).st_size


def warn_if_wrong_mode(file_like, mode):
    def binary(mode):
        return "b" in mode

    def readable(mode):
        return "r" in mode or "+" in mode

    def writable(mode):
        return "w" in mode or "a" in mode or "+" in mode

    fmode = file_like.mode
    if (
        binary(fmode) != binary(mode)
        or (readable(mode) and not readable(fmode))
        or (writable(mode) and not writable(fmode))
    ):
        G_LOGGER.warning(
            "File-like object has a different mode than requested!\n"
            "Note: Requested mode was: {:} but file-like object has mode: {:}".format(mode, file_like.mode)
        )


def is_file_like(obj):
    try:
        obj.read
        obj.write
    except AttributeError:
        return False
    else:
        return True


@mod.export()
def makedirs(path):
    dir_path = os.path.dirname(path)
    if dir_path:
        dir_path = os.path.realpath(dir_path)
        if not os.path.exists(dir_path):
            G_LOGGER.verbose("{:} does not exist, creating now.".format(dir_path))
        os.makedirs(dir_path, exist_ok=True)


@mod.export()
def load_file(src, mode="rb", description=None):
    """
    Reads from the specified source path or file-like object.

    Args:
        src (Union[str, file-like]): The path or file-like object to read from.


        mode (str): The mode to use when reading. Defaults to "rb".
        description (str): A description of what is being read.

    Returns:
        Union[str, bytes, None]: The contents read.

    Raises:
        Exception: If the file or file-like object could not be read.
    """
    if description is not None:
        G_LOGGER.info("Loading {:} from {:}".format(description, src))

    if is_file_like(src):
        warn_if_wrong_mode(src, mode)
        # Reset cursor position after reading from the beginning of the file.
        prevpos = src.tell()
        if src.seekable():
            src.seek(0)
        contents = src.read()
        if src.seekable():
            src.seek(prevpos)
        return contents
    else:
        with open(src, mode) as f:
            return f.read()


@mod.export()
def save_file(contents, dest, mode="wb", description=None):
    """
    Writes text or binary data to the specified destination path or file-like object.

    Args:
        contents (bytes):
                A bytes-like object that can be written to disk.
        dest (Union[str, file-like]):
                The path or file-like object to write to.


        mode (str): The mode to use when writing. Defaults to "wb".
        description (str): A description of what is being written.

    Returns:
        Union[str, file-like, None]: The complete file path or file-like object.

    Raises:
        Exception: If the path could not be written to, or if the file-like object could not be written to.
    """
    if description is not None:
        G_LOGGER.info("Saving {:} to {:}".format(description, dest))

    if is_file_like(dest):
        warn_if_wrong_mode(dest, mode)
        bytes_written = dest.write(contents)
        dest.flush()
        try:
            content_bytes = len(contents.encode())
        except:
            pass
        else:
            if bytes_written != content_bytes:
                G_LOGGER.warning(
                    "Could not write entire file. Note: file contains {:} bytes, but only "
                    "{:} bytes were written".format(content_bytes, bytes_written)
                )
    else:
        makedirs(dest)
        with open(dest, mode) as f:
            f.write(contents)
    return dest


##
## Compression
##


class Compressed(object):
    """
    Represents an object compressed by zlib
    """

    def __init__(self, cobj):
        self.bytes = cobj


def is_compressed(obj):
    return isinstance(obj, Compressed)


def compress(obj):
    G_LOGGER.verbose("Compressing {} object".format(type(obj)))
    return Compressed(zlib.compress(obj))


def decompress(compressed):
    G_LOGGER.verbose("Decompressing bytes")
    return zlib.decompress(compressed.bytes)


##
## Subprocess Utils
##

PIPE_MAX_SEND_BYTES = 1 << 31
"""The maximum number of bytes that can be sent at once over a queue"""


def send_on_queue(queue, obj):
    if sys.getsizeof(obj) > PIPE_MAX_SEND_BYTES:
        G_LOGGER.warning(
            "Object size ({:} bytes) exceeds maximum size that can be sent over queues ({:} bytes). "
            "Attempting to compress - this may take some time. If this does not work or you want to avoid "
            "the compression overhead, you should disable subprocesses by omitting the --use-subprocess flag, "
            "or by setting use_subprocess=False in Comparator.run().".format(sys.getsizeof(obj), PIPE_MAX_SEND_BYTES)
        )
        obj = compress(obj)

    assert sys.getsizeof(obj) <= PIPE_MAX_SEND_BYTES

    G_LOGGER.ultra_verbose("Sending: {:} on queue".format(obj))
    queue.put(obj)


@mod.export()
def try_send_on_queue(queue, obj):
    """
    Attempts to send an object over the queue, compressing it if needed.
    In the event the object cannot be sent, sends `None` instead.

    Args:
        queue (queue.Queue): The queue to send the object over.
        obj (object): The object to send.
    """
    try:
        send_on_queue(queue, obj)
    except Exception as err:
        G_LOGGER.warning("Could not send object on queue: {:}\nSending None instead.".format(err))
        queue.put(None)


def receive_on_queue(queue, timeout=None):
    G_LOGGER.extra_verbose("Waiting for data to become available on queue")
    obj = queue.get(block=True, timeout=timeout)
    if is_compressed(obj):
        obj = decompress(obj)
    G_LOGGER.ultra_verbose("Received {:} on queue".format(obj))
    return obj


@mod.export()
def try_receive_on_queue(queue, timeout=None):
    try:
        obj = receive_on_queue(queue, timeout)
        if obj is None:
            G_LOGGER.warning(
                "Received {:} on the queue. This likely means that there was an error in sending "
                "the object over the queue. You may want to run with use_subprocess=False in Comparator.run() "
                "or omit the --use-subprocess flag to prevent further issues.".format(obj)
            )
        return obj
    except Exception as err:
        G_LOGGER.warning(
            "Could not receive on queue: {:}\nYou may want to run with use_subprocess=False in Comparator.run() "
            "or omit the --use-subprocess flag to prevent further issues.".format(err)
        )
        return None


##
## Function Utils
##


@mod.export()
def invoke_if_callable(func, *args, **kwargs):
    """
    Attempts to invoke a function with arguments. If `func` is not callable, then returns `func`
    The second return value of this function indicates whether the argument was a callable.
    """
    if callable(func):
        ret = func(*args, **kwargs)
        return ret, True
    return func, False


##
## Shapes
##


def is_dimension_dynamic(dim):
    is_dim_str = not isinstance(dim, int)
    return dim is None or is_dim_str or dim < 0


def num_dynamic_dimensions(shape):
    return len([dim for dim in shape if is_dimension_dynamic(dim)])


@mod.export()
def is_shape_dynamic(shape):
    return num_dynamic_dimensions(shape) > 0


@mod.export()
def is_valid_shape_override(new_shape, original_shape):
    ranks_same = len(original_shape) == len(new_shape)
    overrides_valid = all([odim == ndim or is_dimension_dynamic(odim) for odim, ndim in zip(original_shape, new_shape)])
    return ranks_same and overrides_valid


@mod.export()
def override_dynamic_shape(shape, default_shape_value=None):
    default_shape_value = default(default_shape_value, constants.DEFAULT_SHAPE_VALUE)
    return [default_shape_value if is_dimension_dynamic(elem) else elem for elem in shape]


@mod.export()
def volume(obj):
    vol = 1
    for elem in obj:
        vol *= elem
    return vol


@mod.export()
def is_empty_shape(shape):
    return volume(shape) == 0


@mod.export()
def try_match_shape(arr, shape):
    """
    Attempts to permute or reshape the array so its shape matches the specified shape.
    This is a no-op if the array is already the correct shape.

    Args:
        arr (numpy.ndarray): The array to reshape.
        shape (Tuple[int]): The shape to use. May contain at most 1 dynamic dimension.

    Returns:
        numpy.ndarray: The reshaped array.
    """

    def is_rank_same(arr, shape):
        return len(shape) == len(arr.shape)

    def try_reshape(arr, shape):
        original_shape = arr.shape
        try:
            arr = arr.reshape(shape)
        except ValueError:
            G_LOGGER.warning(
                "Could not reshape array from shape: {:} to {:}. Skipping reshape.".format(arr.shape, shape)
            )
        else:
            if arr.shape != original_shape:
                G_LOGGER.info("Reshaped array from shape: {:} to: {:}".format(original_shape, arr.shape))
        return arr

    def try_permute(arr, shape):
        original_shape = arr.shape

        if sorted(arr.shape) != sorted(shape):
            G_LOGGER.extra_verbose("Array of shape: {:} cannot be permuted to: {:}".format(arr.shape, shape))
            return arr

        # We need to remove axes from the original shape as we use them to avoid
        # duplication in the permutation.
        arr_shape_indices = {index: dimlen for index, dimlen in enumerate(arr.shape)}

        # Find which axis in arr.shape corresponds to the specified size. Never returns duplicates.
        def find_axis(dimlen):
            nonlocal arr_shape_indices
            for index, d in arr_shape_indices.items():
                if d == dimlen:
                    del arr_shape_indices[index]
                    return index

        try:
            perm = [find_axis(dimlen) for dimlen in shape]
            arr = np.transpose(arr, perm)
        except Exception as err:
            G_LOGGER.extra_verbose("Skipping permutation due to {:}".format(err))
        else:
            if arr.shape != original_shape:
                G_LOGGER.info(
                    "Permuted array of shape: {:} to: {:} using permutation {:}".format(original_shape, arr.shape, perm)
                )
        return arr

    # Override any dynamic dimensions in the shape with concrete shapes from the array.
    def try_freeze_shape(arr, shape):
        if num_dynamic_dimensions(shape) == 1:
            try:
                static_dims = [dim for dim in shape if not is_dimension_dynamic(dim)]
                determined_dim = volume(arr.shape) // volume(static_dims)
            except ZeroDivisionError:
                determined_dim = 0
            shape = [determined_dim if is_dimension_dynamic(elem) else elem for elem in shape]
        elif is_rank_same(arr, shape):
            shape = [
                arr_shape_elem if is_dimension_dynamic(elem) else elem for elem, arr_shape_elem in zip(shape, arr.shape)
            ]
        return shape

    if shape == arr.shape:
        return arr

    if is_shape_dynamic(shape):
        shape = try_freeze_shape(arr, shape)

    if not is_rank_same(arr, shape):
        arr = try_reshape(arr, shape)

    if is_rank_same(arr, shape):
        arr = try_permute(arr, shape)

    arr = try_reshape(arr, shape)
    return arr


##
## Logging Utilities
##


@mod.export()
def str_from_layer(prefix, index, name, op, input_info, output_info):
    layer_str = "{:} {:<4} | {:} [Op: {:}]\n".format(prefix, index, name, op)
    layer_str += indent_block(input_info)

    layer_str += "\n" if (input_info and output_info) else ""
    indent_level = 1 if (input_info and output_info) else 0
    layer_str += (
        indent_block(" -> {:}".format(indent_block(output_info, level=indent_level).strip()), level=indent_level) + "\n"
    )
    return layer_str


@mod.export()
def indent_block(block, level=1):
    """
    Indents the provided block of text.

    Args:
        block (str): The text to indent.
        level (int): The number of tabs to indent with.

    Returns:
        str: The indented block.
    """
    tab = "\t" * level
    sep = "\n{:}".format(tab)
    return tab + sep.join(str(block).splitlines())


@mod.export()
def make_repr(type_str, *args, **kwargs):
    """
    Creates a string suitable for use with ``__repr__`` for a given
    type with the provided arguments.
    Skips keyword arguments that are set to ``None``.

    For example, ``make_repr("Example", None, "string", w=None, x=2)``
    would return a string: ``"Example(None, 'string', x=2)"``

    Args:
        type_str (str):
                The name of the type to create a representation for.

    Returns:
        Tuple[str, bool]:
                A tuple including the ``__repr__`` string and a boolean
                indicating whether all the arguments were default (i.e. None).
    """
    all_args = list(map(repr, args))

    for key, val in filter(lambda t: t[1] is not None, kwargs.items()):
        all_args.append("{:}={:}".format(key, repr(val)))

    repr_str = "{:}({:})".format(type_str, ", ".join(all_args))
    return repr_str, all(arg == repr(None) for arg in all_args)


##
## Safety
##


@mod.export()
class FreeOnException(object):
    def __init__(self, objs):
        """
        Frees the specified objects if an exception occurs in this context.
        Does nothing otherwise.

        Args:
            objs (List[object]): List of objects with __enter__/__exit__ methods defined.
        """
        assert is_sequence(objs), "FreeOnException requires a sequence of objects!"
        self.objs = objs

    def __enter__(self):
        """
        Returns the objects managed by this context manager.
        """
        return self.objs

    def __exit__(self, exc_type, exc_value, traceback):
        """
        On exception, deletes all tracked objects.
        Does nothing if there are no exceptions.
        """
        if exc_type is not None:
            # Objects are freed in reverse order
            with contextlib.ExitStack() as stack:
                for obj in self.objs:
                    if obj is not None:
                        stack.enter_context(obj)


##
## Attribute Helpers
##


@mod.export()
class TempAttrChange(object):
    """
    Temporarily set an instance member to a particular value for the duration
    of the context manager.
    """

    def __init__(self, arg_group, attr, value):
        self.arg_group = arg_group
        self.attr = attr

        self.old_value = getattr(arg_group, attr)
        self.new_value = value

    def __enter__(self):
        if self.new_value is not None:
            setattr(self.arg_group, self.attr, self.new_value)

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(self.arg_group, self.attr, self.old_value)


@mod.export()
def getattr_nested(obj, attr):
    for typ in attr.split("."):
        obj = getattr(obj, typ)
    return obj
