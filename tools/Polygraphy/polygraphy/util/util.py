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
import contextlib
import copy
import glob
import math
import os
import sys
import tempfile
import zlib
from collections import OrderedDict

from polygraphy import constants, mod
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")

# These modules are not cross-platform so any usage should be guarded
fcntl = mod.lazy_import("fcntl")
msvcrt = mod.lazy_import("msvcrt")


@mod.export()
def is_nan(obj):
    return isinstance(obj, float) and math.isnan(obj)


@mod.export()
def is_inf(obj):
    return isinstance(obj, float) and math.isinf(obj)


@mod.export()
def find_str_in_iterable(name, seq, index=None):
    """
    Attempts to find matching strings in a sequence. Checks for exact matches, then
    case-insensitive substring matches, finally falling back to index based matching.

    Args:
        name (str): The key to search for.
        seq (Sequence[str]): The dictionary to search in.
        index (int): An index to fall back to if the string could not be found.

    Returns:
        str: The element found in the sequence, or None if it could not be found.
    """
    if name in seq:
        return name

    for elem in seq:
        if name.lower() in elem.lower() or elem.lower() in name.lower():
            return elem

    if index is not None and index < len(seq):
        return list(seq)[index]
    return None


@mod.export()
def check_sequence_contains(
    sequence, items, name=None, items_name=None, log_func=None, check_missing=None, check_extra=None
):
    """
    Checks that a sequence contains the provided items and also
    that it does not contain any extra items and issues warnings/errors
    otherwise.

    Args:
        sequence (Sequence[Any]):
                The sequence to check.
        items (Sequence[Any]):
                The items that should be in the sequence.

        name (str):
                The name to use for the sequence displaying warnings/errors.
                Defaults to "the sequence".
        items_name (str):
                The name to use for items in the sequence displaying warnings/errors.
                Defaults to "items".
        log_func (Logger.method):
                The logging method to use to display warnings/errors.
                Defaults to G_LOGGER.critical.
        check_missing (bool):
                Whether to check for missing items in the sequence.
                Defaults to True.
        check_extra (bool):
                Whether to check for extra items in the sequence.
                Defaults to True.

    Returns:
        Tuple[Sequence[Any], Sequence[Any]]:
                The missing and extra items respectively
    """
    check_missing = default(check_missing, True)
    check_extra = default(check_extra, True)
    log_func = default(log_func, G_LOGGER.critical)
    name = default(name, "the sequence")
    items_name = default(items_name, "items")

    sequence = set(sequence)
    items = set(items)

    missing = items - sequence
    if check_missing and missing:
        log_func(
            f"The following {items_name} were not found in {name}: {missing}.\n"
            f"Note: All {items_name} are: {items}, but {items_name} provided were: {sequence}"
        )

    extra = sequence - items
    if check_extra and extra:
        log_func(
            f"Extra {items_name} in {name}: {extra}.\n"
            f"Note: All {items_name} are: {items}, but {items_name} provided were: {sequence}"
        )

    return missing, extra


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
# class MyClass:
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
# class MyClass:
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
        value : The value.
        default : The default value to use if value is None.

    Returns:
        object: Either value, or the default.
    """
    return value if value is not None else default


@mod.export()
def is_sequence(obj):
    return (
        hasattr(obj, "__iter__") and not isinstance(obj, dict) and not isinstance(obj, set) and not isinstance(obj, str)
    )


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
class NamedTemporaryFile:
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
            return os.path.join(tempfile.gettempdir(), f"{prefix}{os.urandom(24).hex()}{suffix}")

        # In the unlikely event the path exists, generate a new one. Only try 100 times so
        # we don't end up in an infinite loop.
        path = rand_path()
        for _ in range(100):
            if not os.path.exists(path):
                break
            path = rand_path()
        else:
            G_LOGGER.critical(f"Could not create a temporary file under: {tempfile.gettempdir()}")

        self.name = path  # Use 'name' to be compatible with tempfile.NamedTemporaryFile
        open(self.name, "x").close()  # `touch` the file
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
        self._fhandle.flush()
        os.fsync(self._fhandle.fileno())
        self._fhandle.close()


@mod.export()
class LockFile:
    """
    Context manager that locks a file for exclusive access.
    Has no effect for file-like objects.
    """

    def __init__(self, path):
        """
        Args:
            path (str): The path to the file.
        """
        self.is_file_like = is_file_like(path)
        if not self.is_file_like:
            self.lock_path = path + ".lock"
            self._fhandle = None

    def __enter__(self):
        """
        Locks the file by creating a temporary `.lock` file and acquiring exclusive access to it.

        Returns:
            file-like: The open file object.
        """
        if self.is_file_like:
            return

        self._fhandle = open(self.lock_path, "wb+")
        if sys.platform.startswith("win"):
            # On Windows, msvcrt.locking() raises an OSError if the file cannot be locked after 10 attempts.
            # To compensate, keep trying until we finally get the lock.
            locked = False
            while not locked:
                try:
                    msvcrt.locking(self._fhandle.fileno(), msvcrt.LK_RLCK, get_file_size(self._fhandle))
                except OSError:
                    locked = False
                else:
                    locked = True
        else:
            fcntl.lockf(self._fhandle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Unlocks and closes the lock file.
        """
        if self.is_file_like:
            return

        if sys.platform.startswith("win"):
            msvcrt.locking(self._fhandle.fileno(), msvcrt.LK_UNLCK, get_file_size(self._fhandle))
        else:
            fcntl.lockf(self._fhandle.fileno(), fcntl.LOCK_UN)

        # The lock file should not be deleted here since other processes might create new handles
        # which therefore don't block correctly if there are already processes holding the old handle.
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
            f"File-like object has a different mode than requested!\nNote: Requested mode was: {mode} but file-like object has mode: {file_like.mode}"
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
def add_file_suffix(path: str, suffix: str):
    """
    Adds a suffix to a path or filename, before the file extension.

    Args:
        path (str): The path or filename.
        suffix (str): The suffix.

    Returns:
        str: The path or filename with the suffix attached.
    """
    path, ext = os.path.splitext(path)
    return f"{path}{suffix}{ext}"


@mod.export()
def makedirs(path):
    dir_path = os.path.dirname(path)
    if dir_path:
        dir_path = os.path.realpath(dir_path)
        if not os.path.exists(dir_path):
            G_LOGGER.verbose(f"{dir_path} does not exist, creating now.")
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
        G_LOGGER.info(f"Loading {description} from {src}")

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
        G_LOGGER.info(f"Saving {description} to {dest}")

    if is_file_like(dest):
        warn_if_wrong_mode(dest, mode)
        bytes_written = dest.write(contents)
        dest.flush()
        os.fsync(dest.fileno())
        try:
            content_bytes = len(contents.encode())
        except:
            pass
        else:
            if bytes_written != content_bytes:
                G_LOGGER.warning(
                    f"Could not write entire file. Note: file contains {content_bytes} bytes, but only {bytes_written} bytes were written"
                )
    else:
        makedirs(dest)
        with open(dest, mode) as f:
            f.write(contents)
    return dest


##
## Compression
##


class Compressed:
    """
    Represents an object compressed by zlib
    """

    def __init__(self, cobj):
        self.bytes = cobj


def is_compressed(obj):
    return isinstance(obj, Compressed)


def compress(obj):
    G_LOGGER.verbose(f"Compressing {type(obj)} object")
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
            f"Object size ({sys.getsizeof(obj)} bytes) exceeds maximum size that can be sent over queues ({PIPE_MAX_SEND_BYTES} bytes). Attempting to compress - this may take some time. If this does not work or you want to avoid the compression overhead, you should disable subprocesses by omitting the --use-subprocess flag, or by setting use_subprocess=False in Comparator.run()."
        )
        obj = compress(obj)

    assert sys.getsizeof(obj) <= PIPE_MAX_SEND_BYTES

    G_LOGGER.ultra_verbose(f"Sending: {obj} on queue")
    queue.put(obj)


@mod.export()
def try_send_on_queue(queue, obj):
    """
    Attempts to send an object over the queue, compressing it if needed.
    In the event the object cannot be sent, sends `None` instead.

    Args:
        queue (queue.Queue): The queue to send the object over.
        obj : The object to send.
    """
    try:
        send_on_queue(queue, obj)
    except Exception as err:
        G_LOGGER.warning(f"Could not send object on queue: {err}\nSending None instead.")
        queue.put(None)


def receive_on_queue(queue, timeout=None):
    G_LOGGER.extra_verbose("Waiting for data to become available on queue")
    obj = queue.get(block=True, timeout=timeout)
    if is_compressed(obj):
        obj = decompress(obj)
    G_LOGGER.ultra_verbose(f"Received {obj} on queue")
    return obj


@mod.export()
def try_receive_on_queue(queue, timeout=None):
    try:
        obj = receive_on_queue(queue, timeout)
        if obj is None:
            G_LOGGER.warning(
                f"Received {obj} on the queue. This likely means that there was an error in sending the object over the queue. You may want to run with use_subprocess=False in Comparator.run() or omit the --use-subprocess flag to prevent further issues."
            )
        return obj
    except Exception as err:
        G_LOGGER.warning(
            f"Could not receive on queue: {err}\nYou may want to run with use_subprocess=False in Comparator.run() or omit the --use-subprocess flag to prevent further issues."
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
            G_LOGGER.warning(f"Could not reshape array from shape: {arr.shape} to {shape}. Skipping reshape.")
        else:
            if arr.shape != original_shape:
                G_LOGGER.info(f"Reshaped array from shape: {original_shape} to: {arr.shape}")
        return arr

    def try_permute(arr, shape):
        original_shape = arr.shape

        if sorted(arr.shape) != sorted(shape):
            G_LOGGER.extra_verbose(f"Array of shape: {arr.shape} cannot be permuted to: {shape}")
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
            G_LOGGER.extra_verbose(f"Skipping permutation due to {err}")
        else:
            if arr.shape != original_shape:
                G_LOGGER.info(f"Permuted array of shape: {original_shape} to: {arr.shape} using permutation {perm}")
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


@mod.export()
def is_contiguous(array):
    """
    Checks whether the provided NumPy array is contiguous in memory.

    Args:
        array (np.ndarray): The NumPy array.

    Returns:
        bool: Whether the array is contiguous in memory.
    """
    return array.flags["C_CONTIGUOUS"]


@mod.export()
def make_contiguous(array):
    """
    Makes a NumPy array contiguous if it's not already.

    Args:
        array (np.ndarray): The NumPy array.

    Returns:
        np.ndarray: The contiguous NumPy array.
    """
    if not is_contiguous(array):
        return np.ascontiguousarray(array)
    return array


@mod.export()
def resize_buffer(buffer, shape):
    """
    Resizes the provided buffer and makes it contiguous in memory,
    possibly reallocating the buffer.

    Args:
        buffer (np.ndarray): The buffer to resize.
        shape (Sequence[int]): The desired shape of the buffer.

    Returns:
        np.ndarray: The resized buffer, possibly reallocated.
    """
    if shape != buffer.shape:
        try:
            buffer.resize(shape, refcheck=False)
        except ValueError as err:
            G_LOGGER.warning(
                f"Could not resize host buffer to shape: {shape}. "
                f"Allocating a new buffer instead.\nNote: Error was: {err}"
            )
            buffer = np.empty(shape, dtype=np.dtype(buffer.dtype))
    return make_contiguous(buffer)


##
## Logging Utilities
##


@mod.export()
def str_from_layer(prefix, index, name, op, input_names, input_meta, output_names, output_meta):
    def tensor_names_to_string(tensor_names, meta):
        sep = ",\n "
        elems = [f"{name} {meta[name]}".strip() for name in tensor_names]
        return "{" + sep.join(elems) + "}"

    layer_str = f"{prefix} {index:<4} | {name} [Op: {op}]\n"
    layer_str += indent_block(tensor_names_to_string(input_names, input_meta))

    layer_str += "\n" if (input_names and output_names) else ""
    indent_level = 1 if (input_names and output_names) else 0
    layer_str += (
        indent_block(
            f" -> {indent_block(tensor_names_to_string(output_names, output_meta), level=indent_level).strip()}",
            level=indent_level,
        )
        + "\n"
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
    tab = f"{constants.TAB}" * level
    sep = f"\n{tab}"
    return tab + sep.join(str(block).splitlines())


# Some objects don't have correct `repr` implementations, so we need to handle them specially.
# For other objects, we do nothing.
def handle_special_repr(obj):
    # 1. Work around incorrect `repr` implementations

    # Use a special __repr__ override so that we can inline strings
    class InlineString(str):
        def __repr__(self) -> str:
            return self

    if is_nan(obj) or is_inf(obj):
        return InlineString(f"float('{obj}')")

    # 2. If this object is a collection, recursively apply this logic.
    # Note that we only handle the built-in collections here, since custom collections
    # may have special behavior that we don't know about.

    if type(obj) not in [tuple, list, dict, set]:
        return obj

    obj = copy.copy(obj)
    # Tuple needs special handling since it doesn't support assignment.
    if type(obj) is tuple:
        args = tuple(handle_special_repr(elem) for elem in obj)
        obj = type(obj)(args)
    elif type(obj) is list:
        for index, elem in enumerate(obj):
            obj[index] = handle_special_repr(elem)
    elif type(obj) is dict:
        new_items = {}
        for key, value in obj.items():
            new_items[handle_special_repr(key)] = handle_special_repr(value)
        obj.clear()
        obj.update(new_items)
    elif type(obj) is set:
        new_elems = set()
        for value in obj:
            new_elems.add(handle_special_repr(value))
        obj.clear()
        obj.update(new_elems)

    # 3. Finally, return the modified version of the object
    return obj


def apply_repr(obj):
    obj = handle_special_repr(obj)
    return repr(obj)


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
        Tuple[str, bool, bool]:
                A tuple including the ``__repr__`` string and two booleans
                indicating whether all the positional and keyword arguments were default
                (i.e. None) respectively.
    """
    processed_args = list(map(apply_repr, args))

    processed_kwargs = []
    for key, val in filter(lambda t: t[1] is not None, kwargs.items()):
        processed_kwargs.append(f"{key}={apply_repr(val)}")

    repr_str = f"{type_str}({', '.join(processed_args + processed_kwargs)})"

    def all_default(arg_list):
        return all(arg == apply_repr(None) for arg in arg_list)

    return repr_str, all_default(processed_args), all_default(processed_kwargs)


##
## Safety
##


@mod.export()
class FreeOnException:
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
class TempAttrChange:
    """
    Temporarily set attributes to a particular value for the duration
    of the context manager.
    """

    def __init__(self, arg_group, attr_values):
        self.arg_group = arg_group
        self.old_values = {}
        self.new_values = attr_values

    def __enter__(self):
        for attr, new_value in self.new_values.items():
            if new_value is not None:
                self.old_values[attr] = getattr(self.arg_group, attr)
                setattr(self.arg_group, attr, new_value)

    def __exit__(self, exc_type, exc_value, traceback):
        for attr, old_value in self.old_values.items():
            setattr(self.arg_group, attr, old_value)


@mod.export()
def getattr_nested(obj, attr):
    for typ in attr.split("."):
        obj = getattr(obj, typ)
    return obj
