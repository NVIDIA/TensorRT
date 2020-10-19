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
from polygraphy.util.format import DataFormat, FormatManager
from polygraphy.logger import G_LOGGER
from polygraphy.common import constants

from collections import OrderedDict
import pickle
import zlib
import sys
import os

import numpy as np


NP_TYPE_FROM_STR = {np.dtype(dtype).name: np.dtype(dtype) for dtype in np.sctypeDict.values()}
STR_FROM_NP_TYPE = {dtype: name for name, dtype in NP_TYPE_FROM_STR.items()}


def version(version_str):
    return tuple([int(num) for num in version_str.split(".")])


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


def unique_list(sequence):
    """
    Creates a list without duplicate elements, preserving order.

    Args:
        sequence (Sequence): The sequence to make unique

    Returns:
        list: A list containing the same elements as sequence, in the same order, but without duplicates.
    """
    return list(OrderedDict.fromkeys(sequence))


# default_value exists to solve issues that might result from Python's normal default arguments.
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
#         self.value = default_value(value, [])
#
# Then we get the expected behavior:
#
# >>> x = MyClass()
# >>> x.value.append("SHOULD NOT BE IN Y")
# >>> y = MyClass()
# >>> y.value
# []
def default_value(value, default):
    """
    Returns a specified default value if the provided value is None.

    Args:
        value (object): The value.
        default (object): The default value to use if value is None.

    Returns:
        object: Either value, or the default.
    """
    return value if value is not None else default


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
    args += (None, ) * (num - len(args))
    return args[0:num]

##
## Shapes
##

def is_dimension_dynamic(dim):
    is_dim_str = not isinstance(dim, int)
    return dim is None or is_dim_str or dim < 0


def num_dynamic_dimensions(shape):
    return len([dim for dim in shape if is_dimension_dynamic(dim)])


def is_shape_dynamic(shape):
    return num_dynamic_dimensions(shape) > 0


def is_valid_shape_override(new_shape, original_shape):
    ranks_same = len(original_shape) == len(new_shape)
    overrides_valid = all([odim == ndim or is_dimension_dynamic(odim) for odim, ndim in zip(original_shape, new_shape)])
    return ranks_same and overrides_valid


def override_dynamic_shape(shape):
    return [constants.DEFAULT_SHAPE_VALUE if is_dimension_dynamic(elem) else elem for elem in shape]


def shapes_match(shape0, shape1):
    return len(shape0) == len(shape1) and all([s0 == s1 for s0, s1 in zip(shape0, shape1)])


def volume(obj):
    vol = 1
    for elem in obj:
        vol *= elem
    return vol


def is_empty_shape(shape):
    return volume(shape) == 0

##
## Compression and Serialization
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


def pickle_load(path):
    with open(path, "rb") as f:
        return pickle.loads(f.read())


def pickle_save(path, obj):
    with open(path, "wb") as f:
        return f.write(pickle.dumps(obj))


PIPE_MAX_SEND_BYTES = 1 << 31
"""The maximum number of bytes that can be sent at once over a queue"""


def send_on_queue(queue, obj):
    obj = pickle.dumps(obj)

    if sys.getsizeof(obj) > PIPE_MAX_SEND_BYTES:
        G_LOGGER.warning("Object size ({:} bytes) exceeds maximum size that can be sent over queues ({:} bytes). "
                         "Attempting to compress - this may take some time. If this does not work or you want to avoid "
                         "the compression overhead, you should disable subprocesses by omitting the --use-subprocess flag, "
                         "or by setting use_subprocess=False in Comparator.run().".format(sys.getsizeof(obj), PIPE_MAX_SEND_BYTES))
        obj = compress(obj)

    assert sys.getsizeof(obj) <= PIPE_MAX_SEND_BYTES

    G_LOGGER.ultra_verbose("Sending: {:} on queue".format(obj))
    queue.put(obj)


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
    obj = pickle.loads(obj)
    G_LOGGER.ultra_verbose("Received {:} on queue".format(obj))
    return obj


def try_receive_on_queue(queue, timeout=None):
    try:
        obj = receive_on_queue(queue, timeout)
        if obj is None:
            G_LOGGER.warning("Received {:} on the queue. This likely means that there was an error in sending "
                             "the object over the queue. You may want to run with use_subprocess=False in Comparator.run() "
                             "or omit the --use-subprocess flag to prevent further issues.".format(obj))
        return obj
    except Exception as err:
        G_LOGGER.warning("Could not receive on queue: {:}\nYou may want to run with use_subprocess=False in Comparator.run() "
                         "or omit the --use-subprocess flag to prevent further issues.".format(err))
        return None


def try_call(func, *args, **kwargs):
    """
    Attempts to invoke a function with arguments. If `func` is not callable, then returns `func`
    The second return value of this function indicates whether the argument was a callable.
    """
    if callable(func):
        ret = func(*args, **kwargs)
        return ret, True
    return func, False


##
## File creation
##

def insert_suffix(path, suffix):
    """
    Inserts the provided suffix into the given path, before any file extensions.

    Returns:
        str: The path, with suffix inserted, or None if no path was provided.
    """
    if path is None:
        return None
    path, ext = os.path.splitext(path)
    return "".join([path, suffix, ext])


def lazy_write(contents, path, mode="wb"):
    """
    Writes a file to the specified path.

    Args:
        contents (Callable() -> bytes):
                Either a bytes-like object that can be written to disk, or a callable which will return such an object.
        path (str): The path to write to.


        mode(str): The mode to use when writing. Defaults to "wb".

    Returns:
        str: The complete file path, or `None` if nothing was written.
    """
    if path is not None:
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            G_LOGGER.verbose("{:} does not exist, creating now.".format(dir_path))
            os.makedirs(dir_path, exist_ok=True)

        contents, _ = try_call(contents)

        with open(path, mode) as f:
            G_LOGGER.info("Writing to {:}".format(path))
            f.write(contents)
        return path
    return None


def try_match_shape(arr, shape):
    """
    Attempts to permute or reshape the array so its shape matches the specified shape.
    This is a no-op if the array is already the correct shape.

    Args:
        arr (np.ndarray): The array to reshape.
        shape (Tuple[int]): The shape to use. May contain at most 1 dynamic dimension.

    Returns:
        np.ndarray: The reshaped array.
    """
    def is_rank_same(arr, shape):
        return len(shape) == len(arr.shape)

    def try_reshape(arr, shape):
        try:
            arr = arr.reshape(shape)
            G_LOGGER.verbose("Reshaped array to shape: {:}".format(arr.shape))
        except ValueError:
            G_LOGGER.warning("Could not reshape array (shape: {:}) to {:}. Skipping reshape.".format(arr.shape, shape))
        return arr

    def try_permute(arr, shape):
        try:
            perm = FormatManager.permutation(FormatManager.determine_format(arr.shape), FormatManager.determine_format(shape))
            G_LOGGER.verbose("Permuting shape: {:} using permutation {:}".format(arr.shape, perm))
            arr = np.transpose(arr, perm)
        except Exception as err:
            # FormatManager may not recognize the format or be able generate the permutation for the format combination
            G_LOGGER.extra_verbose("Skipping permutation due to {:}".format(err))
        return arr

    # Override any dynamic dimensions in the shape with concrete shapes from the array.
    def try_fix_shape(arr, shape):
        if num_dynamic_dimensions(shape) == 1:
            try:
                static_dims = [dim for dim in shape if not is_dimension_dynamic(dim)]
                determined_dim = volume(arr.shape) // volume(static_dims)
            except ZeroDivisionError:
                determined_dim = 0
            shape = [determined_dim if is_dimension_dynamic(elem) else elem for elem in shape]
        elif is_rank_same(arr, shape):
            shape = [arr_shape_elem if is_dimension_dynamic(elem) else elem for elem, arr_shape_elem in zip(shape, arr.shape)]
        return shape

    if shape == arr.shape:
        return arr

    # When ranks are unequal, we try to squeeze first
    if not is_rank_same(arr, shape):
        shape = [elem for elem in shape if elem != 1]
        arr = np.squeeze(arr)

    if is_shape_dynamic(shape):
        shape = try_fix_shape(arr, shape)

    # If the rank is still not the same, do a reshape on the second
    if not is_rank_same(arr, shape):
        arr = try_reshape(arr, shape)

    # Next, permute if the ranks now match
    if is_rank_same(arr, shape):
        arr = try_permute(arr, shape)

    # Do a final reshape after the outputs have been permuted.
    arr = try_reshape(arr, shape)
    return arr


def str_from_module_info(module, name=None):
    name = default_value(name, "Loaded Module: {:<14}".format(module.__name__))
    paths = str(list(map(os.path.realpath, module.__path__)))
    return "{:} | Version: {:<8} | Path: {:}".format(name, str(module.__version__), paths)


def log_module_info(module, name=None, severity=G_LOGGER.VERBOSE):
    G_LOGGER.log(str_from_module_info(module, name), severity=severity)


def str_from_layer(prefix, index, name, op, input_info, output_info):
    layer_str = "{:} {:<4} | {:} [Op: {:}]\n".format(prefix, index, name, op)
    layer_str += "{tab}{:}".format(input_info, tab=constants.TAB)
    if input_info and output_info:
        layer_str += "\n" + constants.TAB
    else:
        layer_str += " "
    layer_str += "-> {:}\n".format(output_info)
    return layer_str


def indent_block(block, level=1):
    """
    Indents the provided block of text.

    Args:
        block (str): The text to indent.
        level (int): The number of tabs to indent with.

    Returns:
        str: The indented block.
    """
    tab = constants.TAB * level
    sep = "\n{:}".format(tab)
    return tab + sep.join(str(block).splitlines())
