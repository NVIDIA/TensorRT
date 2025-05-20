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
"""
This file includes utility functions for arrays/tensors that work for multiple
libraries like NumPy and PyTorch.
"""

import builtins
import functools
import math
import numbers

from polygraphy import mod
from polygraphy.datatype import DataType
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")
torch = mod.lazy_import("torch>=1.13.0")


@mod.export()
def is_torch(obj):
    """
    Whether the provided object is a PyTorch tensor.
    This function does *not* introduce a dependency on the PyTorch module.

    Args:
        obj (Any): The object to check.

    Returns:
        bool: Whether the object is a PyTorch tensor.
    """
    return (
        torch.is_installed() and torch.is_importable() and isinstance(obj, torch.Tensor)
    )


@mod.export()
def is_numpy(obj):
    """
    Whether the provided object is a NumPy array or scalar.
    This function does *not* introduce a dependency on the NumPy module.

    Args:
        obj (Any): The object to check.

    Returns:
        bool: Whether the object is a NumPy array.
    """
    return (
        np.is_installed()
        and np.is_importable()
        and (isinstance(obj, np.ndarray) or isinstance(obj, np.generic))
    )


@mod.export()
def is_device_view(obj):
    """
    Whether the provided object is a DeviceView array.

    Args:
        obj (Any): The object to check.

    Returns:
        bool: Whether the object is a DeviceView.
    """
    from polygraphy.cuda import DeviceView

    return isinstance(obj, DeviceView)


# The current design dispatches to the correct function implementation separately for each function call.
# Obviously, this has some performance cost and an alternative approach would be a more familiar inheritance
# pattern wherein we would have a BaseArray class and then child classes like NumpyArray, TorchArray, PolygraphyDeviceArray etc.
# That way, the dispatching logic would only have to run once when we construct an instance of one of these
# classes.
#
# The tradeoff is that the caller would then have to be careful that they are *not* passing in NumPy arrays,
# Torch tensors etc. directly, but have first wrapped them appropriately. Plus, at the interface boundaries,
# we would have to unwrap them once again since we don't want to expose the wrappers at the API level (the user
# should be able to work directly with NumPy arrays, PyTorch tensors etc.).
#
# To illustrate this a bit better, consider the two possible workflows:
#
# Option 1 (dispatch logic in each function, current design):
#
# def my_api_func(obj)
#     nbytes = util.array.nbytes(obj) # Dispatching logic needs to run on each function call
#     dtype = util.array.dtype(obj)
#     # Do something interesting, then...
#     return obj
#
# Option 2 (class hierarchy, possible alternative design):
#
# # Assume we have:
#
# class BaseArray:
#     ...
#
# class TorchArray:
#     ...
#
# # etc.
#
# def my_api_func()
#     obj = wrap_array(obj) # Dispatch logic only runs once
#     nbytes = obj.nbytes
#     dtype = obj.dtype
#     # Do something interesting, then...
#     return unwrap_array(obj) # Need to return the np.ndarray/torch.Tensor/DeviceView, *not* the wrapper
#
# In Polygraphy, the number of calls to `wrap_array`/`unwrap_array` would most likely be quite high
# relative to the number of calls to the actual methods, so the perfomance hit of the current implementation
# may not be that significant. If it is, then it should be straightforward, though time-consuming, to switch to Option 2.
#
def dispatch(num_arrays=1):
    """
    Decorator that will dispatch to functions specific to a framework type, like NumPy or PyTorch,
    based on the type of the input.

    The decorated function should return a dictionary with implementations for all supported types.
    The following keys may be specified: ["torch", "numpy", "device_view", "number"].

    Args:
        num_arrays (int):
            The number of arrays expected.
            The naming convention for the array arguments is as follows:
            - For a single array, the argument is called "obj".
            - For two arrays, the arguments are called "lhs" and "rhs".
            - For N>2 arrays, the arguments are called "obj0", "obj1", ... "obj<N-1>"
            In the case of more than one array, this function will automatically convert the rest to be of the
            same kind as the first.
    """

    def dispatch_impl(func):
        def _get_key(obj):
            key = None

            if is_device_view(obj):
                key = "device_view"
            elif is_numpy(obj):
                key = "numpy"
            elif is_torch(obj):
                key = "torch"
            elif isinstance(obj, numbers.Number):
                key = "number"

            if not key:
                G_LOGGER.critical(
                    f"Function: {func.__name__} is unsupported for objects of type: {type(obj).__name__}"
                )
            return key

        if num_arrays < 0:
            G_LOGGER.critical(
                f"Function: {func.__name__} is unsupported with {num_arrays} < 0"
            )

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if len(args) < num_arrays:
                G_LOGGER.critical(
                    f"Function: {func.__name__} is unsupported for less than {num_arrays} positional arguments"
                )

            mapping = func()
            obj0 = args[0]
            key = _get_key(obj0)

            if key not in mapping:
                G_LOGGER.critical(
                    f"Function: {func.__name__} is unsupported for objects of type: {type(obj0).__name__}"
                )

            # Note that we can use to_torch/to_numpy here without a circular dependency because those functions
            # take the num_arrays=1 path.
            def convert_array(obj):
                if key == "torch":
                    return to_torch(obj)
                elif key == "numpy":
                    return to_numpy(obj)
                else:
                    G_LOGGER.critical(
                        f"Function: {func.__name__} is unsupported for objects of type: {type(obj).__name__}"
                    )

            converted_args = (
                [obj0]
                + list(map(convert_array, args[1:num_arrays]))
                + list(args[num_arrays:])
            )

            return mapping[key](*converted_args, **kwargs)

        return wrapped

    return dispatch_impl


##
## Conversion Functions
##


@mod.export()
@dispatch()
def to_torch():
    """
    Converts an array or tensor to a PyTorch tensor.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        torch.Tensor: The PyTorch tensor.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "torch": lambda obj: obj,
        "numpy": lambda obj: torch.from_numpy(obj),
        "number": lambda obj: torch.tensor(obj),
    }


@mod.export()
@dispatch()
def to_numpy():
    """
    Converts an array or tensor to a NumPy array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        np.ndarray: The NumPy array.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "torch": lambda obj: obj.numpy(force=True),
        "numpy": lambda obj: obj,
        "number": lambda obj: np.array(obj),
    }


##
## Metadata
##


@mod.export()
@dispatch()
def nbytes():
    """
    Calculate the number of bytes required by the input array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]): The array or tensor.

    Returns:
        int: The number of bytes required by the array.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "torch": lambda obj: obj.nelement() * obj.element_size(),
        "numpy": lambda obj: obj.nbytes,
        "device_view": lambda obj: obj.nbytes,
    }


@mod.export()
@dispatch()
def size():
    """
    Calculate the volume of the input array

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]): The array or tensor.

    Returns:
        int: The volume of the array.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "torch": lambda obj: obj.numel(),
        "numpy": lambda obj: obj.size,
    }


@mod.export()
@dispatch()
def data_ptr():
    """
    Return a pointer to the first element of the input array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]): The array or tensor.

    Returns:
        int: A pointer to the first element of the array.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "torch": lambda obj: obj.data_ptr(),
        "numpy": lambda obj: obj.ctypes.data,
        "device_view": lambda obj: obj.ptr,
    }


@mod.export()
@dispatch()
def is_on_cpu():
    """
    Returns whether the input array is in CPU memory.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]): The array or tensor.

    Returns:
        bool: Whether the array is in CPU, i.e. host, memory.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "torch": lambda obj: obj.device.type == "cpu",
        "numpy": lambda _: True,
        "device_view": lambda _: False,
    }


@mod.export()
@dispatch()
def is_on_gpu():
    """
    Returns whether the input array is in GPU memory.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]): The array or tensor.

    Returns:
        bool: Whether the array is in GPU, i.e. host, memory.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "torch": lambda obj: obj.device.type == "cuda",
        "numpy": lambda _: False,
        "device_view": lambda _: True,
    }


@mod.export()
@dispatch()
def dtype():
    """
    Return the data type the input array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]): The array or tensor.

    Returns:
        DataType: The data type of the array

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    func = lambda obj: DataType.from_dtype(obj.dtype)
    return {"torch": func, "numpy": func, "device_view": func}


@mod.export()
@dispatch()
def shape():
    """
    Return the shape the input array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]): The array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray, DeviceView]: The shape of the array

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    func = lambda obj: obj.shape
    return {"torch": func, "numpy": func, "device_view": func}


@mod.export()
def view(obj, dtype, shape):
    """
    Return a view of the the input array with the given data type and shape.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]):
                The array or tensor. Must be contiguous.
        dtype (DataType): The data type to use for the view.
        shape (Sequence[int]): The shape to use for the view.

    Returns:
        Union[torch.Tensor, numpy.ndarray, DeviceView]: The view of the array

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    if not is_contiguous(obj):
        G_LOGGER.critical(f"Input array to view() must be contiguous in memory")

    if is_device_view(obj):
        return obj.view(shape=shape, dtype=dtype)

    dtype = (
        DataType.to_dtype(dtype, "numpy")
        if is_numpy(obj)
        else DataType.to_dtype(dtype, "torch")
    )
    return obj.reshape(-1).view(dtype).reshape(shape)


@mod.export()
@dispatch()
def is_contiguous():
    """
    Checks whether the provided array is contiguous in memory.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]): The array or tensor.

    Returns:
        bool: Whether the array is contiguous in memory.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "torch": lambda obj: obj.is_contiguous(),
        "numpy": lambda obj: obj.flags["C_CONTIGUOUS"],
        "device_view": lambda _: True,
    }


##
## Memory Management
##


@mod.export()
@dispatch()
def make_contiguous():
    """
    Makes an array contiguous if it's not already.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceView]): The array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray, DeviceView]: The contiguous array.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def impl_numpy(obj):
        if is_contiguous(obj):
            return obj
        return np.ascontiguousarray(obj)

    return {
        "torch": lambda obj: obj.contiguous(),
        "numpy": impl_numpy,
        "device_view": lambda obj: obj,
    }


@mod.export()
@dispatch()
def resize_or_reallocate():
    """
    Resizes the provided buffer, possibly reallocating the buffer.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray, DeviceArray]): The array or tensor.
        shape (Sequence[int]): The desired shape of the buffer.

    Returns:
        Union[torch.Tensor, numpy.ndarray, DeviceArray]: The resized buffer, possibly reallocated.
    """

    def numpy_impl(obj, shape):
        if shape != obj.shape:
            try:
                obj.resize(shape, refcheck=False)
            except ValueError as err:
                G_LOGGER.warning(
                    f"Could not resize NumPy array to shape: {shape}. "
                    f"Allocating a new array instead.\nNote: Error was: {err}"
                )
                obj = np.empty(shape, dtype=np.dtype(obj.dtype))
        return obj

    return {
        "numpy": numpy_impl,
        "torch": lambda obj, shape: obj.resize_(shape) if shape != obj.shape else obj,
        "device_view": lambda obj, shape: (
            obj.resize(shape) if shape != obj.shape else obj
        ),
    }


##
## Math Helpers
##


@mod.export()
@dispatch()
def cast():
    """
    Casts an array to the specified type.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.
        dtype (DataType): The type to cast to.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The casted array.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj, dtype: np.array(obj.astype(dtype.numpy())),
        "torch": lambda obj, dtype: obj.to(DataType.to_dtype(dtype, "torch")),
    }


@mod.export()
@dispatch()
def any():
    """
    Return whether any of the values in the provided array evaluate to True.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        bool: Whether any of the values in the array evaluate to True.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.any(obj),
        "torch": lambda obj: bool(torch.any(obj)),
    }


@mod.export()
@dispatch()
def all():
    """
    Return whether all of the values in the provided array evaluate to True.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        bool: Whether all of the values in the array evaluate to True.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.all(obj),
        "torch": lambda obj: bool(torch.all(obj)),
    }


@mod.export()
@dispatch(num_arrays=2)
def equal():
    """
    Returns whether two arrays are equal

    Args:
        lhs (Union[torch.Tensor, numpy.ndarray]):
                The first array or tensor.
        rhs (Union[torch.Tensor, numpy.ndarray]):
                The second array or tensor.

    Returns:
        bool: Whether the arrays are equal.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    return {
        "torch": lambda lhs, rhs: torch.equal(lhs, rhs),
        "numpy": lambda lhs, rhs: np.array_equal(lhs, rhs),
    }


@mod.export()
@dispatch(num_arrays=2)
def subtract():
    """
    Subtracts the second array from the first.

    Args:
        lhs (Union[torch.Tensor, numpy.ndarray]):
                The first array or tensor.
        rhs (Union[torch.Tensor, numpy.ndarray]):
                The second array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The difference.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    return {
        "torch": lambda lhs, rhs: lhs - rhs,
        "numpy": lambda lhs, rhs: np.array(lhs - rhs),
    }


@mod.export()
@dispatch(num_arrays=2)
def divide():
    """
    Divides the first array by the second.

    Args:
        lhs (Union[torch.Tensor, numpy.ndarray]):
                The first array or tensor.
        rhs (Union[torch.Tensor, numpy.ndarray]):
                The second array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The quotient.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    return {
        "torch": lambda lhs, rhs: lhs / rhs,
        "numpy": lambda lhs, rhs: lhs / rhs,
    }


@mod.export()
@dispatch(num_arrays=2)
def allclose():
    """
    Returns whether all the values in two arrays are within the given thresholds.

    Args:
        lhs (Union[torch.Tensor, numpy.ndarray]):
                The first array or tensor.
        rhs (Union[torch.Tensor, numpy.ndarray]):
                The second array or tensor.
        rtol (float): The relative tolerance. Defaults to 1e-5.
        atol (float): The absolute tolerance. Defaults to 1e-8.

    Returns:
        bool: Whether the arrays are close.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    DEFAULT_RTOL = 1e-5
    DEFAULT_ATOL = 1e-8

    return {
        "torch": lambda lhs, rhs, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL: torch.allclose(
            lhs, rhs, rtol=rtol, atol=atol
        ),
        "numpy": lambda lhs, rhs, rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL: np.allclose(
            lhs, rhs, rtol=rtol, atol=atol
        ),
    }


@mod.export()
def unravel_index(index, shape):
    """
    Unravels a flat index into a N-dimensional index based on the specified shape.

    Args:
        index (int): The flat index.
        shape (Sequence[int]): The shape on which to unravel the index.

    Returns:
        Tuple[int]: The N-dimensional index.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    index = int(index)

    nd_index = []
    for dim in reversed(shape):
        nd_index.insert(0, index % dim)
        index = index // dim

    return tuple(nd_index)


@mod.export()
@dispatch()
def histogram():
    """
    Compute a histogram for the given array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.
        range (Tuple[float, float]): The lower and upper range of the bins.

    Returns:
        Tuple[Union[torch.Tensor, numpy.ndarray], Union[torch.Tensor, numpy.ndarray]]:
            The histogram values and the bin edges

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def torch_impl(obj, range=None):
        # PyTorch doesn't support histograms for all types, so cast to FP32
        original_dtype = obj.dtype
        hist, bins = torch.histogram(obj.to(torch.float32), bins=10, range=range)
        return hist.to(original_dtype), bins.to(original_dtype)

    return {
        "numpy": lambda obj, range=None: np.histogram(obj, bins=10, range=range),
        "torch": torch_impl,
    }


@mod.export()
@dispatch()
def max():
    """
    Returns the maximum value of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Any: The maximum value

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.amax(obj).item(),
        "torch": lambda obj: torch.max(obj).item(),
    }


@mod.export()
@dispatch()
def argmax():
    """
    Returns the flattened index of the maximum value of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        int: The flattened index.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def torch_impl(obj):
        # Torch argmax doesn't support bools
        return torch.argmax(obj.to(torch.float32))

    return {
        "numpy": lambda obj: np.argmax(obj),
        "torch": lambda obj: torch_impl(obj),
    }


@mod.export()
@dispatch()
def min():
    """
    Returns the minimum value of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Any: The minimum value

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.amin(obj).item(),
        "torch": lambda obj: torch.min(obj).item(),
    }


@mod.export()
@dispatch()
def argmin():
    """
    Returns the flattened index of the minimum value of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        int: The flattened index.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def torch_impl(obj):
        # Torch argmin doesn't support bools
        return torch.argmin(obj.to(torch.float32))

    return {
        "numpy": lambda obj: np.argmin(obj),
        "torch": lambda obj: torch_impl(obj),
    }


@mod.export()
@dispatch()
def mean():
    """
    Returns the mean value of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.
        dtype (DataType): The mean compute type.

    Returns:
        Any: The mean value

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj, dtype=None: np.mean(
            obj, dtype=DataType.to_dtype(dtype, "numpy") if dtype is not None else None
        ),
        "torch": lambda obj, dtype=None: torch.mean(
            obj, dtype=DataType.to_dtype(dtype, "torch") if dtype is not None else None
        ),
    }


@mod.export()
@dispatch()
def std():
    """
    Returns the standard deviation of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Any: The standard deviation

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def torch_impl(obj):
        # torch.var is only supported for floats, so cast up and then back.
        obj_fp32 = obj.to(torch.float32)
        try:
            return torch.std(obj_fp32, correction=0)
        except AttributeError:
            return torch.std(obj_fp32, unbiased=False)

    return {
        "numpy": lambda obj: np.std(obj),
        "torch": torch_impl,
    }


@mod.export()
@dispatch()
def var():
    """
    Returns the variance of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Any: The variance

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def torch_impl(obj):
        # torch.var is only supported for floats, so cast up and then back.
        obj_fp32 = obj.to(torch.float32)
        try:
            return torch.var(obj_fp32, correction=0)
        except AttributeError:
            return torch.var(obj_fp32, unbiased=False)

    return {
        "numpy": lambda obj: np.var(obj),
        "torch": torch_impl,
    }


@mod.export()
@dispatch()
def median():
    """
    Returns the median value of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Any: The median value

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def torch_impl(obj):
        # Median in PyTorch doesn't work as expected for arrays with an even number of elements - instead
        # of returning the average of the two middle elements, it just returns the smaller one.
        # It is also not implemented for some types, so cast to FP32 for compute.

        original_dtype = obj.dtype
        obj = obj.to(torch.float32)

        rv = 0
        if obj.nelement() % 2 == 1:
            rv = torch.median(obj)
        else:
            smaller = torch.median(obj)
            larger = torch.median(torch.cat([obj.flatten(), torch.max(obj)[None]]))
            rv = (smaller + larger) / 2.0
        return rv.to(original_dtype)

    return {
        "numpy": lambda obj: np.median(obj),
        "torch": torch_impl,
    }


@mod.export()
@dispatch()
def quantile():
    """
    Returns the value of the q quantile of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.
        q  (float): Quantile to compute, expected range [0, 1]

    Returns:
        Any: The quantile value

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def numpy_impl(obj, q):
        if obj.size == 0:
            return np.inf
        return np.quantile(obj, q)

    def torch_impl(obj, q):
        if obj.numel() == 0:
            return torch.inf
        original_dtype = obj.dtype
        obj = obj.to(torch.float32)
        qunatile_val = torch.quantile(obj, q)
        return qunatile_val.to(original_dtype)

    return {
        "numpy": numpy_impl,
        "torch": torch_impl,
    }


@mod.export()
@dispatch()
def topk():
    """
    Returns a tuple of the top k values and indices of an array along a specified axis.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.
        k (int): The number of values to return. This is clamped to the length of obj along the given axis.
        axis (int): The axis to perform the topk computation on

    Returns:
        Tuple[Union[torch.Tensor, numpy.ndarray], Union[torch.Tensor, numpy.ndarray]]: A tuple containing a pair of arrays,
            the first being the values and the second being the indices of the top k values along the specified axis

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def numpy_impl(obj, k, axis):
        # NumPy doesn't have a Top K implementation
        indices = np.argsort(-obj, axis=axis, kind="stable")
        axis_len = indices.shape[axis]
        indices = np.take(indices, np.arange(0, builtins.min(k, axis_len)), axis=axis)
        return np.take_along_axis(obj, indices, axis=axis), indices

    def torch_impl(obj, k, axis):
        axis_len = obj.shape[axis]

        # Top K has no implementation for float16 in torch-cpu, so
        # If gpu is available, run computation there
        # Otherwise, run the calculation on cpu using fp32 precision
        if obj.dtype == torch.float16:
            if torch.cuda.is_available():
                original_device = obj.device
                ret = tuple(
                    torch.topk(obj.to("cuda"), builtins.min(k, axis_len), dim=axis)
                )
                return (ret[0].to(original_device), ret[1].to(original_device))
            else:
                ret = tuple(
                    torch.topk(
                        obj.type(torch.float32), builtins.min(k, axis_len), dim=axis
                    )
                )
                return (ret[0].type(torch.float16), ret[1].type(torch.float16))
        return tuple(torch.topk(obj, builtins.min(k, axis_len), dim=axis))

    return {
        "numpy": numpy_impl,
        "torch": torch_impl,
    }


@mod.export()
@dispatch()
def abs():
    """
    Returns the absolute value of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Any: The absolute value

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """

    def torch_abs_impl(obj):
        # PyTorch doesn't support abs for all types, so cast to FP32
        original_dtype = obj.dtype
        return torch.abs(obj.to(torch.float32)).to(original_dtype)

    return {
        "numpy": lambda obj: np.array(np.abs(obj)),
        "torch": lambda obj: torch_abs_impl(obj),
    }


@mod.export()
@dispatch()
def isfinite():
    """
    Returns a boolean array indicating if each element of obj is finite or not.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The boolean array indicating which elements of obj are finite.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.isfinite(obj),
        "torch": lambda obj: torch.isfinite(obj),
    }


@mod.export()
@dispatch()
def isinf():
    """
    Returns a boolean array indicating if each element of obj is infinite or not.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The boolean array indicating which elements of obj are infinite.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.isinf(obj),
        "torch": lambda obj: torch.isinf(obj),
    }


@mod.export()
@dispatch()
def isnan():
    """
    Returns a boolean array indicating if each element of obj is NaN or not.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The boolean array indicating which elements of obj are NaN.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.isnan(obj),
        "torch": lambda obj: torch.isnan(obj),
        "number": lambda obj: math.isnan(obj),
    }


@mod.export()
@dispatch()
def argwhere():
    """
    Returns a indices of non-zero array elements

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: An (N, obj.ndim) array containing indices of non-zero elements of obj

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.argwhere(obj),
        "torch": lambda obj: torch.argwhere(obj),
    }


@mod.export()
@dispatch()
def ravel():
    """
    Flattens the input array

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The flattened input tensor

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.ravel(obj),
        "torch": lambda obj: torch.ravel(obj),
    }


@mod.export()
@dispatch()
def logical_not():
    """
    Computes the logical not of an array

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]):
                The input array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The logical not.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.logical_not(obj),
        "torch": lambda obj: torch.logical_not(obj),
    }


@mod.export()
@dispatch(num_arrays=2)
def logical_xor():
    """
    Computes the logical exclusive-or of two arrays.

    Args:
        lhs (Union[torch.Tensor, numpy.ndarray]):
                The first array or tensor.
        rhs (Union[torch.Tensor, numpy.ndarray]):
                The second array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The logical xor.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda lhs, rhs: np.logical_xor(lhs, rhs),
        "torch": lambda lhs, rhs: torch.logical_xor(lhs, rhs),
    }


@mod.export()
@dispatch(num_arrays=2)
def logical_and():
    """
    Computes the logical and of two arrays.

    Args:
        lhs (Union[torch.Tensor, numpy.ndarray]):
                The first array or tensor.
        rhs (Union[torch.Tensor, numpy.ndarray]):
                The second array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The logical and.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda lhs, rhs: np.logical_and(lhs, rhs),
        "torch": lambda lhs, rhs: torch.logical_and(lhs, rhs),
    }


@mod.export()
@dispatch(num_arrays=2)
def greater():
    """
    Returns a boolean array indicating where lhs is greater than rhs

    Args:
        lhs (Union[torch.Tensor, numpy.ndarray]):
                The first array or tensor.
        rhs (Union[torch.Tensor, numpy.ndarray]):
                The second array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: Boolean array indicating whether lhs > rhs.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda lhs, rhs: np.greater(lhs, rhs),
        "torch": lambda lhs, rhs: torch.gt(lhs, rhs),
    }


@mod.export()
@dispatch(num_arrays=3)
def where():
    """
    Returns an array containing elements from lhs when cond is true, and rhs when cond is false.
    Computes the logical and of two arrays.

    Args:
        cond (Union[torch.Tensor, numpy.ndarray]):
                The condition array or tensor.
        lhs (Union[torch.Tensor, numpy.ndarray]):
                The first array or tensor.
        rhs (Union[torch.Tensor, numpy.ndarray]):
                The second array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: Selected elements from lhs if cond is true, and rhs otherwise

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda cond, lhs, rhs: np.where(cond, lhs, rhs),
        "torch": lambda cond, lhs, rhs: torch.where(cond, lhs, rhs),
    }


@mod.export()
@dispatch(num_arrays=2)
def power():
    """
    Computes the element-wise power of an array to the given exponent.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]):
                The base array or tensor.
        exponent (Union[int, float, torch.Tensor, numpy.ndarray]):
                The exponent value or array.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The power result.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj, exponent: np.power(obj, exponent),
        "torch": lambda obj, exponent: torch.pow(obj, exponent),
    }


@mod.export()
@dispatch()
def sum():
    """
    Computes the sum of all elements in the array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Union[Number, torch.Tensor, numpy.ndarray]: The sum of all elements.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.sum(obj),
        "torch": lambda obj: torch.sum(obj),
    }


@mod.export()
@dispatch()
def sqrt():
    """
    Computes the element-wise square root of an array.

    Args:
        obj (Union[torch.Tensor, numpy.ndarray]): The array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The square root results.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda obj: np.sqrt(obj),
        "torch": lambda obj: torch.sqrt(obj),
    }


@mod.export()
@dispatch(num_arrays=2)
def multiply():
    """
    Computes the element-wise multiplication of two arrays.

    Args:
        lhs (Union[torch.Tensor, numpy.ndarray]):
                The first array or tensor.
        rhs (Union[torch.Tensor, numpy.ndarray]):
                The second array or tensor.

    Returns:
        Union[torch.Tensor, numpy.ndarray]: The element-wise product.

    Raises:
        PolygraphyException: if the input is of an unrecognized type.
    """
    return {
        "numpy": lambda lhs, rhs: np.multiply(lhs, rhs),
        "torch": lambda lhs, rhs: torch.mul(lhs, rhs),
    }
