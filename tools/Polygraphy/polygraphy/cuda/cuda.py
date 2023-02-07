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
import ctypes
import time
import os
import sys

from polygraphy import func, mod, util
from polygraphy.logger import G_LOGGER

np = mod.lazy_import("numpy")


def void_ptr(val=None):
    return ctypes.c_void_p(val)


@mod.export()
class MemcpyKind:
    """
    Enumerates different kinds of copy operations.
    """

    HostToHost = ctypes.c_int(0)
    """Copies from host memory to host memory"""
    HostToDevice = ctypes.c_int(1)
    """Copies from host memory to device memory"""
    DeviceToHost = ctypes.c_int(2)
    """Copies from device memory to host memory"""
    DeviceToDevice = ctypes.c_int(3)
    """Copies from device memory to device memory"""
    Default = ctypes.c_int(4)


@mod.export()
class Cuda:
    """
    NOTE: Do *not* construct this class manually.
    Instead, use the ``wrapper()`` function to get the global wrapper.

    Wrapper that exposes low-level CUDA functionality.
    """

    def __init__(self):
        self.handle = None

        fallback_lib = None
        if sys.platform.startswith("win"):
            cuda_paths = [os.environ.get("CUDA_PATH", "")]
            cuda_paths += os.environ.get("PATH", "").split(os.path.pathsep)
            lib_pat = "cudart64_*.dll"
        else:
            cuda_paths = [
                *os.environ.get("LD_LIBRARY_PATH", "").split(os.path.pathsep),
                os.path.join("/", "usr", "local", "cuda", "lib64"),
                os.path.join("/", "usr", "lib"),
                os.path.join("/", "lib"),
            ]
            lib_pat = "libcudart.so*"
            fallback_lib = "libcudart.so"

        cuda_paths = list(filter(lambda x: x, cuda_paths))  # Filter out empty paths (i.e. "")

        candidates = util.find_in_dirs(lib_pat, cuda_paths)
        if not candidates:
            log_func = G_LOGGER.critical if fallback_lib is None else G_LOGGER.warning
            log_func(f"Could not find the CUDA runtime library.\nNote: Paths searched were:\n{cuda_paths}")

            lib = fallback_lib
            G_LOGGER.warning(f"Attempting to load: '{lib}' using default loader paths")
        else:
            G_LOGGER.verbose(f"Found candidate CUDA libraries: {candidates}")
            lib = candidates[0]

        self.handle = ctypes.CDLL(lib)

        if not self.handle:
            G_LOGGER.critical("Could not load the CUDA runtime library. Is it on your loader path?")

    @func.constantmethod
    def check(self, status):
        if status != 0:
            G_LOGGER.critical(
                f"CUDA Error: {status}. To figure out what this means, refer to https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038"
            )

    @func.constantmethod
    def create_stream(self):
        # Signature: () -> int
        ptr = void_ptr()
        self.check(self.handle.cudaStreamCreate(ctypes.byref(ptr)))
        return ptr.value

    @func.constantmethod
    def stream_synchronize(self, ptr):
        # Signature: int -> None
        self.check(self.handle.cudaStreamSynchronize(void_ptr(ptr)))

    @func.constantmethod
    def destroy_stream(self, ptr):
        # Signature: int -> None
        self.check(self.handle.cudaStreamDestroy(void_ptr(ptr)))

    @func.constantmethod
    def malloc(self, nbytes):
        """
        Allocates memory on the GPU.

        Args:
            nbytes (int): The number of bytes to allocate.

        Returns:
            int: The memory address of the allocated region, i.e. a device pointer.

        Raises:
            PolygraphyException: If an error was encountered during the allocation.
        """
        ptr = void_ptr()
        nbytes = ctypes.c_size_t(nbytes)  # Required to prevent overflow
        self.check(self.handle.cudaMalloc(ctypes.byref(ptr), nbytes))
        return ptr.value

    @func.constantmethod
    def free(self, ptr):
        """
        Frees memory allocated on the GPU.

        Args:
            ptr (int): The memory address, i.e. a device pointer.

        Raises:
            PolygraphyException: If an error was encountered during the free.
        """
        self.check(self.handle.cudaFree(void_ptr(ptr)))

    @func.constantmethod
    def memcpy(self, dst, src, nbytes, kind, stream_ptr=None):
        """
        Copies data between host and device memory.

        Args:
            dst (int):
                    The memory address of the destination, i.e. a pointer.
            src (int):
                    The memory address of the source, i.e. a pointer.
            nbytes (int):
                    The number of bytes to copy.
            kind (MemcpyKind):
                    The kind of copy to perform.
            stream_ptr (int):
                    The memory address of a CUDA stream, i.e. a pointer.
                    If this is not provided, a synchronous copy is performed.

        Raises:
            PolygraphyException: If an error was encountered during the copy.
        """
        nbytes = ctypes.c_size_t(nbytes)  # Required to prevent overflow
        if stream_ptr is not None:
            self.check(self.handle.cudaMemcpyAsync(void_ptr(dst), void_ptr(src), nbytes, kind, void_ptr(stream_ptr)))
        else:
            self.check(self.handle.cudaMemcpy(void_ptr(dst), void_ptr(src), nbytes, kind))


G_CUDA = None


@mod.export()
def wrapper():
    """
    Returns the global Polygraphy CUDA wrapper.

    Returns:
        Cuda: The global CUDA wrapper.
    """
    global G_CUDA
    if G_CUDA is None:
        G_CUDA = Cuda()
    return G_CUDA


@mod.export()
class Stream:
    """
    High-level wrapper for a CUDA stream.
    """

    def __init__(self):
        self.ptr = wrapper().create_stream()
        """int: The memory address of the underlying CUDA stream"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Frees the underlying CUDA stream.
        """
        self.free()

    def free(self):
        """
        Frees the underlying CUDA stream.

        You can also use a context manager to manage the stream lifetime.
        For example:
        ::

            with Stream() as stream:
                ...
        """
        wrapper().destroy_stream(self.ptr)
        self.handle = ctypes.c_void_p(None)

    def synchronize(self):
        """
        Synchronizes the stream.
        """
        wrapper().stream_synchronize(self.ptr)


def try_get_stream_handle(stream):
    if stream is None:
        return None
    return stream.ptr


@mod.export()
class DeviceView:
    """
    A read-only view of a GPU memory region.
    """

    def __init__(self, ptr, shape, dtype):
        """
        Args:
            ptr (int): A pointer to the region of memory.

            shape (Tuple[int]): The shape of the region.
            dtype (numpy.dtype): The data type of the region.
        """
        self.ptr = int(ptr)
        """int: The memory address of the underlying GPU memory"""
        self.shape = shape
        """Tuple[int]: The shape of the device buffer"""
        self.itemsize = None
        self.dtype = dtype
        """np.dtype: The data type of the device buffer"""

    def _check_host_buffer(self, host_buffer, copying_from):
        if host_buffer.dtype != self.dtype:
            G_LOGGER.error(
                f"Host buffer type: {host_buffer.dtype} does not match the type of this device buffer: {self.dtype}. This may cause CUDA errors!"
            )

        if not util.is_contiguous(host_buffer):
            G_LOGGER.critical(
                "Provided host buffer is not contiguous in memory.\n"
                "Hint: Use `util.make_contiguous()` or `np.ascontiguousarray()` to make the array contiguous in memory."
            )

        # If the host buffer is an input, the device buffer should be large enough to accomodate it.
        # Otherwise, the host buffer needs to be large enough to accomodate the device buffer.
        if copying_from:
            if host_buffer.nbytes > self.nbytes:
                G_LOGGER.critical(
                    f"Provided host buffer is larger than device buffer.\n"
                    f"Note: host buffer is {host_buffer.nbytes} bytes but device buffer is only {self.nbytes} bytes.\n"
                    f"Hint: Use `resize()` to resize the device buffer to the correct shape."
                )
        else:
            if host_buffer.nbytes < self.nbytes:
                G_LOGGER.critical(
                    f"Provided host buffer is smaller than device buffer.\n"
                    f"Note: host buffer is only {host_buffer.nbytes} bytes but device buffer is {self.nbytes} bytes.\n"
                    f"Hint: Use `util.resize_buffer()` to resize the host buffer to the correct shape."
                )

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, new):
        self._dtype = new
        self.itemsize = np.dtype(new).itemsize

    @property
    def nbytes(self):
        """
        The number of bytes in the memory region.
        """
        return util.volume(self.shape) * self.itemsize

    @func.constantmethod
    def copy_to(self, host_buffer, stream=None):
        """
        Copies from this device buffer to the provided host buffer.

        Args:
            host_buffer (numpy.ndarray):
                    The host buffer to copy into. The buffer must be contiguous in
                    memory (see np.ascontiguousarray) and large enough to accomodate
                    the device buffer.
            stream (Stream):
                    A Stream instance. Performs a synchronous copy if no stream is provided.

        Returns:
            np.ndarray: The host buffer
        """
        if not self.nbytes:
            return host_buffer

        self._check_host_buffer(host_buffer, copying_from=False)
        wrapper().memcpy(
            dst=host_buffer.ctypes.data,
            src=self.ptr,
            nbytes=self.nbytes,
            kind=MemcpyKind.DeviceToHost,
            stream_ptr=try_get_stream_handle(stream),
        )
        return host_buffer

    @func.constantmethod
    def numpy(self):
        """
        Create a new NumPy array containing the contents of this device buffer.

        Returns:
            np.ndarray: The newly created NumPy array.
        """
        arr = np.empty(self.shape, dtype=self.dtype)
        self.copy_to(arr)
        return arr

    def __str__(self):
        return f"DeviceView[(dtype={np.dtype(self.dtype).name}, shape={self.shape}), ptr={hex(self.ptr)}]"

    def __repr__(self):
        return util.make_repr("DeviceView", ptr=self.ptr, shape=self.shape, dtype=self.dtype)[0]


@mod.export()
class DeviceArray(DeviceView):
    """
    An array on the GPU.
    """

    def __init__(self, shape=None, dtype=None):
        """
        Args:
            shape (Tuple[int]): The initial shape of the buffer.
            dtype (numpy.dtype): The data type of the buffer.
        """
        super().__init__(ptr=0, shape=util.default(shape, tuple()), dtype=util.default(dtype, np.float32))
        self.allocated_nbytes = 0
        self.resize(self.shape)

    def __enter__(self):
        return self

    @staticmethod
    def raw(shape):
        """
        Creates an untyped device array of the specified shape.

        Args:
            shape (Tuple[int]):
                The initial shape of the buffer, in units of bytes.
                For example, a shape of ``(4, 4)`` would allocate a 16 byte array.

        Returns:
            DeviceArray: The raw device array.
        """
        return DeviceArray(shape=shape, dtype=np.byte)

    def resize(self, shape):
        """
        Resizes or reshapes the array to the specified shape.

        If the allocated memory region is already large enough,
        no reallocation is performed.

        Args:
            shape (Tuple[int]): The new shape.
        """
        nbytes = util.volume(shape) * self.itemsize
        if nbytes > self.allocated_nbytes:
            self.free()
            self.ptr = wrapper().malloc(nbytes)
            self.allocated_nbytes = nbytes
        self.shape = shape

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Frees the underlying memory of this DeviceArray.
        """
        self.free()

    def free(self):
        """
        Frees the GPU memory associated with this array.

        You can also use a context manager to ensure that memory is freed. For example:
        ::

            with DeviceArray(...) as arr:
                ...
        """
        wrapper().free(self.ptr)
        self.shape = tuple()
        self.allocated_nbytes = 0
        self.ptr = 0

    def copy_from(self, host_buffer, stream=None):
        """
        Copies from the provided host buffer into this device buffer.

        Args:
            host_buffer (numpy.ndarray):
                    The host buffer to copy from. The buffer must be contiguous in
                    memory (see np.ascontiguousarray) and not larger than this device buffer.
            stream (Stream):
                    A Stream instance. Performs a synchronous copy if no stream is provided.

        Returns:
            DeviceArray: self
        """
        if not host_buffer.nbytes:
            return self

        self._check_host_buffer(host_buffer, copying_from=True)
        wrapper().memcpy(
            dst=self.ptr,
            src=host_buffer.ctypes.data,
            nbytes=host_buffer.nbytes,
            kind=MemcpyKind.HostToDevice,
            stream_ptr=try_get_stream_handle(stream),
        )
        return self

    def view(self, shape=None, dtype=None):
        """
        Creates a read-only DeviceView from this DeviceArray.

        Args:
            shape (Sequence[int]):
                    The desired shape of the view.
                    Defaults to the shape of this array or view.
            dtype (numpy.dtype):
                    The desired data type of the view.
                    Defaults to the data type of this array or view.

        Returns:
            DeviceView: A view of this arrays data on the device.
        """
        shape = util.default(shape, self.shape)
        dtype = util.default(dtype, self.dtype)
        view = DeviceView(self.ptr, shape, dtype)

        if view.nbytes > self.nbytes:
            G_LOGGER.critical(
                "A view cannot exceed the number of bytes of the original array.\n"
                f"Note: Original array has shape: {self.shape} and dtype: {self.dtype}, which requires {self.nbytes} bytes, "
                f"while the view has shape: {shape} and dtype: {dtype}, which requires {view.nbytes} bytes, "
            )
        return view

    def __str__(self):
        return f"DeviceArray[(dtype={np.dtype(self.dtype).name}, shape={self.shape}), ptr={hex(self.ptr)}]"

    def __repr__(self):
        return util.make_repr("DeviceArray", shape=self.shape, dtype=self.dtype)[0]
