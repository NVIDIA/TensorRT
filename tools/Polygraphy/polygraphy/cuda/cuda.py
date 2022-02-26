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
class MemcpyKind(object):
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
class Cuda(object):
    """
    NOTE: Do *not* construct this class manually.
    Instead, use the ``wrapper()`` function to get the global wrapper.

    Wrapper that exposes low-level CUDA functionality.
    """

    def __init__(self):
        self.handle = None
        if sys.platform.startswith("win"):
            cuda_paths = [os.environ.get("CUDA_PATH", "")]
            cuda_paths += os.environ.get("PATH", "").split(os.path.pathsep)
            cuda_paths = list(filter(lambda x: x, cuda_paths))  # Filter out empty paths (i.e. "")

            candidates = util.find_in_dirs("cudart64_*.dll", cuda_paths)
            if not candidates:
                G_LOGGER.critical(
                    "Could not find the CUDA runtime library.\nNote: Paths searched were:\n{:}".format(cuda_paths)
                )

            self.handle = ctypes.CDLL(candidates[0])
        else:
            self.handle = ctypes.CDLL("libcudart.so")

        if not self.handle:
            G_LOGGER.critical("Could not load the CUDA runtime library. Is it on your loader path?")

    @func.constantmethod
    def check(self, status):
        if status != 0:
            G_LOGGER.critical(
                "CUDA Error: {:}. To figure out what this means, refer to "
                "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038".format(
                    status
                )
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
class Stream(object):
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


# Make a numpy array contiguous if it's not already.
def make_np_contiguous(arr):
    if not arr.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(arr)
    return arr


@mod.export()
class DeviceView(object):
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

    def _check_dtype_matches(self, host_buffer):
        if host_buffer.dtype != self.dtype:
            G_LOGGER.error(
                "Host buffer type: {:} does not match the type of this device buffer: {:}. "
                "This may cause CUDA errors!".format(host_buffer.dtype, self.dtype)
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
                    The host buffer to copy into. The buffer will be reshaped to match the
                    shape of this device buffer. If the provided host buffer is too small,
                    it will be freed and reallocated.
                    The buffer may also be reallocated if it is not contiguous in
                    memory (see np.ascontiguousarray).
            stream (Stream):
                    A Stream instance. Performs a synchronous copy if no stream is provided.

        Returns:
            numpy.ndarray: The host buffer, possibly reallocated.
        """
        self._check_dtype_matches(host_buffer)

        if self.shape != host_buffer.shape:
            try:
                host_buffer.resize(self.shape, refcheck=False)
            except ValueError as err:
                G_LOGGER.warning(
                    "Could not resize host buffer to shape: {:}. Allocating a new buffer instead.\n"
                    "Note: Error was: {:}".format(self.shape, err)
                )
                host_buffer = np.empty(self.shape, dtype=np.dtype(self.dtype))

        if not self.nbytes:
            return host_buffer

        host_buffer = make_np_contiguous(host_buffer)
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
        return self.copy_to(arr)

    def __str__(self):
        return "DeviceView[(dtype={:}, shape={:}), ptr={:}]".format(
            np.dtype(self.dtype).name, self.shape, hex(self.ptr)
        )


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

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Frees the underlying memory of this DeviceArray.
        """
        self.free()

    def allocate(self, nbytes):
        if nbytes:
            self.ptr = wrapper().malloc(nbytes)
            self.allocated_nbytes = nbytes

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
            self.allocate(nbytes)
        self.shape = shape

    def copy_from(self, host_buffer, stream=None):
        """
        Copies from the provided host buffer into this device buffer.
        The device array may be resized if the currently allocated memory region
        is smaller than the host_buffer.

        Args:
            host_buffer (numpy.ndarray):
                    The host buffer to copy from. If the buffer is not contiguous in memory,
                    an additional copy may be performed.
            stream (Stream):
                    A Stream instance. Performs a synchronous copy if no stream is provided.

        Returns:
            DeviceArray: Self
        """
        if host_buffer.nbytes:
            self._check_dtype_matches(host_buffer)
            self.resize(host_buffer.shape)
            host_buffer = make_np_contiguous(host_buffer)
            wrapper().memcpy(
                dst=self.ptr,
                src=host_buffer.ctypes.data,
                nbytes=host_buffer.nbytes,
                kind=MemcpyKind.HostToDevice,
                stream_ptr=try_get_stream_handle(stream),
            )
        return self

    def view(self):
        """
        Creates a read-only DeviceView from this DeviceArray.

        Returns:
            DeviceView: A view of this arrays data on the device.
        """
        return DeviceView(self.ptr, self.shape, self.dtype)

    def __str__(self):
        return "DeviceArray[(dtype={:}, shape={:}), ptr={:}]".format(
            np.dtype(self.dtype).name, self.shape, hex(self.ptr)
        )
