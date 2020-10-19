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
from polygraphy.logger.logger import G_LOGGER, LogMode
from polygraphy.util import misc

import ctypes

import numpy as np


class Cuda(object):
    class MemcpyKind(object):
        HostToHost = ctypes.c_int(0)
        HostToDevice = ctypes.c_int(1)
        DeviceToHost = ctypes.c_int(2)
        DeviceToDevice = ctypes.c_int(3)
        Default = ctypes.c_int(4)


    def __init__(self):
        self.handle = ctypes.CDLL("libcudart.so")
        if not self.handle:
            G_LOGGER.critical("Could not load the CUDA runtime library. Is it on your loader path?")


    def check(self, status):
        if status != 0:
            G_LOGGER.critical("CUDA Error: {:}. To figure out what this means, refer to "
                              "https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038".format(status))


    def create_stream(self):
        stream = ctypes.c_void_p(None)
        self.check(self.handle.cudaStreamCreate(ctypes.byref(stream)))
        return stream


    def stream_synchronize(self, stream):
        self.check(self.handle.cudaStreamSynchronize(stream))


    def destroy_stream(self, stream):
        self.check(self.handle.cudaStreamDestroy(stream))


    def malloc(self, nbytes):
        ptr = ctypes.c_void_p(None)
        self.check(self.handle.cudaMalloc(ctypes.byref(ptr), nbytes))
        return ptr


    def free(self, ptr):
        self.check(self.handle.cudaFree(ptr))


    def htod(self, dst, src, nbytes, stream=None):
        if stream is not None:
            self.check(self.handle.cudaMemcpyAsync(dst, src, nbytes, Cuda.MemcpyKind.HostToDevice, stream))
        else:
            self.check(self.handle.cudaMemcpy(dst, src, nbytes, Cuda.MemcpyKind.HostToDevice))


    def dtoh(self, dst, src, nbytes, stream=None):
        if stream is not None:
            self.check(self.handle.cudaMemcpyAsync(dst, src, nbytes, Cuda.MemcpyKind.DeviceToHost, stream))
        else:
            self.check(self.handle.cudaMemcpy(dst, src, nbytes, Cuda.MemcpyKind.DeviceToHost))


G_CUDA = None
def wrapper():
    global G_CUDA
    G_CUDA = misc.default_value(G_CUDA, Cuda())
    return G_CUDA


class Stream(object):
    def __init__(self):
        self.handle = wrapper().create_stream()


    def free(self):
        wrapper().destroy_stream(self.handle)


    def synchronize(self):
        wrapper().stream_synchronize(self.handle)


    def address(self):
        return self.handle.value


def try_get_stream_handle(stream):
    if stream is None:
        return None
    return stream.handle


class DeviceBuffer(object):
    def __init__(self, shape=None, dtype=None):
        """
        Represents a buffer on the GPU.

        Args:
            shape (Tuple[int]): The initial shape of the buffer.
            dtype (np.dtype): The data type of the buffer.
        """
        self.shape = misc.default_value(shape, tuple())
        self.dtype = misc.default_value(dtype, np.float32)
        self.allocated_nbytes = 0
        self._ptr = ctypes.c_void_p(None)
        self.resize(self.shape)


    def address(self):
        return self._ptr.value


    def allocate(self, nbytes):
        if nbytes:
            self._ptr = wrapper().malloc(nbytes)
            self.allocated_nbytes = nbytes


    def free(self):
        wrapper().free(self._ptr)
        self.shape = tuple()
        self.allocated_nbytes = 0
        self._ptr = ctypes.c_void_p(None)


    def resize(self, shape):
        nbytes = misc.volume(shape) * np.dtype(self.dtype).itemsize
        if nbytes > self.allocated_nbytes:
            self.free()
            self.allocate(nbytes)
        self.shape = shape


    def _check_dtype_matches(self, host_buffer):
        if host_buffer.dtype != self.dtype:
            G_LOGGER.warning("Host buffer type: {:} does not match the type of device buffer: {:}. "
                             "This may cause CUDA errors!".format(host_buffer.dtype, self.dtype))


    def copy_from(self, host_buffer, stream=None):
        if host_buffer.nbytes:
            # When copying from a host buffer, we always adopt its dtype
            self.dtype = host_buffer.dtype
            self._check_dtype_matches(host_buffer)
            self.resize(host_buffer.shape)
            buffer = np.ascontiguousarray(host_buffer.ravel())
            host_ptr = buffer.ctypes.data_as(ctypes.c_void_p)
            wrapper().htod(dst=self._ptr, src=host_ptr, nbytes=host_buffer.nbytes, stream=try_get_stream_handle(stream))


    def copy_to(self, host_buffer, stream=None):
        """
        Copies from this device buffer to the provided host buffer.
        Host buffer must be contiguous in memory (see np.ascontiguousarray).

        Args:
            host_buffer (np.ndarray): The host buffer to copy into.
            stream (Stream):
                    A Stream instance (see util/cuda.py). Performs a synchronous copy if no stream is provided.

        Returns:
            np.ndarray: The host buffer, possibly reallocated if the provided buffer was too small.
        """
        nbytes = misc.volume(self.shape) * np.dtype(self.dtype).itemsize
        self._check_dtype_matches(host_buffer)

        try:
            host_buffer.resize(self.shape, refcheck=False)
        except ValueError:
            host_buffer = np.empty(self.shape, dtype=np.dtype(self.dtype))

        if nbytes:
            host_ptr = host_buffer.ctypes.data_as(ctypes.c_void_p)
            wrapper().dtoh(dst=host_ptr, src=self._ptr, nbytes=nbytes, stream=try_get_stream_handle(stream))
        host_buffer = host_buffer.reshape(self.shape)
        return host_buffer


    def __str__(self):
        return "DeviceBuffer[(dtype={:}, shape={:}), ptr={:}]".format(np.dtype(self.dtype).name, self.shape, hex(self._ptr.value))
