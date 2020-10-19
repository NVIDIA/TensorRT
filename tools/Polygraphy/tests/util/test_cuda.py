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
import ctypes

import numpy as np
import pytest
from polygraphy.util import misc
from polygraphy.util.cuda import DeviceBuffer, Stream


class ResizeTestCase(object):
    # *_bytes is the size of the allocated buffer, old/new are the apparent shapes of the buffer.
    def __init__(self, old, old_size, new, new_size):
        self.old = old
        self.old_bytes = old_size * np.float32().itemsize
        self.new = new
        self.new_bytes = new_size * np.float32().itemsize

RESIZES = [
    ResizeTestCase(tuple(), 1, (1, 1, 1), 1), # Reshape (no-op)
    ResizeTestCase((2, 2, 2), 8, (1, 1), 8), # Resize to smaller buffer
    ResizeTestCase((2, 2, 2), 8, (9, 9), 81), # Resize to larger buffer
]


class TestDeviceBuffer(object):
    @pytest.mark.parametrize("shapes", RESIZES)
    def test_device_buffer_resize(self, shapes):
        buf = DeviceBuffer(shapes.old)
        assert buf.allocated_nbytes == shapes.old_bytes
        assert buf.shape == shapes.old
        buf.resize(shapes.new)
        assert buf.allocated_nbytes == shapes.new_bytes
        assert buf.shape == shapes.new


    def test_device_buffer_memcpy_async(self):
        stream = Stream()
        arr = np.ones((1, 384), dtype=np.int32)

        buf = DeviceBuffer()
        buf.resize(arr.shape)
        buf.copy_from(arr)

        new_arr = np.empty((1, 384), dtype=np.int32)
        buf.copy_to(new_arr, stream)

        stream.synchronize()

        assert np.all(new_arr == arr)


    def test_device_buffer_memcpy_sync(self):
        arr = np.ones((1, 384), dtype=np.int32)

        buf = DeviceBuffer()
        buf.resize(arr.shape)
        buf.copy_from(arr)

        new_arr = np.empty((1, 384), dtype=np.int32)
        buf.copy_to(new_arr)

        assert np.all(new_arr == arr)


    def test_device_buffer_free(self):
        buf = DeviceBuffer(shape=(64, 64), dtype=np.float32)
        assert buf.allocated_nbytes == 64 * 64 * np.float32().itemsize

        buf.free()
        assert buf.allocated_nbytes == 0
        assert buf.shape == tuple()


    def test_empty_tensor_to_host(self):
        buf = DeviceBuffer(shape=(5, 2, 0, 3, 0), dtype=np.float32)
        assert misc.volume(buf.shape) == 0

        host_buf = np.empty(tuple(), dtype=np.float32)
        assert misc.volume(host_buf.shape) == 1

        host_buf = buf.copy_to(host_buf)
        assert host_buf.shape == buf.shape
        assert host_buf.nbytes == 0
        assert misc.volume(host_buf.shape) == 0


class TestStream(object):
    def test_handle_is_ctypes_ptr(self):
        # If the handle is not a c_void_p (e.g. just an int), then it may cause segfaults when used.
        stream = Stream()
        assert isinstance(stream.handle, ctypes.c_void_p)
