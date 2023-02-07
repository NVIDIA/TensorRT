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
import numpy as np
import pytest
import tensorrt as trt
from polygraphy import mod, util
from polygraphy.cuda import DeviceArray, Stream, DeviceView, wrapper, MemcpyKind
from tests.helper import time_func


class TestDeviceView:
    def test_basic(self):
        with DeviceArray(shape=(1, 4, 2), dtype=np.float32) as arr:
            v = DeviceView(arr.ptr, arr.shape, arr.dtype)
            assert v.ptr == arr.ptr
            assert v.shape == arr.shape
            assert v.dtype == arr.dtype
            assert v.nbytes == arr.nbytes

    def test_with_int_ptr(self):
        ptr = 74892
        v = DeviceView(ptr=ptr, shape=(1,), dtype=np.float32)
        assert v.ptr == ptr

    def test_copy_to(self):
        with DeviceArray((2, 2), dtype=np.float32) as arr:
            arr.copy_from(np.ones((2, 2), dtype=np.float32) * 4)

            v = DeviceView(arr.ptr, arr.shape, arr.dtype)
            host_buf = np.zeros((2, 2), dtype=np.float32)
            v.copy_to(host_buf)

            assert np.all(host_buf == 4)

    def test_numpy(self):
        with DeviceArray((2, 2), dtype=np.float32) as arr:
            arr.copy_from(np.ones((2, 2), dtype=np.float32) * 4)

            v = DeviceView(arr.ptr, arr.shape, arr.dtype)
            assert np.all(v.numpy() == 4)


class ResizeTestCase:
    # *_bytes is the size of the allocated buffer, old/new are the apparent shapes of the buffer.
    def __init__(self, old, old_size, new, new_size):
        self.old = old
        self.old_bytes = old_size * np.float32().itemsize
        self.new = new
        self.new_bytes = new_size * np.float32().itemsize


RESIZES = [
    ResizeTestCase(tuple(), 1, (1, 1, 1), 1),  # Reshape (no-op)
    ResizeTestCase((2, 2, 2), 8, (1, 1), 8),  # Resize to smaller buffer
    ResizeTestCase((2, 2, 2), 8, (9, 9), 81),  # Resize to larger buffer
]


class TestDeviceBuffer:
    @pytest.mark.parametrize("shapes", RESIZES)
    def test_device_buffer_resize(self, shapes):
        with DeviceArray(shapes.old) as buf:
            assert buf.allocated_nbytes == shapes.old_bytes
            assert buf.shape == shapes.old
            buf.resize(shapes.new)
            assert buf.allocated_nbytes == shapes.new_bytes
            assert buf.shape == shapes.new

    @pytest.mark.serial  # Sometimes the GPU may run out of memory if too many other tests are also running.
    def test_large_allocation(self):
        dtype = np.byte
        # See if we can alloc 3GB (bigger than value of signed int)
        shape = (3 * 1024 * 1024 * 1024,)
        with DeviceArray(shape=shape, dtype=dtype) as buf:
            assert buf.allocated_nbytes == util.volume(shape) * np.dtype(dtype).itemsize

    def test_device_buffer_memcpy_async(self):
        shape = (1, 384)
        arr = np.ones(shape, dtype=np.int32)

        with DeviceArray(shape) as buf, Stream() as stream:
            buf.copy_from(arr)

            new_arr = np.empty(shape=shape, dtype=np.int32)
            buf.copy_to(new_arr, stream)

            stream.synchronize()

            assert np.all(new_arr == arr)

    def test_device_buffer_memcpy_sync(self):
        shape = (1, 384)
        arr = np.ones(shape, dtype=np.int32)

        with DeviceArray(shape) as buf:
            buf.copy_from(arr)

            new_arr = np.empty(shape=shape, dtype=np.int32)
            buf.copy_to(new_arr)

            assert np.all(new_arr == arr)

    def test_device_buffer_free(self):
        buf = DeviceArray(shape=(64, 64), dtype=np.float32)
        assert buf.allocated_nbytes == 64 * 64 * np.float32().itemsize

        buf.free()
        assert buf.allocated_nbytes == 0
        assert buf.shape == tuple()

    def test_empty_tensor_to_host(self):
        with DeviceArray(shape=(5, 2, 0, 3, 0), dtype=np.float32) as buf:
            assert util.volume(buf.shape) == 0

            host_buf = np.empty(shape=(5, 2, 0, 3, 0), dtype=np.float32)
            assert util.volume(host_buf.shape) == 0

            buf.copy_to(host_buf)
            assert host_buf.shape == buf.shape
            assert host_buf.nbytes == 0
            assert util.volume(host_buf.shape) == 0

    @pytest.mark.flaky
    @pytest.mark.serial
    def test_copy_from_overhead(self):
        host_buf = np.ones(shape=(4, 8, 1024, 1024), dtype=np.float32)
        with DeviceArray(shape=host_buf.shape, dtype=host_buf.dtype) as dev_buf:
            memcpy_time = time_func(
                lambda: wrapper().memcpy(
                    dst=dev_buf.ptr,
                    src=host_buf.ctypes.data,
                    nbytes=host_buf.nbytes,
                    kind=MemcpyKind.HostToDevice,
                )
            )

            copy_from_time = time_func(lambda: dev_buf.copy_from(host_buf))

        print(f"memcpy time: {memcpy_time}, copy_from time: {copy_from_time}")
        assert copy_from_time <= (memcpy_time * 1.05)

    @pytest.mark.flaky
    @pytest.mark.serial
    def test_copy_to_overhead(self):
        host_buf = np.ones(shape=(4, 8, 1024, 1024), dtype=np.float32)
        with DeviceArray(shape=host_buf.shape, dtype=host_buf.dtype) as dev_buf:
            memcpy_time = time_func(
                lambda: wrapper().memcpy(
                    dst=host_buf.ctypes.data,
                    src=dev_buf.ptr,
                    nbytes=host_buf.nbytes,
                    kind=MemcpyKind.DeviceToHost,
                )
            )

            copy_to_time = time_func(lambda: dev_buf.copy_to(host_buf))

        print(f"memcpy time: {memcpy_time}, copy_to time: {copy_to_time}")
        assert copy_to_time <= (memcpy_time * 1.05)

    def test_raw(self):
        with DeviceArray.raw((25,)) as buf:
            assert buf.shape == (25,)
            assert buf.nbytes == 25
            buf.resize((30,))
            assert buf.shape == (30,)
            assert buf.nbytes == 30
