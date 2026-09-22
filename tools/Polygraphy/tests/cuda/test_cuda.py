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
import importlib.util
import os
import platform

import numpy as np
import pytest
import torch

from polygraphy import util
from polygraphy.cuda import DeviceArray, DeviceView, MemcpyKind, Stream, wrapper
from polygraphy.cuda import cuda as cuda_mod
from polygraphy.cuda.cuda import Cuda, _find_cuda_lib_dirs
from tests.helper import time_func


class TestFindCudaLibDirs:
    def _windows_env(self, monkeypatch, path_dirs=None, sys_path=None, machine="AMD64"):
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("sys.path", sys_path or [])
        monkeypatch.setattr("platform.machine", lambda: machine)
        # By default pretend the `nvidia.cuda_runtime` wheel is not installed, so tests do not depend
        # on the host environment. Tests exercising that lookup override this.
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(
            os,
            "environ",
            {
                "CUDA_PATH": "cuda_path",
                "CUDA_HOME": "cuda_home",
                "CUDA_PATH_V12_4": "cuda_v12_4",
                "PATH": os.path.pathsep.join(path_dirs or []),
                "NOT_CUDA": "unrelated",
            },
        )

    def test_windows_toolkit_env_vars_included(self, monkeypatch):
        # Each toolkit-root variable contributes the root, its `bin`, and its arch-specific `bin`
        # subdir (where cudart64_*.dll may live), in that order.
        self._windows_env(monkeypatch)
        dirs = _find_cuda_lib_dirs()
        for root in ["cuda_path", "cuda_home", "cuda_v12_4"]:
            assert root in dirs
            assert os.path.join(root, "bin") in dirs
            assert os.path.join(root, "bin", "x64") in dirs
            assert dirs.index(os.path.join(root, "bin")) == dirs.index(root) + 1
            assert dirs.index(os.path.join(root, "bin", "x64")) == dirs.index(root) + 2
        assert "unrelated" not in dirs

    @pytest.mark.parametrize("machine,arch_dir", [("AMD64", "x64"), ("ARM64", "arm64")])
    def test_windows_arch_specific_bin_subdir(self, monkeypatch, machine, arch_dir):
        # The arch-specific bin subdir is appended for both x64 and ARM64 hosts.
        self._windows_env(monkeypatch, machine=machine)
        dirs = _find_cuda_lib_dirs()
        assert os.path.join("cuda_path", "bin", arch_dir) in dirs

    def test_windows_unknown_arch_no_arch_subdir(self, monkeypatch):
        # For an unrecognized machine type no arch-specific subdir is added.
        self._windows_env(monkeypatch, machine="x86")
        dirs = _find_cuda_lib_dirs()
        assert not any("bin" + os.sep in d for d in dirs)

    def test_windows_path_and_wheel_dirs_included(self, monkeypatch, tmp_path):
        # Two PATH entries confirm pathsep split; a sys.path entry yields the pip-wheel cudart bin dir.
        # The sys.path entry must be a real directory (see test below) so use tmp_path.
        site_packages = str(tmp_path)
        path_dirs = ["some_path_dir_a", "some_path_dir_b"]
        self._windows_env(monkeypatch, path_dirs=path_dirs, sys_path=[site_packages])
        dirs = _find_cuda_lib_dirs()
        for path_dir in path_dirs:
            assert path_dir in dirs
        assert os.path.join(site_packages, "nvidia", "cuda_runtime", "bin") in dirs

    def test_windows_skips_non_directory_sys_path_entries(self, monkeypatch, tmp_path):
        # `sys.path` entries that are files (e.g. a console-script `.exe` or zipapp) or that do not
        # exist must not contribute a (nonsensical) wheel bin dir; only real directories do.
        real_dir = tmp_path / "site-packages"
        real_dir.mkdir()
        a_file = tmp_path / "polygraphy.exe"
        a_file.write_text("")
        missing = tmp_path / "does_not_exist"
        self._windows_env(
            monkeypatch, sys_path=[str(real_dir), str(a_file), str(missing)]
        )
        dirs = _find_cuda_lib_dirs()
        assert os.path.join(str(real_dir), "nvidia", "cuda_runtime", "bin") in dirs
        assert not any(str(a_file) in d for d in dirs)
        assert not any(str(missing) in d for d in dirs)

    def test_windows_normalizes_and_dedups_path_variants(self, monkeypatch):
        # A PATH entry that differs from a toolkit-derived dir only in redundant components
        # (e.g. a trailing `.` or doubled separators) must collapse to a single entry.
        redundant = os.path.join("cuda_path", "bin", ".")
        self._windows_env(monkeypatch, path_dirs=[redundant])
        dirs = _find_cuda_lib_dirs()
        assert dirs.count(os.path.join("cuda_path", "bin")) == 1

    def test_windows_finds_installed_cuda_runtime_package(self, monkeypatch):
        # The installed `nvidia.cuda_runtime` wheel is located directly via the import system, so its
        # `bin` dir is searched even when its parent is not a `sys.path` entry.
        pkg_dir = os.path.join("C:\\elsewhere", "nvidia", "cuda_runtime")

        class FakeSpec:
            submodule_search_locations = [pkg_dir]

        self._windows_env(monkeypatch)
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: FakeSpec() if name == "nvidia.cuda_runtime" else None,
        )
        dirs = _find_cuda_lib_dirs()
        assert os.path.join(pkg_dir, "bin") in dirs

    def test_windows_deduplicates_preserving_order(self, monkeypatch):
        # CUDA_PATH and a versioned variable can point at the same root; the candidate list must
        # not contain duplicates, and first-seen order is preserved.
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("sys.path", [])
        monkeypatch.setattr("platform.machine", lambda: "AMD64")
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(
            os,
            "environ",
            {"CUDA_PATH": "same_root", "CUDA_PATH_V12_4": "same_root", "PATH": ""},
        )

        dirs = _find_cuda_lib_dirs()

        assert dirs == [
            "same_root",
            os.path.join("same_root", "bin"),
            os.path.join("same_root", "bin", "x64"),
        ]

    def test_windows_filters_empty(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("sys.path", [])
        monkeypatch.setattr("platform.machine", lambda: "AMD64")
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        # No CUDA-related variables set; everything should be filtered out.
        monkeypatch.setattr(
            os, "environ", {"CUDA_PATH": "", "CUDA_HOME": "", "PATH": ""}
        )

        dirs = _find_cuda_lib_dirs()

        assert dirs == []

    def test_linux_branch_unchanged(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr("sys.path", [])
        # Pretend the `nvidia.cuda_runtime` wheel is not installed (covered separately below).
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(
            os,
            "environ",
            {
                "LD_LIBRARY_PATH": os.path.pathsep.join(["/foo", "/bar"]),
                # Windows-only vars must not influence the Linux candidate list.
                "CUDA_PATH": "C:\\cuda_path",
                "CUDA_PATH_V12_4": "C:\\cuda_v12_4",
            },
        )

        dirs = _find_cuda_lib_dirs()

        # The exact-equality assertion proves no Windows var leaks in.
        assert dirs == [
            "/foo",
            "/bar",
            os.path.join("/", "usr", "local", "cuda", "lib64"),
            os.path.join("/", "usr", "lib"),
            os.path.join("/", "lib"),
        ]

    def test_linux_finds_installed_cuda_runtime_package(self, monkeypatch):
        # On Linux the wheel ships libcudart.so* under `nvidia/cuda_runtime/lib`, located via both a
        # `sys.path` site-packages entry and the import system.
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setattr("sys.path", ["/site-packages"])
        pkg_dir = os.path.join("/elsewhere", "nvidia", "cuda_runtime")

        class FakeSpec:
            submodule_search_locations = [pkg_dir]

        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: FakeSpec() if name == "nvidia.cuda_runtime" else None,
        )
        monkeypatch.setattr(os, "environ", {"LD_LIBRARY_PATH": ""})

        # `/site-packages` must exist for the `sys.path` lookup to consider it.
        monkeypatch.setattr(os.path, "isdir", lambda p: True)

        dirs = _find_cuda_lib_dirs()

        assert os.path.join("/site-packages", "nvidia", "cuda_runtime", "lib") in dirs
        assert os.path.join(pkg_dir, "lib") in dirs


class TestCudaDllDirectory:
    def test_windows_registers_dependent_dll_directory(self, monkeypatch):
        # On Windows, the directory of the discovered CUDA runtime must be registered via
        # os.add_dll_directory so its dependent DLLs (e.g. those in a pip wheel) resolve.
        monkeypatch.setattr("sys.platform", "win32")

        lib = os.path.join(
            "C:\\site-packages", "nvidia", "cuda_runtime", "bin", "cudart64_12.dll"
        )
        monkeypatch.setattr(
            cuda_mod, "_find_cuda_lib_dirs", lambda: [os.path.dirname(lib)]
        )
        monkeypatch.setattr(cuda_mod.util, "find_in_dirs", lambda pat, dirs: [lib])

        # Spy on os.add_dll_directory; raising=False since it does not exist on non-Windows hosts.
        registered = []
        monkeypatch.setattr(
            os, "add_dll_directory", lambda d: registered.append(d), raising=False
        )
        # Avoid actually loading a DLL.
        monkeypatch.setattr(cuda_mod.ctypes, "CDLL", lambda lib: object())

        Cuda()

        assert registered == [os.path.dirname(lib)]


class TestDeviceView:
    def test_basic(self):
        with DeviceArray(shape=(1, 4, 2), dtype=np.float32) as arr:
            v = DeviceView(arr.ptr, arr.shape, arr.dtype)
            assert v.ptr == arr.ptr
            assert v.shape == arr.shape
            assert v.dtype == arr.dtype
            assert v.nbytes == arr.nbytes
            # For backwards compatibility
            assert isinstance(arr.dtype, np.dtype)
            assert isinstance(v.dtype, np.dtype)

    def test_with_int_ptr(self):
        ptr = 74892
        v = DeviceView(ptr=ptr, shape=(1,), dtype=np.float32)
        assert v.ptr == ptr

    @pytest.mark.parametrize("module", [np, torch])
    def test_copy_to(self, module):
        with DeviceArray((2, 2), dtype=np.float32) as arr:
            arr.copy_from(module.ones((2, 2), dtype=module.float32) * 4)

            v = DeviceView(arr.ptr, arr.shape, arr.dtype)
            host_buf = module.zeros((2, 2), dtype=module.float32)
            v.copy_to(host_buf)

            assert module.all(host_buf == 4)

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
        host_buf = np.ones(shape=(4, 8, 512, 512), dtype=np.float32)
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
        assert copy_from_time <= (memcpy_time * 1.12)

    @pytest.mark.flaky
    @pytest.mark.serial
    def test_copy_to_overhead(self):
        host_buf = np.ones(shape=(4, 8, 512, 512), dtype=np.float32)
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
        assert copy_to_time <= (memcpy_time * 1.12)

    def test_raw(self):
        with DeviceArray.raw((25,)) as buf:
            assert buf.shape == (25,)
            assert buf.nbytes == 25
            buf.resize((30,))
            assert buf.shape == (30,)
            assert buf.nbytes == 30
