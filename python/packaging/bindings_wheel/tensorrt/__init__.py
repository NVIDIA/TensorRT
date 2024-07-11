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

import ctypes
import os
import sys
import warnings

# For standalone wheels, attempt to import the wheel containing the libraries.
_libs_wheel_imported = False
try:
    import ##TENSORRT_MODULE##_libs
except (ImportError, ModuleNotFoundError):
    pass
else:
    _libs_wheel_imported = True


if not _libs_wheel_imported and sys.platform.startswith("win"):
    # On Windows, we need to manually open the TensorRT libraries - otherwise we are unable to
    # load the bindings. If we imported the tensorrt_libs wheel, then that should have taken care of it for us.
    def find_lib(name):
        paths = os.environ["PATH"].split(os.path.pathsep)
        for path in paths:
            libpath = os.path.join(path, name)
            if os.path.isfile(libpath):
                return libpath

        if name.startswith("cudnn") or name.startswith("cublas"):
            return ""

        raise FileNotFoundError(
            "Could not find: {:}. Is it on your PATH?\nNote: Paths searched were:\n{:}".format(name, paths)
        )

    # Order matters here because of dependencies
    LIBRARIES = {
        "tensorrt": [
            "nvinfer_##TENSORRT_MAJOR##.dll",
            "cublas64_##CUDA_MAJOR##.dll",
            "cublasLt64_##CUDA_MAJOR##.dll",
            "cudnn64_##CUDNN_MAJOR##.dll",
            "nvinfer_plugin_##TENSORRT_MAJOR##.dll",
            "nvonnxparser_##TENSORRT_MAJOR##.dll",
        ],
        "tensorrt_dispatch": [
            "nvinfer_dispatch_##TENSORRT_MAJOR##.dll",
        ],
        "tensorrt_lean": [
            "nvinfer_lean_##TENSORRT_MAJOR##.dll",
        ],
    }["##TENSORRT_MODULE##"]

    for lib in LIBRARIES:
        lib_path = find_lib(lib)
        if lib_path != "":
            ctypes.CDLL(lib_path)

del _libs_wheel_imported

from .##TENSORRT_MODULE## import *

__version__ = "##TENSORRT_PYTHON_VERSION##"


# Provides Python's `with` syntax
def common_enter(this):
    warnings.warn(
        "Context managers for TensorRT types are deprecated. "
        "Memory will be freed automatically when the reference count reaches 0.",
        DeprecationWarning,
    )
    return this


def common_exit(this, exc_type, exc_value, traceback):
    """
    Context managers are deprecated and have no effect. Objects are automatically freed when
    the reference count reaches 0.
    """
    pass


# Logger does not have a destructor.
ILogger.__enter__ = common_enter
ILogger.__exit__ = lambda this, exc_type, exc_value, traceback: None

ICudaEngine.__enter__ = common_enter
ICudaEngine.__exit__ = common_exit

IExecutionContext.__enter__ = common_enter
IExecutionContext.__exit__ = common_exit

Runtime.__enter__ = common_enter
Runtime.__exit__ = common_exit

IHostMemory.__enter__ = common_enter
IHostMemory.__exit__ = common_exit

if "##TENSORRT_MODULE##" == "tensorrt":
    Builder.__enter__ = common_enter
    Builder.__exit__ = common_exit

    INetworkDefinition.__enter__ = common_enter
    INetworkDefinition.__exit__ = common_exit

    OnnxParser.__enter__ = common_enter
    OnnxParser.__exit__ = common_exit

    IBuilderConfig.__enter__ = common_enter
    IBuilderConfig.__exit__ = common_exit


# Add logger severity into the default implementation to preserve backwards compatibility.
Logger.Severity = ILogger.Severity

for attr, value in ILogger.Severity.__members__.items():
    setattr(Logger, attr, value)


# Computes the volume of an iterable.
def volume(iterable):
    """
    Computes the volume of an iterable.

    :arg iterable: Any python iterable, including a :class:`Dims` object.

    :returns: The volume of the iterable. This will return 1 for empty iterables, as a scalar has an empty shape and the volume of a tensor with empty shape is 1.
    """
    vol = 1
    for elem in iterable:
        vol *= elem
    return vol


# Converts a TensorRT datatype to the equivalent numpy type.
def nptype(trt_type):
    """
    Returns the numpy-equivalent of a TensorRT :class:`DataType` .

    :arg trt_type: The TensorRT data type to convert.

    :returns: The equivalent numpy type.
    """
    import numpy as np

    mapping = {
        float32: np.float32,
        float16: np.float16,
        int8: np.int8,
        int32: np.int32,
        int64: np.int64,
        bool: np.bool_,
        uint8: np.uint8,
        # Note: fp8 and bfloat16 have no equivalent numpy type
    }
    if trt_type in mapping:
        return mapping[trt_type]
    raise TypeError("Could not resolve TensorRT datatype to an equivalent numpy datatype.")


# Add a numpy-like itemsize property to the datatype.
def _itemsize(trt_type):
    """
    Returns the size in bytes of this :class:`DataType`.
    The returned size is a rational number, possibly a `Real` denoting a fraction of a byte.

    :arg trt_type: The TensorRT data type.

    :returns: The size of the type.
    """
    mapping = {
        float32: 4,
        float16: 2,
        bfloat16: 2,
        int8: 1,
        int32: 4,
        int64: 8,
        bool: 1,
        uint8: 1,
        fp8: 1,
        int4: 0.5,
    }
    
    if trt_type in mapping:
        return mapping[trt_type]


DataType.itemsize = property(lambda this: _itemsize(this))
