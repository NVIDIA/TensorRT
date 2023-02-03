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
import glob
import os
import sys
import warnings


def try_load(library):
    try:
        ctypes.CDLL(library)
    except OSError:
        pass


# Try loading all packaged libraries. This is a nop if there are no libraries packaged.
CURDIR = os.path.realpath(os.path.dirname(__file__))
for lib in glob.iglob(os.path.join(CURDIR, "*.so*")):
    try_load(lib)


# On Windows, we need to manually open the TensorRT libraries - otherwise we are unable to
# load the bindings.
def find_lib(name):
    paths = os.environ["PATH"].split(os.path.pathsep)
    for path in paths:
        libpath = os.path.join(path, name)
        if os.path.isfile(libpath):
            return libpath

    raise FileNotFoundError(
        "Could not find: {:}. Is it on your PATH?\nNote: Paths searched were:\n{:}".format(name, paths)
    )


if sys.platform.startswith("win"):
    # Order matters here because of dependencies
    LIBRARIES = [
        "nvinfer.dll",
        "cublas64_##CUDA_MAJOR##.dll",
        "cublasLt64_##CUDA_MAJOR##.dll",
        "cudnn64_##CUDNN_MAJOR##.dll",
        "nvinfer_plugin.dll",
        "nvonnxparser.dll",
        "nvparsers.dll",
    ]

    for lib in LIBRARIES:
        ctypes.CDLL(find_lib(lib))


from .tensorrt import *

__version__ = "##TENSORRT_VERSION##"


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

Builder.__enter__ = common_enter
Builder.__exit__ = common_exit

ICudaEngine.__enter__ = common_enter
ICudaEngine.__exit__ = common_exit

IExecutionContext.__enter__ = common_enter
IExecutionContext.__exit__ = common_exit

Runtime.__enter__ = common_enter
Runtime.__exit__ = common_exit

INetworkDefinition.__enter__ = common_enter
INetworkDefinition.__exit__ = common_exit

UffParser.__enter__ = common_enter
UffParser.__exit__ = common_exit

CaffeParser.__enter__ = common_enter
CaffeParser.__exit__ = common_exit

OnnxParser.__enter__ = common_enter
OnnxParser.__exit__ = common_exit

IHostMemory.__enter__ = common_enter
IHostMemory.__exit__ = common_exit

Refitter.__enter__ = common_enter
Refitter.__exit__ = common_exit

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
        bool: np.bool_,
        uint8: np.uint8,
    }
    if trt_type in mapping:
        return mapping[trt_type]
    raise TypeError("Could not resolve TensorRT datatype to an equivalent numpy datatype.")


# Add a numpy-like itemsize property to the datatype.
def _itemsize(trt_type):
    """
    Returns the size in bytes of this :class:`DataType` .

    :arg trt_type: The TensorRT data type.

    :returns: The size of the type.
    """
    mapping = {
        float32: 4,
        float16: 2,
        int8: 1,
        int32: 4,
        bool: 1,
        uint8: 1,
    }
    if trt_type in mapping:
        return mapping[trt_type]


DataType.itemsize = property(lambda this: _itemsize(this))
