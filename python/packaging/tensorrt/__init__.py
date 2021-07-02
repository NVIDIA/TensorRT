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
import glob
import os
import warnings


def try_load(library):
    try:
        ctypes.CDLL(library)
    except OSError:
        pass


# Try loading all packaged libraries
CURDIR = os.path.realpath(os.path.dirname(__file__))
for lib in glob.iglob(os.path.join(CURDIR, "*.so*")):
    try_load(lib)


from .tensorrt import *

__version__ = "##TENSORRT_VERSION##"


# Provides Python's `with` syntax
def common_enter(this):
    warnings.warn("Context managers for TensorRT types are deprecated. "
                  "Memory will be freed automatically when the reference count reaches 0.",
                  DeprecationWarning)
    return this


def common_exit(this, exc_type, exc_value, traceback):
    """
    Context managers are deprecated and have no effect. Objects are automatically freed when
    the reference count reaches 0.
    """
    pass


# Logger does not have a destructor.
ILogger.__enter__ = common_enter
ILogger.__exit__ = lambda this, exc_type, exc_value, traceback : None

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
    '''
    Returns the numpy-equivalent of a TensorRT :class:`DataType` .

    :arg trt_type: The TensorRT data type to convert.

    :returns: The equivalent numpy type.
    '''
    import numpy as np
    mapping = {
        float32: np.float32,
        float16: np.float16,
        int8: np.int8,
        int32: np.int32,
        bool: np.bool,
    }
    if trt_type in mapping:
        return mapping[trt_type]
    raise TypeError("Could not resolve TensorRT datatype to an equivalent numpy datatype.")


# Add a numpy-like itemsize property to the datatype.
def _itemsize(trt_type):
    '''
    Returns the size in bytes of this :class:`DataType` .

    :arg trt_type: The TensorRT data type.

    :returns: The size of the type.
    '''
    mapping = {
        float32: 4,
        float16: 2,
        int8: 1,
        int32: 4,
        bool: 1,
    }
    if trt_type in mapping:
        return mapping[trt_type]

DataType.itemsize = property(lambda this: _itemsize(this))
