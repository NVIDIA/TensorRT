#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
from cuda.bindings import driver as cuda, runtime as cudart

class CudaStreamContext:
    """CUDA stream lifecycle management with context manager support"""
    def __init__(self):
        """Initialize CUDA stream"""
        self._stream = cuda_call(cudart.cudaStreamCreate())

    def __enter__(self):
        """Create CUDA stream when entering context (if not already created)"""
        if self._stream is None:
            self._stream = cuda_call(cudart.cudaStreamCreate())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroy CUDA stream when exiting context"""
        self.free()

    @property
    def stream(self) -> cudart.cudaStream_t:
        if self._stream is None:
            raise RuntimeError("Stream not created. Use 'with' statement.")
        return self._stream

    def synchronize(self):
        """Synchronize the stream"""
        if self._stream is None:
            raise RuntimeError("Stream not created. Use 'with' statement.")
        cuda_call(cudart.cudaStreamSynchronize(self._stream))

    def free(self):
        """Explicitly free the CUDA stream"""
        if self._stream is not None:
            try:
                cuda_call(cudart.cudaStreamDestroy(self._stream))
                self._stream = None
            except Exception as e:
                logging.warning(f"Failed to destroy CUDA stream: {e}")

    def __del__(self):
        """Cleanup stream on destruction"""
        if hasattr(self, '_stream') and self._stream is not None:
            self.free()

    def __str__(self):
        return f"CudaStreamContext: {self._stream}"

    def __repr__(self):
        return self.__str__()

def cuda_call(call):
    """Helper function to make CUDA calls and check for errors"""
    def _cudaGetErrorEnum(error):
        if isinstance(error, cuda.CUresult):
            err, name = cuda.cuGetErrorName(error)
            return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, cudart.cudaError_t):
            return cudart.cudaGetErrorName(error)[1]
        else:
            raise RuntimeError("Unknown error type: {}".format(error))

    err, res = call[0], call[1:]
    if err.value:
        raise RuntimeError(
            "CUDA error code={}({})".format(
                err.value, _cudaGetErrorEnum(err)
            )
        )
    if len(res) == 1:
        return res[0]
    elif len(res) == 0:
        return None
    else:
        return res

def getComputeCapacity(devID=0):
    major = cuda_call(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID))
    minor = cuda_call(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, devID))
    return (major, minor)

def memcpy_host_to_device_async(device_ptr: int, host_arr: np.ndarray, stream):
    """Wrapper for async host-to-device memory copy"""
    cuda_call(cudart.cudaMemcpyAsync(device_ptr, host_arr.ctypes.data, host_arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))


def memcpy_device_to_host_async(host_arr: np.ndarray, device_ptr: int, stream):
    """Wrapper for async device-to-host memory copy"""
    cuda_call(cudart.cudaMemcpyAsync(host_arr.ctypes.data, device_ptr, host_arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))

def memcpy_device_to_device_async(dst_device_ptr: int, src_device_ptr: int, nbytes: int, stream):
    """Wrapper for async device-to-device memory copy"""
    cuda_call(cudart.cudaMemcpyAsync(dst_device_ptr, src_device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))

def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    """Wrapper for synchronous host-to-device memory copy"""
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr.ctypes.data, host_arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    """Wrapper for synchronous device-to-host memory copy"""
    cuda_call(cudart.cudaMemcpy(host_arr.ctypes.data, device_ptr, host_arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

def memcpy_device_to_device(dst_device_ptr: int, src_device_ptr: int, nbytes: int):
    """Wrapper for synchronous device-to-device memory copy"""
    cuda_call(cudart.cudaMemcpy(dst_device_ptr, src_device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))

# Initialize CUDA
cuda_call(cudart.cudaFree(0))
