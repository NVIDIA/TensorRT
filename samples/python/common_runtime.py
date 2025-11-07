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

import ctypes
from typing import Optional, List, Union

import numpy as np
import tensorrt as trt

from cuda.bindings import driver as cuda, runtime as cudart, nvrtc


class ArrayWithOwner(np.ndarray):
    """Numpy array that holds a reference to its owner object"""
    def __new__(cls, input_array, owner):
        obj = np.asarray(input_array).view(cls)
        obj._owner = owner
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._owner = getattr(obj, '_owner', None)



def cuda_call(call):
    """Helper function to make CUDA calls and check for errors"""
    def _cudaGetErrorEnum(error):
        if isinstance(error, cuda.CUresult):
            err, name = cuda.cuGetErrorName(error)
            return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
        elif isinstance(error, cudart.cudaError_t):
            return cudart.cudaGetErrorName(error)[1]
        elif isinstance(error, nvrtc.nvrtcResult):
            return nvrtc.nvrtcGetErrorString(error)[1]
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


def create_cuda_context(device):
    """
    Create CUDA context with version-aware API handling.

    Handles different CUDA API versions based on actual documented signatures:
    - CUDA 11.8-12.9: cuCtxCreate(flags, device) - 2 arguments
    - CUDA 13.0+: cuCtxCreate(ctxCreateParams, flags, device) - 3 arguments

    Args:
        device: CUDA device handle from cuDeviceGet

    Returns:
        CUDA context handle
    """
    # Try different API versions
    try:
        # Try CUDA 13.0+ API first (3 arguments with ctxCreateParams)
        # cuCtxCreate(ctxCreateParams, flags, device)
        return cuda_call(cuda.cuCtxCreate(None, 0, device))
    except TypeError:
        # CUDA 11.8-12.9 API: cuCtxCreate(flags, device)
        return cuda_call(cuda.cuCtxCreate(0, device))



class HostDeviceMem:
    """Pair of host and device memory using RAII composition"""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        if dtype is None:
            dtype = np.dtype(np.uint8)
        else:
            dtype = np.dtype(dtype)
        self._size = size
        self._dtype = dtype

        # Use RAII classes for memory management
        self._host_mem = PinnedHostMem(size, dtype)
        self._device_mem = DeviceMem(size * dtype.itemsize)

    @property
    def host(self) -> np.ndarray:
        # Return the array directly - ArrayWithOwner ensures proper lifetime management
        return self._host_mem.array

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        # Delegate to PinnedHostMem for proper data handling
        self._host_mem.array = data

    @property
    def device_ptr(self) -> int:
        """Device memory pointer"""
        return self._device_mem.device_ptr

    @property
    def nbytes(self) -> int:
        return self._host_mem.nbytes


    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device_ptr}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()


class DeviceMem:
    """Device-only memory allocation for cases where host memory is not needed"""
    def __init__(self, size: int):
        self._device_ptr = cuda_call(cudart.cudaMalloc(size))
        self._nbytes = size

    @property
    def device_ptr(self) -> int:
        """Device memory pointer"""
        return self._device_ptr

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def free(self):
        """Explicitly free device memory"""
        if self._device_ptr is not None:
            try:
                cuda_call(cudart.cudaFree(self._device_ptr))
                self._device_ptr = None
            except Exception:
                # Log but don't raise - cleanup should be best effort
                pass

    def __str__(self):
        return f"Device:\n{self.device_ptr}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        # Fallback cleanup - not guaranteed to be called
        self.free()


class PinnedHostMem:
    """Pinned host memory allocation for faster GPU transfers"""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        if dtype is None:
            dtype = np.dtype(np.uint8)
        else:
            dtype = np.dtype(dtype)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))

        self._host_ptr = host_mem
        self._host_size = size
        self._nbytes = nbytes
        self._dtype = dtype

    @property
    def array(self) -> np.ndarray:
        # Create view with proper memory ownership
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(self._dtype))
        host_array = np.ctypeslib.as_array(ctypes.cast(self._host_ptr, pointer_type), (self._host_size,))
        return ArrayWithOwner(host_array, self)

    @array.setter
    def array(self, data: Union[np.ndarray, bytes]):
        """Set the array data with proper bounds checking"""
        host_array = self.array  # Get the numpy array view
        if isinstance(data, np.ndarray):
            if data.size > self._host_size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self._host_size}"
                )
            np.copyto(host_array[:data.size], data.flat, casting='safe')
        else:
            assert self._dtype == np.uint8
            host_array[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)


    @property
    def nbytes(self) -> int:
        return self._nbytes

    def free(self):
        """Explicitly free pinned host memory"""
        if self._host_ptr is not None:
            try:
                cuda_call(cudart.cudaFreeHost(self._host_ptr))
                self._host_ptr = None
            except Exception:
                # Log but don't raise - cleanup should be best effort
                pass

    def __str__(self):
        return f"PinnedHost:\n{self.array}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        # Fallback cleanup - not guaranteed to be called
        self.free()

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
        if self._stream is not None:
            try:
                cuda_call(cudart.cudaStreamDestroy(self._stream))
            except Exception:
                # Silently handle cleanup failures
                pass
            self._stream = None

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
            except Exception:
                # Log but don't raise - cleanup should be best effort
                pass

    def __del__(self):
        """Cleanup stream on destruction"""
        if hasattr(self, '_stream') and self._stream is not None:
            self.free()

    def __str__(self):
        return f"CudaStreamContext: {self._stream}"

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
    inputs = []
    outputs = []
    bindings = []
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(binding)

        # Allocate host and device buffers
        try:
            dtype = np.dtype(trt.nptype(trt_type))
            bindingMemory = HostDeviceMem(size, dtype)
        except TypeError: # no numpy support: create a byte array instead (BF16, FP8, INT4)
            size = int(size * trt_type.itemsize)
            bindingMemory = HostDeviceMem(size)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device_ptr))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
    return inputs, outputs, bindings


# Frees the resources allocated in allocate_buffers
def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem]):
    """
    Explicitly free CUDA memory resources.

    While __del__ methods provide automatic cleanup, they are not guaranteed to be called.
    This function provides explicit resource management for critical applications.
    """
    for inp in inputs:
        if hasattr(inp, '_device_mem') and hasattr(inp._device_mem, 'free'):
            inp._device_mem.free()
        if hasattr(inp, '_host_mem') and hasattr(inp._host_mem, 'free'):
            inp._host_mem.free()

    for out in outputs:
        if hasattr(out, '_device_mem') and hasattr(out._device_mem, 'free'):
            out._device_mem.free()
        if hasattr(out, '_host_mem') and hasattr(out._host_mem, 'free'):
            out._host_mem.free()


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr.ctypes.data, host_arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    cuda_call(cudart.cudaMemcpy(host_arr.ctypes.data, device_ptr, host_arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


# Additional CUDA wrapper functions for common operations


def cuda_init():
    """Initialize CUDA driver API with error checking."""
    cuda_call(cuda.cuInit(0))


def cuda_get_device(device_id: int = 0):
    """Get CUDA device handle with error checking."""
    return cuda_call(cuda.cuDeviceGet(device_id))


# CUDA Runtime API functions (preferred over driver API when available)


def cuda_memcpy_htod(device_ptr: int, host_data: np.ndarray):
    """Copy data from host to device using CUDA runtime API with error checking.

    Note: Consider using HostDeviceMem.host setter for integrated memory management.
    """
    cuda_call(cudart.cudaMemcpy(device_ptr, host_data, host_data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


def _do_inference_base(inputs, outputs, stream, execute_async_func):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [cuda_call(cudart.cudaMemcpyAsync(inp.device_ptr, inp.host.ctypes.data, inp.nbytes, kind, stream)) for inp in inputs]
    # Run inference.
    execute_async_func()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [cuda_call(cudart.cudaMemcpyAsync(out.host.ctypes.data, out.device_ptr, out.nbytes, kind, stream)) for out in outputs]
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host.copy() for out in outputs]


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, engine, bindings, inputs, outputs, stream):
    """
    Perform inference using the provided context and stream.

    Usage with context manager:
        with stream:  # Ensures proper stream lifecycle
            outputs = do_inference(context, engine, bindings, inputs, outputs, stream)
    """
    stream_handle = stream.stream
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream_handle)
    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream_handle, execute_async_func)
