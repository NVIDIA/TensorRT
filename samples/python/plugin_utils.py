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

from cuda.bindings import driver as cuda, runtime as cudart, nvrtc

import numpy as np
import os
from common_runtime import cuda_call, create_cuda_context, cuda_init, cuda_get_device, cuda_memcpy_htod
import argparse
import threading

import tensorrt as trt
import cupy as cp


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Options for Circular Padding plugin C++ example"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Precision to use for plugin",
    )

    return parser.parse_args()


def volume(d):
    return np.prod(d)




def getComputeCapacity(devID):
    major = cuda_call(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, devID))
    minor = cuda_call(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, devID))
    return (major, minor)


# Taken from https://github.com/NVIDIA/cuda-python/blob/main/examples/common/common.py
class KernelHelper:
    def __init__(self, code, devID):
        prog = cuda_call(
            nvrtc.nvrtcCreateProgram(str.encode(code), b"sourceCode.cu", 0, [], [])
        )
        CUDA_HOME = os.getenv("CUDA_HOME")
        if CUDA_HOME == None:
            CUDA_HOME = os.getenv("CUDA_PATH")
        if CUDA_HOME == None:
            raise RuntimeError("Environment variable CUDA_HOME or CUDA_PATH is not set")
        include_dirs = os.path.join(CUDA_HOME, "include")

        # Initialize CUDA
        cuda_call(cudart.cudaFree(0))

        major, minor = getComputeCapacity(devID)
        _, nvrtc_minor = cuda_call(nvrtc.nvrtcVersion())
        use_cubin = nvrtc_minor >= 1
        prefix = "sm" if use_cubin else "compute"
        arch_arg = bytes(f"--gpu-architecture={prefix}_{major}{minor}", "ascii")

        try:
            opts = [
                b"--fmad=true",
                arch_arg,
                "--include-path={}".format(include_dirs).encode("UTF-8"),
                b"--std=c++11",
                b"-default-device",
            ]
            cuda_call(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        except RuntimeError as err:
            logSize = cuda_call(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * logSize
            cuda_call(nvrtc.nvrtcGetProgramLog(prog, log))
            print(log.decode())
            print(err)
            exit(-1)

        if use_cubin:
            dataSize = cuda_call(nvrtc.nvrtcGetCUBINSize(prog))
            data = b" " * dataSize
            cuda_call(nvrtc.nvrtcGetCUBIN(prog, data))
        else:
            dataSize = cuda_call(nvrtc.nvrtcGetPTXSize(prog))
            data = b" " * dataSize
            cuda_call(nvrtc.nvrtcGetPTX(prog, data))

        self.module = cuda_call(cuda.cuModuleLoadData(np.char.array(data)))

    def getFunction(self, name):
        return cuda_call(cuda.cuModuleGetFunction(self.module, name))


class CudaCtxManager(trt.IPluginResource):
    def __init__(self, device=None):
        trt.IPluginResource.__init__(self)
        self.device = device
        self.cuda_ctx = None

    def clone(self):
        cloned = CudaCtxManager()
        cloned.__dict__.update(self.__dict__)
        # Delay the CUDA ctx creation until clone()
        # since only a cloned resource is registered by TRT
        cloned.cuda_ctx = create_cuda_context(self.device)
        return cloned

    def release(self):
        cuda_call(cuda.cuCtxDestroy(self.cuda_ctx))

class UnownedMemory:
    def __init__(self, ptr, shape, dtype):
        mem = cp.cuda.UnownedMemory(ptr, volume(shape) * cp.dtype(dtype).itemsize, self)
        cupy_ptr = cp.cuda.MemoryPointer(mem, 0)
        self.d = cp.ndarray(shape, dtype=dtype, memptr=cupy_ptr)
