#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tensorrt as trt
import torch
import sys
import os
import time
import argparse
from cuda.bindings import runtime as cudart
from ctypes import py_object, pythonapi, c_void_p, c_char_p
from typing import Optional

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import nccl.core as nccl
except ImportError:
    nccl = None

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import common


def communicator_to_capsule(comm):
    """
    Convert nccl.core.Communicator to PyCapsule for TensorRT compatibility.
    
    Args:
        comm: nccl.core.Communicator instance with .ptr attribute set to ncclComm_t handle
        
    Returns:
        PyCapsule wrapping the communicator pointer, suitable for set_communicator()
        
    Raises:
        ValueError: If comm.ptr is invalid (0 or None), indicating destroyed communicator
        TypeError: If comm doesn't have a .ptr attribute
    """
    # Validate input
    if comm is None:
        raise TypeError("Communicator cannot be None")
    
    if not hasattr(comm, 'ptr'):
        raise TypeError(f"Object {type(comm)} does not have 'ptr' attribute. "
                       "Expected nccl.core.Communicator instance.")
    
    # Get the raw pointer from the Communicator object
    ptr = comm.ptr
    
    # Validate that communicator is still alive (ptr != 0)
    if ptr == 0:
        raise ValueError("NCCL Communicator has been destroyed (ptr=0). "
                        "Cannot create capsule for destroyed communicator.")
    
    # Convert to PyCapsule using ctypes.pythonapi
    PyCapsule_New = pythonapi.PyCapsule_New
    PyCapsule_New.restype = py_object
    PyCapsule_New.argtypes = [c_void_p, c_char_p, c_void_p]
    
    capsule = PyCapsule_New(c_void_p(ptr), b"ncclComm_t", None)
    
    if capsule is None:
        raise RuntimeError("Failed to create PyCapsule from communicator pointer")
    
    return capsule


def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None, output_shape: Optional[tuple] = None):
    """Allocate host and device buffers for TensorRT engine."""
    inputs = []
    outputs = []
    bindings = []
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # Pick out the max shape to allocate enough memory for the binding.
        shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
        # For dynamic shapes, use fixed output shape
        if output_shape is not None:
            shape = output_shape
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(binding)

        # Allocate host and device buffers
        if trt_type == trt.DataType.BF16:
            dtype = np.dtype(np.uint16)
            bindingMemory = common.HostDeviceMem(size, dtype)
        elif trt_type == trt.DataType.HALF:
            dtype = np.dtype(np.uint16)
            bindingMemory = common.HostDeviceMem(size, dtype)
        elif trt_type == trt.DataType.FLOAT:
            dtype = np.dtype(np.float32)
            bindingMemory = common.HostDeviceMem(size, dtype)
        else:
            try:
                dtype = np.dtype(trt.nptype(trt_type))
                bindingMemory = common.HostDeviceMem(size, dtype)
            except TypeError:
                size = int(size * trt_type.itemsize)
                bindingMemory = common.HostDeviceMem(size)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device_ptr))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
    return inputs, outputs, bindings


class AttentionSD:
    """Base class for Attention model using TensorRT (Single Device)"""
    
    def __init__(self, mpi_comm, rank, onnx_path):
        """
        Initialize the Attention class
        
        Args:
            mpi_comm: MPI communicator
            rank: Current instance ID
            onnx_path: Path to the ONNX model
        """
        self.onnx_path = onnx_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.mpi_comm = mpi_comm
        self.rank = rank

    def setup(self, actual_input_shape, output_shape):
        """
        Set up everything before doing inference.
        """
        engine_string = self.build_serialized_network()
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_string)
        if self.engine is None:
            print("Failed deserializing engine!")
            exit(-1)
        print("Succeeded deserializing engine!")

        self.context = self.engine.create_execution_context()
        
        # For dynamic shapes, we need to specify the actual input shape we want to use
        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, actual_input_shape)
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings = allocate_buffers(
            self.engine, profile_idx=0, output_shape=output_shape
        )
        
        num_io = self.engine.num_io_tensors
        tensor_names = [self.engine.get_tensor_name(i) for i in range(num_io)]
        for i in range(num_io):
            self.context.set_tensor_address(tensor_names[i], self.bindings[i])

    def build_serialized_network(self):
        """Create and serialize a network from the ONNX model."""
        # Create builder and empty network
        builder = trt.Builder(self.logger)
        network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        
        # Setup parser and parse the ONNX model
        print(f"Parsing ONNX model from {self.onnx_path}")
        parser = trt.OnnxParser(network, self.logger)

        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Get input dimensions and data type
        input_tensor = network.get_input(0)
        input_shape = input_tensor.shape
        input_name = input_tensor.name
        input_dtype = input_tensor.dtype
        
        print(f"[Rank {self.rank}] Input shape: {input_shape}")
        print(f"[Rank {self.rank}] Input name: {input_name}")
        print(f"[Rank {self.rank}] Input data type: {input_dtype}")

        # Create a builder config
        config = builder.create_builder_config()
        
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * 1024 * 1024 * 1024)  # 16GB workspace
        config.set_memory_pool_limit(trt.MemoryPoolType.TACTIC_SHARED_MEMORY, 1 * 1024 * 1024 * 1024)  # 1GB shared mem
        
        profile = builder.create_optimization_profile()
        
        # Set the shape range for the input tensor
        min_shape = (1, 1, 4096) 
        opt_shape = (56320, 1, 4096)
        max_shape = (56320, 1, 4096)
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        
        # Build the serialized network
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print(f"[Rank {self.rank}] Failed building serialized engine!")
            exit(-1)
        print(f"[Rank {self.rank}] Succeeded building serialized engine!")
        
        return serialized_engine
    
    def infer(self, input_data, output_shape, num_iterations):
        """
        Execute inference on the input data.
        
        Args:
            input_data: Input data for inference
            output_shape: Expected output shape for reshaping
            num_iterations: Number of inference iterations for averaging timing results
            
        Returns:
            output_data: List of output tensors
        """
        print(f"[Rank {self.rank}] Input shape: {input_data.shape}")
        
        # Copy input data to device
        for input_buffer in self.inputs:
            common.memcpy_host_to_device(input_buffer.device_ptr, input_data)
        
        # Warmup
        with common.CudaStreamContext() as stream:
            self.context.execute_async_v3(stream.stream)
            common.cuda_call(cudart.cudaStreamSynchronize(stream.stream))
        
            # Run inference
            start = time.time()
            for _ in range(num_iterations):
                self.context.execute_async_v3(stream.stream)
                common.cuda_call(cudart.cudaStreamSynchronize(stream.stream))
            end = time.time()
        
            print(f"[Rank {self.rank}] Time spent in TRT attention: {(end-start)/num_iterations * 1000} ms")
        
        # Get output
        output_data = []
        for output in self.outputs:
            common.memcpy_device_to_host(output.host, output.device_ptr)

            # Process based on data type
            if self.engine.get_tensor_dtype(self.engine.get_tensor_name(1)) == trt.DataType.BF16:
                numpy_output = np.frombuffer(output.host, dtype=np.uint16).reshape(output_shape)
                torch_output = torch.from_numpy(numpy_output).view(torch.bfloat16)
                torch_output = torch_output.reshape(output_shape)
            elif self.engine.get_tensor_dtype(self.engine.get_tensor_name(1)) == trt.DataType.HALF:
                numpy_output = np.frombuffer(output.host, dtype=np.float16).reshape(output_shape)
                torch_output = torch.from_numpy(numpy_output)
            else:
                numpy_output = np.frombuffer(output.host, dtype=np.float32).reshape(output_shape)
                torch_output = torch.from_numpy(numpy_output)
            
            output_data.append(torch_output)
                
        return output_data
    
    def cleanup(self):
        """
        Free the buffer resources.
        """
        common.free_buffers(self.inputs, self.outputs)


class AttentionMD(AttentionSD):
    """Multi-device Attention model using TensorRT with NCCL for communication"""
    
    def __init__(self, mpi_comm, num_ranks, rank, onnx_path):
        """
        Initialize the multi-device Attention class
        
        Args:
            mpi_comm: MPI communicator
            num_ranks: Number of instances/devices
            rank: Current instance ID
            onnx_path: Path to the ONNX model
        """
        super(AttentionMD, self).__init__(mpi_comm, rank, onnx_path)
        self.num_ranks = num_ranks
        self.nccl_comm = None

    def setup_multidevice(self, root):
        """
        Set up CUDA devices and initialize NCCL communicator.
        
        Args:
            root: Root rank for communication
        """
        assert nccl is not None
        assert root <= self.num_ranks - 1
        assert self.rank <= self.num_ranks - 1

        num_devices = common.cuda_call(cudart.cudaGetDeviceCount())
        assert num_devices >= self.num_ranks

        common.cuda_call(cudart.cudaSetDevice(self.rank))

        if self.rank == root:
            nccl_comm_id = nccl.get_unique_id()
        else:
            nccl_comm_id = None

        nccl_comm_id = self.mpi_comm.bcast(nccl_comm_id, root=root)
        self.nccl_comm = nccl.Communicator.init(nranks=self.num_ranks, rank=self.rank, unique_id=nccl_comm_id)

    def setup(self, actual_input_shape, output_shape, root=0):
        """
        Set up the multi-device environment and build/load the engine
        
        Args:
            root: Root rank for communication
        """
        self.setup_multidevice(root)
        
        # Load or build TRT engine
        if self.rank == root:
            engine_bin = bytes(self.build_serialized_network())
        else:
            engine_bin = None
        
        # Broadcast the serialized engine from root to all ranks
        engine_bin = self.mpi_comm.bcast(engine_bin, root=root)
        
        # Deserialize the engine
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(engine_bin)
        if self.engine is None:
            print(f"[Rank {self.rank}] Failed deserializing engine!")
            exit(-1)
        print(f"[Rank {self.rank}] Succeeded deserializing engine!")
        
        # Create an execution context
        self.context = self.engine.create_execution_context()
        
        # Set the NCCL communicator for multi-device communication
        capsule = communicator_to_capsule(self.nccl_comm)
        if not self.context.set_communicator(capsule):
            print(f"[Rank {self.rank}] Failed to set communicator")
            exit(-1)

        # For dynamic shapes, we need to specify the actual input shape we want to use
        input_name = self.engine.get_tensor_name(0)
        self.context.set_input_shape(input_name, actual_input_shape)
        
        # Allocate buffers for local portion of data
        self.inputs, self.outputs, self.bindings = allocate_buffers(
            self.engine, profile_idx=0, output_shape=output_shape
        )
        
        num_io = self.engine.num_io_tensors
        tensor_names = [self.engine.get_tensor_name(i) for i in range(num_io)]
        for i in range(num_io):
            self.context.set_tensor_address(tensor_names[i], self.bindings[i])


def generate_random_input(sequence_length, batch_size):
    """Generate random float16 input data with the given shape."""
    torch.manual_seed(42)
    torch_input = torch.rand((sequence_length, batch_size, 4096)).to(torch.float16)
    input_data = np.ascontiguousarray(torch_input.cpu().numpy())
    return input_data, (sequence_length, batch_size, 4096)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample script for Attention MDTRT")
    parser.add_argument("--onnx-path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--sequence-length", type=int, default=56320, help="Sequence length for input")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-iterations", type=int, default=50, help="Number of inference iterations for timing")
    parser.add_argument("--save-output", type=str, default=None, help="Save output tensor to .npy file (root rank only)")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize MPI if available
    if MPI is not None:
        mpi_comm = MPI.COMM_WORLD
        num_ranks = mpi_comm.Get_size()
        rank = mpi_comm.Get_rank()
        root = 0
    else:
        # Fallback for single-process execution
        mpi_comm = None
        num_ranks = 1
        rank = 0
        root = 0
    
    actual_input_shape = (args.sequence_length, args.batch_size, 4096)
    output_shape = (args.sequence_length, args.batch_size, 4096)
    
    # Print configuration
    if rank == root:
        print(f"[setup] Configuration:")
        print(f"[setup]   Number of GPUs: {num_ranks}")
        print(f"[setup]   Sequence Length: {args.sequence_length}")
        print(f"[setup]   Batch Size: {args.batch_size}")
        print(f"[setup]   Data Type: float16")
        print(f"[setup]   Input Shape: {actual_input_shape}")
        print(f"[setup]   Output Shape: {output_shape}")
    
    # Generate random input data with FULL sequence length (only on root rank)
    if rank == root:
        input_data, input_shape = generate_random_input(args.sequence_length, args.batch_size)
        print(f"[Rank {rank}] Generated random input data with shape: {input_shape}")
        
        if num_ranks == 1:
            print(f"[Rank {rank}] Running single-device inference...")
            try:
                attention_sd = AttentionSD(mpi_comm, rank, args.onnx_path)
                attention_sd.setup(actual_input_shape, output_shape)
                sd_output = attention_sd.infer(input_data, output_shape, args.num_iterations)[0]
                print(f"[Rank {rank}] Single-device inference completed")
                print(f"[Rank {rank}] Output shape: {sd_output.shape}")
                if args.save_output:
                    np.save(args.save_output, sd_output.float().cpu().numpy())
                    print(f"[Rank {rank}] Output saved to {args.save_output}")
                attention_sd.cleanup()
            except Exception as e:
                print(f"[Rank {rank}] Error in single-device inference: {e}")
                sys.exit(1)
    else:
        input_data = None
    
    # Broadcast full input data to all ranks for multi-device inference
    if MPI is not None and num_ranks > 1:
        input_data = mpi_comm.bcast(input_data, root=root)
    
    # Run multi-device inference if num_gpus > 1
    if num_ranks > 1:
        if MPI is None:
            print(f"Error: MPI is required for multi-GPU tests but not available. Ensure you run with mpirun.")
            sys.exit(1)
        
        if nccl is None:
            print(f"Error: nccl is required for multi-GPU tests but not available.")
            sys.exit(1)
        
        print(f"[Rank {rank}] Running multi-device inference...")
        try:
            attention_md = AttentionMD(mpi_comm, num_ranks, rank, args.onnx_path)
            attention_md.setup(actual_input_shape, output_shape, root)
            md_output = attention_md.infer(input_data, output_shape, args.num_iterations)[0]
            print(f"[Rank {rank}] Multi-device inference completed")
            print(f"[Rank {rank}] Output shape: {md_output.shape}")
            if rank == root and args.save_output:
                np.save(args.save_output, md_output.float().cpu().numpy())
                print(f"[Rank {rank}] Output saved to {args.save_output}")
            attention_md.cleanup()
        except Exception as e:
            print(f"[Rank {rank}] Error in multi-device inference: {e}")
            sys.exit(1)
    
    print(f"[Rank {rank}] Test completed successfully!")


if __name__ == "__main__":
    main()
