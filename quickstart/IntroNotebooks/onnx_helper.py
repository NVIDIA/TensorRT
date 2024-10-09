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

import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

# For ONNX:

class ONNXClassifierWrapper():
    def __init__(self, file, target_dtype = np.float32):
        
        self.target_dtype = target_dtype
        self.num_classes = 1000
        self.load(file)
        
        self.stream = None
      
    def load(self, file):
        f = open(file, "rb")
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

        self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype = self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        
        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        assert(len(tensor_names) == 2)

        self.context.set_tensor_address(tensor_names[0], int(self.d_input))
        self.context.set_tensor_address(tensor_names[1], int(self.d_output))

        self.stream = cuda.Stream()
        
    def predict(self, batch): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)
            
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v3(self.stream.handle)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()
        
        return self.output

def convert_onnx_to_engine(onnx_filename, engine_filename = None, max_workspace_size = 1 << 30, fp16_mode = True):
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, builder.create_network() as network, trt.OnnxParser(network, logger) as parser, builder.create_builder_config() as builder_config:
        builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        if (fp16_mode):
            builder_config.set_flag(trt.BuilderFlag.FP16)

        print("Parsing ONNX file.")
        with open(onnx_filename, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print("Building TensorRT engine. This may take a few minutes.")
        serialized_engine = builder.build_serialized_network(network, builder_config)

        if engine_filename:
            with open(engine_filename, 'wb') as f:
                f.write(serialized_engine)

        return serialized_engine, logger
