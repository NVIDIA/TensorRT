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

import os
import sys
import onnx
import argparse
import numpy as np
from PIL import Image

import onnxruntime
import tensorrt as trt
from modelopt.onnx.autocast import convert_to_mixed_precision
from polygraphy.backend.onnxrt import OnnxrtRunner
from polygraphy.backend.trt import TrtRunner

class Sample:
    def __init__(self, mnist_dir, working_dir):
        if not os.path.exists(mnist_dir):
            print(f"ERROR: {mnist_dir} is not existed.")
            sys.exit(1)            
        if not os.path.exists(working_dir):
            print(f"ERROR: {working_dir} is not existed.")
            sys.exit(1)   
        self.mnist_dir = mnist_dir
        self.working_dir = working_dir
        self.origin_onnx_path = os.path.join(self.mnist_dir, "mnist.onnx")
        self.converted_onnx_path = os.path.join(self.working_dir, "mnist_converted.onnx")
        self.trt_engine_path = os.path.join(self.working_dir, "mnist_converted.engine")
        self.origin_output_path = os.path.join(self.working_dir, "origin_output.npz")
        self.converted_output_path = os.path.join(self.working_dir, "converted_output.npz")
        self.trt_output_path = os.path.join(self.working_dir, "trt_output.npz")

    # Prepare input data in npz format.
    def prepare_input_npz(self):
        print("Start prepare input data in npz format.")
        for i in range(10):
            image = Image.open(os.path.join(self.mnist_dir, str(i) + ".pgm"))
            image_np = np.array(image)
            image_np = 1.0 - (image_np / 255.0)
            image_np_reshape = image_np.reshape(1, 1, 28, 28).astype(np.float32)
            file_path = os.path.join(self.working_dir, str(i) + ".npz")
            np.savez(file_path, Input3=image_np_reshape)
            print(f"Saved input number {i} to {file_path}.")

    # Convert fp32 onnx model to fp32-fp16 mixed precision
    def convert_model(self):
        fp32_nodes = ["Plus214"]
        fp32_op_types = ["MatMul"]
        print(f"Start autocast on {self.origin_onnx_path}.")
        print("Use 8.npz as input data file for reference runner.")
        calibration_path = os.path.join(self.working_dir, "8.npz")
        converted_model = convert_to_mixed_precision(
            onnx_path=self.origin_onnx_path,                        # Path to the input ONNX model.
            low_precision_type="fp16",                              # Target precision to reduce to ('fp16' or 'bf16').
            nodes_to_exclude=fp32_nodes,                            # List of regex patterns to match node names that should remain in FP32.
            op_types_to_exclude=fp32_op_types,                      # List of operation types that should remain in FP32.
            data_max=8.0,                                           # Maximum absolute value for node input and output values.
            init_max=8.0,                                           # Maximum absolute value for initializers.
            keep_io_types=True,                                     # Whether to preserve input/output types.
            calibration_data=calibration_path,                      # Path to input data file for reference runner.
        )
        onnx.save(converted_model, self.converted_onnx_path)
        print(f"Saved converted model to {self.converted_onnx_path}, autocast stage finished.")

    # Infer with runner (onnxrt or trtrt)
    def infer_with_runner(self, runner, output_file_name):
        all_output = []
        all_result = True
        with runner:
            for i in range(10):
                input = np.load(os.path.join(self.working_dir, str(i) + ".npz"))
                output = runner.infer(feed_dict=input)
                all_output.append(output["Plus214_Output_0"].copy())
                max_index = np.argmax(output["Plus214_Output_0"])
                if max_index != i:
                    all_result = False
                result = "passed" if max_index == i else "failed"      
                print(f"Input number is {i}, inferred number is {max_index}, {result}.")
        if all_result:
            print("Number infer test passed.")
        else:
            print("ERROR: Number infer test failed.")
            sys.exit(1)
        output_path = os.path.join(self.working_dir, output_file_name)
        output_data = np.concatenate(all_output, axis=0)
        np.savez(output_path, Plus214_Output_0=output_data)
        print(f"Saved all outputs to {output_path}, infer stage finished.")    

    # Check if outputs meet tolerances
    def check_accuracy(self, output1_path, output2_path, r_tol=5e-3, a_tol=5e-3):
        print(f" Comparing '{output1_path}' and '{output2_path}' with rtol({r_tol})+a_tol({a_tol})...")
        output1 = np.load(output1_path)
        output2 = np.load(output2_path)
        result = np.allclose(output1["Plus214_Output_0"], output2["Plus214_Output_0"], rtol=r_tol, atol=a_tol)
        if result:
            print("Outputs and references matches.")
        else:
            print("ERROR: Outputs and references mismatches.")
            sys.exit(1)

    # Build trt engine with strongly_type
    def build_trt_engine(self):
        print(f"Start build trt engine on {self.converted_onnx_path}.")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        config = builder.create_builder_config()

        parser = trt.OnnxParser(network, TRT_LOGGER)
        with open(self.converted_onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                sys.exit(1)

        engine = builder.build_engine_with_config(network, config)
        with open(self.trt_engine_path, "wb") as f:
            f.write(engine.serialize())
            print(f"Saved trt engine to {self.trt_engine_path}.")

    # Run onnxruntime infer stage
    def infer_with_onnxrt(self, onnx_path, output_file_name):
        print(f"Start onnxruntime infer on {onnx_path}.")
        sess = onnxruntime.InferenceSession(onnx_path)
        runner = OnnxrtRunner(sess)
        self.infer_with_runner(runner, output_file_name)

    # Run trtruntime infer stage
    def infer_with_trt(self, output_file_name):
        print(f"Start trtruntime infer on {self.trt_engine_path}.")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.trt_engine_path, 'rb') as f:
            engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            runner = TrtRunner(engine)
            self.infer_with_runner(runner, output_file_name)

    def run(self):
        print("STAGE 1: Prepare input data, do inference with onnx runtime on original fp32 onnx model.")
        self.prepare_input_npz()
        self.infer_with_onnxrt(self.origin_onnx_path, self.origin_output_path)

        print("STAGE 2: Convert original fp32 onnx model to fp32-fp16 mixed onnx model, do inference with onnx runtime on converted onnx model, and check accuracy.")        
        self.convert_model()
        self.infer_with_onnxrt(self.converted_onnx_path, self.converted_output_path)
        self.check_accuracy(self.origin_output_path, self.converted_output_path)

        print("STAGE 3: Build trt engine on the converted onnx model with strong typing mode, do inference with trt runtime on the engine, and check accuracy.") 
        self.build_trt_engine()
        self.infer_with_trt(self.trt_output_path)
        self.check_accuracy(self.origin_output_path, self.trt_output_path)
        
        print("Sample finished successfully.")

def main():
    parser = argparse.ArgumentParser(
        description="Sample to convert FP32 ONNX model to mixed FP32-FP16 precision for TensorRT strong typing usage.",
    )
    parser.add_argument(
        "--mnist_dir",
        type=str,
        required=True,
        help="Path to the MNIST directory.",
    )
    parser.add_argument(
        "--working_dir",
        type=str,
        default=os.path.abspath("."),
        help="Path to the working directory. Defaults to current directory.",
    )    
    args = parser.parse_args()

    sample = Sample(args.mnist_dir, args.working_dir)
    sample.run()

if __name__ == "__main__":
    main()
