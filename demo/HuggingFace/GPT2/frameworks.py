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

import os
import sys
import argparse

from typing import List

# huggingface
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
)

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# helpers
from NNDF.interface import FrameworkCommand
from NNDF.general_utils import confirm_folder_delete, NNFolderWorkspace
from NNDF.networks import (
    NetworkResult,
    NetworkMetadata,
    NetworkRuntime,
    Precision,
    NetworkModel,
    NetworkModels,
    TimingProfile,
)
from GPT2.export import GPT2TorchFile
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
from GPT2.measurements import gpt2_inference, full_inference_greedy


class GPT2HuggingFace(FrameworkCommand):
    def __init__(self):
        super().__init__(
            GPT2ModelTRTConfig, description="Runs framework results for GPT2 model."
        )

        # Default inference input used during inference stage
        self.onnx_gpt2 = None
        self.torch_gpt2_dir = None

    def generate_and_download_framework(
        self, metadata: NetworkMetadata, workspace: NNFolderWorkspace
    ) -> NetworkModels:

        cache_variant = False
        if metadata.other.kv_cache:
            cache_variant = True

        trt_gpt2_config = self.config
        metadata_serialized = trt_gpt2_config.get_metadata_string(metadata)
        workspace_dir = workspace.get_path()

        pytorch_model_dir = os.path.join(workspace_dir, metadata_serialized)
        # We keep track of the generated torch location for cleanup later
        self.torch_gpt2_dir = pytorch_model_dir

        model = None
        tfm_config = GPT2Config(use_cache=cache_variant)

        if not os.path.exists(pytorch_model_dir):
            # Generate the pre-trained weights
            model = GPT2LMHeadModel(tfm_config).from_pretrained(metadata.variant)
            model.save_pretrained(pytorch_model_dir)
            print("Pytorch Model saved to {}".format(pytorch_model_dir))
        else:
            print(
                "Frameworks file already exists, skipping generation and loading from file instead."
            )
            model = GPT2LMHeadModel(tfm_config).from_pretrained(pytorch_model_dir)

        root_onnx_model_name = "{}.onnx".format(metadata_serialized)
        root_onnx_model_fpath = os.path.join(
            os.getcwd(), workspace_dir, root_onnx_model_name
        )
        onnx_model_fpath = root_onnx_model_fpath

        gpt2 = GPT2TorchFile(model, metadata)
        self.onnx_gpt2 = gpt2.as_onnx_model(onnx_model_fpath, force_overwrite=False)

        onnx_models = [
            NetworkModel(
                name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=self.onnx_gpt2.fpath,
            )
        ]
        torch_models = [
            NetworkModel(
                name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=pytorch_model_dir,
            )
        ]

        return NetworkModels(torch=torch_models, onnx=onnx_models, trt=None)

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        save_onnx_model: bool = True,
        keep_pytorch_model: bool = True,
    ) -> None:
        """
        Cleans up the working directory and leaves models if available.
        Should not assume any functions from the framework class has been called.
        Returns:
            None
        """
        # Clean-up generated files
        if not save_onnx_model and self.onnx_gpt2 is not None:
            self.onnx_gpt2.cleanup()

            # Remove any onnx external files by removing integer named values and weight files
            workspace_path = workspace.get_path()
            for d in os.listdir(workspace_path):
                fpath = os.path.join(workspace_path, d)
                if os.path.isfile(fpath) and os.path.splitext(d)[1] == ".weight":
                    os.remove(fpath)
                elif d.isnumeric():
                    os.remove(fpath)

        if not keep_pytorch_model:
            # Using rmtree can be dangerous, have user confirm before deleting.
            confirm_folder_delete(
                self.torch_gpt2_dir,
                prompt="Confirm you want to delete downloaded pytorch model folder?",
            )

        if not keep_pytorch_model and not save_onnx_model:
            workspace.cleanup(force_remove=False)

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        network_fpaths: NetworkModels,
        inference_input: str,
        timing_profile: TimingProfile,
    ) -> NetworkResult:

        # Execute some tests
        tokenizer = GPT2Tokenizer.from_pretrained(metadata.variant)
        input_ids = tokenizer(inference_input, return_tensors="pt").input_ids

        # By default, HuggingFace model structure is one giant file.
        gpt2_torch_fpath = network_fpaths.torch[0].fpath
        config = GPT2Config(use_cache=metadata.other.kv_cache)
        gpt2_model = GPT2LMHeadModel(config).from_pretrained(gpt2_torch_fpath)
        gpt2_torch = GPT2TorchFile.TorchModule(
            gpt2_model.transformer, gpt2_model.lm_head, gpt2_model.config
        )
        greedy_output = gpt2_torch.generate(input_ids) #greedy search

        # get single decoder iteration inference timing profile
        _, decoder_e2e_median_time = gpt2_inference(
            gpt2_torch, input_ids, timing_profile
        )

        # get complete decoder inference result and its timing profile
        sample_output, full_e2e_median_runtime = full_inference_greedy(
            gpt2_torch,
            input_ids,
            timing_profile,
            max_length=GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
        )

        semantic_outputs = []
        for i, sample_output in enumerate(sample_output):
            semantic_outputs.append(
                tokenizer.decode(sample_output, skip_special_tokens=True)
            )

        return NetworkResult(
            input=inference_input,
            output_tensor=greedy_output,
            semantic_output=semantic_outputs,
            median_runtime=[
                NetworkRuntime(
                    name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    runtime=decoder_e2e_median_time,
                ),
                NetworkRuntime(
                    name=GPT2ModelTRTConfig.NETWORK_FULL_NAME,
                    runtime=full_e2e_median_runtime,
                ),
            ],
            models=network_fpaths,
        )

    def run_framework(
        self,
        metadata: NetworkMetadata,
        network_input: List[str],
        working_directory: str,
        keep_onnx_model: bool,
        keep_pytorch_model: bool,
        timing_profile: TimingProfile,
    ) -> List[NetworkResult]:
        """
        Main entry point of our function which compiles and generates our model data.
        """
        results = []
        workspace = NNFolderWorkspace(
            self.config.network_name, metadata, working_directory
        )
        try:
            network_fpaths = self.generate_and_download_framework(metadata, workspace)
            for ninput in network_input:
                results.append(
                    self.execute_inference(
                        metadata, network_fpaths, ninput, timing_profile
                    )
                )
        finally:
            self.cleanup(workspace, keep_onnx_model, keep_pytorch_model)

        return results

    def args_to_network_metadata(self, args: argparse.Namespace) -> NetworkMetadata:
        return NetworkMetadata(
            variant=args.variant,
            precision=Precision(fp16=False),
            other=self.config.MetadataClass(kv_cache=args.enable_kv_cache),
        )


# Entry point
RUN_CMD = GPT2HuggingFace()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
