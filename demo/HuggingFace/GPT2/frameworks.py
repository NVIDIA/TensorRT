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

import os
import sys
import argparse

from typing import List, Union

# huggingface
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    # GPT-J uses GPT2 tokenizer
    GPT2Tokenizer,
)

# torch
import torch

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# helpers
from NNDF.interface import FrameworkCommand
from NNDF.general_utils import confirm_folder_delete, NNFolderWorkspace
from NNDF.networks import (
    BenchmarkingResult,
    NetworkResult,
    NetworkMetadata,
    NetworkRuntime,
    Precision,
    NetworkModel,
    NetworkModels,
    TimingProfile,
)
from GPT2.export import GPT2TorchFile
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig, GPT2BenchmarkingArgs
from GPT2.measurements import gpt2_inference, full_inference, calculate_perplexity


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
        workspace_dir, _ , onnx_root = workspace.set_model_path(metadata_serialized, is_encoder_decoder = False)
        pytorch_model_dir = os.path.join(workspace_dir, "pytorch_model")
        # We keep track of the generated torch location for cleanup later
        self.torch_gpt2_dir = pytorch_model_dir

        model = None
        tfm_config = AutoConfig.from_pretrained(metadata.variant, use_cache=cache_variant)

        if not os.path.exists(pytorch_model_dir):
            # Generate the pre-trained weights
            model = AutoModelForCausalLM.from_config(tfm_config).from_pretrained(metadata.variant)
            model.config.use_cache = cache_variant # somehow the use_cache config automatically set to True even though specified in tfm_config before. Force change
            model.save_pretrained(pytorch_model_dir)
            print("Pytorch Model saved to {}".format(pytorch_model_dir))
        else:
            print(
                "Frameworks file already exists, skipping generation and loading from file instead."
            )
            model = AutoModelForCausalLM.from_config(tfm_config).from_pretrained(pytorch_model_dir)
            model.config.use_cache = cache_variant # somehow the use_cache config automatically set to True even though specified in tfm_config before. Force change
        
        onnx_model_fpath = os.path.join(onnx_root, metadata_serialized + ".onnx")

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

        if not keep_pytorch_model:
            # Using rmtree can be dangerous, have user confirm before deleting.
            confirm_folder_delete(
                self.torch_gpt2_dir,
                prompt="Confirm you want to delete downloaded pytorch model folder?",
            )

        if not keep_pytorch_model and not save_onnx_model:
            workspace.cleanup(force_remove=False)

    def setup_tokenizer_and_model(
        self,
        metadata: NetworkMetadata,
        network_fpaths: NetworkModels,
    ):
        tokenizer = GPT2Tokenizer.from_pretrained(metadata.variant)

        # GPT2 has no proper token set. Use custom token. Only "generate()" will auto
        # replace with EOS token when using generating mode
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # By default, HuggingFace model structure is one giant file.
        gpt2_torch_fpath = network_fpaths.torch[0].fpath
        config = AutoConfig.from_pretrained(metadata.variant, use_cache=metadata.other.kv_cache)
        gpt2_model = AutoModelForCausalLM.from_config(config).from_pretrained(gpt2_torch_fpath)
        gpt2_torch = GPT2TorchFile.TorchModule(
            gpt2_model.transformer, gpt2_model.lm_head, gpt2_model.config
        )

        return tokenizer, gpt2_torch

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        network_fpaths: NetworkModels,
        inference_input: str,
        timing_profile: TimingProfile,
        use_cpu: bool,
        batch_size: int = 1,
        num_beams: int = 1,
        benchmarking_mode: bool = False,
        benchmarking_args: GPT2BenchmarkingArgs = None,
    ) -> Union[NetworkResult, BenchmarkingResult]:

        tokenizer, gpt2_torch = self.setup_tokenizer_and_model(metadata, network_fpaths)

        # Prepare the input tokens and find out output sequence length.
        if not benchmarking_mode:
            output_seq_len = GPT2ModelTRTConfig.MAX_LENGTH[metadata.variant]
            input_ids = tokenizer([inference_input] * batch_size, padding=True, return_tensors="pt").input_ids
        else:
            input_seq_len = benchmarking_args.input_seq_len
            output_seq_len = benchmarking_args.output_seq_len
            input_ids = torch.randint(0, GPT2ModelTRTConfig.VOCAB_SIZE[metadata.variant], (batch_size, input_seq_len))

        # get single decoder iteration inference timing profile
        _, decoder_e2e_time = gpt2_inference(
            gpt2_torch, input_ids, timing_profile, use_cuda=(not use_cpu)
        )

        # get complete decoder inference result and its timing profile
        sample_output, full_e2e_runtime = full_inference(
            gpt2_torch,
            input_ids,
            tokenizer,
            timing_profile,
            max_length=output_seq_len,
            min_length=GPT2ModelTRTConfig.MIN_OUTPUT_LENGTH[metadata.variant] if not benchmarking_mode else output_seq_len,
            use_cuda=(not use_cpu),
            batch_size=batch_size,
            use_cache=metadata.other.kv_cache,
            num_beams=num_beams
        )

        # Prepare runtime results.
        runtime = [
            NetworkRuntime(
                name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                runtime=decoder_e2e_time,
            ),
            NetworkRuntime(
                name=GPT2ModelTRTConfig.NETWORK_FULL_NAME,
                runtime=full_e2e_runtime,
            ),
        ]

        # Skip result checking in benchmarking mode since the input data is random.
        if benchmarking_mode:
            return BenchmarkingResult(median_runtime=runtime, models=network_fpaths)

        # Remove the padding and end tokens.
        semantic_outputs = tokenizer.decode(
            sample_output[-1, :], skip_special_tokens=True
        )

        if isinstance(semantic_outputs, list):
            semantic_outputs = " ".join(semantic_outputs).strip()

        return NetworkResult(
            input=inference_input,
            output_tensor=sample_output,
            semantic_output=semantic_outputs,
            median_runtime=runtime,
            models=network_fpaths,
        )

    def execute_calculate_perplexity(
        self,
        metadata: NetworkMetadata,
        network_fpaths: NetworkModels,
        reference: str,
    ):
        tokenizer, gpt2_torch = self.setup_tokenizer_and_model(metadata, network_fpaths)
        reference = reference.replace("\\n", "\n")
        ppl_input_ids = tokenizer([reference], padding=True, return_tensors="pt").input_ids
        perplexity = calculate_perplexity(
            gpt2_torch, ppl_input_ids, GPT2ModelTRTConfig.MAX_LENGTH[metadata.variant]
        )

        return perplexity

    def run_framework(
        self,
        metadata: NetworkMetadata,
        network_input: List[str],
        working_directory: str,
        keep_onnx_model: bool,
        keep_pytorch_model: bool,
        timing_profile: TimingProfile,
        use_cpu: bool = False,
        batch_size: int = 1,
        args: object = None,
        benchmarking_mode: bool = False,
        perplexity_reference: List[str] = None,
    ) -> Union[List[NetworkResult], BenchmarkingResult]:

        """
        Main entry point of our function which compiles and generates our model data.
        """
        inference_results = []
        ppl_results = []
        workspace = NNFolderWorkspace(
            self.config.network_name, metadata, working_directory
        )
        try:
            network_fpaths = self.generate_and_download_framework(metadata, workspace)
            if not benchmarking_mode:
                for ninput in network_input:
                    inference_results.append(
                        self.execute_inference(
                            metadata, network_fpaths, ninput, timing_profile, use_cpu, batch_size, args.num_beams
                        )
                    )
                if perplexity_reference is not None:
                    for r in perplexity_reference:
                        ppl_results.append(
                            self.execute_calculate_perplexity(
                                metadata, network_fpaths, r
                            )
                        )
            else:
                benchmarking_args = GPT2BenchmarkingArgs(args.input_seq_len, args.output_seq_len)
                inference_results = self.execute_inference(
                    metadata, network_fpaths, None, timing_profile, use_cpu, batch_size, args.num_beams, True, benchmarking_args
                )
        finally:
            self.cleanup(workspace, keep_onnx_model, keep_pytorch_model)

        return inference_results, ppl_results

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
