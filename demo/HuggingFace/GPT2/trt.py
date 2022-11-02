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
from typing import Dict, List, Tuple, Union

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# numpy
import numpy as np

# polygraphy
from polygraphy.backend.trt import Profile

# torch
import torch

# huggingface
from transformers import GPT2Tokenizer, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin

# tensorrt
from tensorrt import PreviewFeature

# TRT-HuggingFace
from NNDF.interface import TRTInferenceCommand
from NNDF.networks import (
    BenchmarkingResult,
    NetworkMetadata,
    NetworkModels,
    NetworkModel,
    NetworkResult,
    NetworkRuntime,
    Precision,
    TimingProfile,
)

from NNDF.tensorrt_utils import TRTNativeRunner, TRTPolygraphyRunner
from GPT2.frameworks import GPT2HuggingFace
from NNDF.general_utils import NNFolderWorkspace
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig, GPT2BenchmarkingArgs
from GPT2.measurements import gpt2_inference, full_inference_greedy
from GPT2.export import GPT2ONNXFile, GPT2TRTEngine


class TRTHFRunner(TRTNativeRunner, GenerationMixin):
    """Runner that adds interop support for HF and HF provided greedy_search functions."""

    def _allocate_memory(self, input_dict: Dict[str, np.ndarray], output_dict: Dict[str, np.ndarray]):
        """Helper function for binding several inputs at once and pre-allocating the results."""
        bindings = [None] * self.trt_engine.num_bindings

        for input_name, input_array in input_dict.items():
            # Allocate memory for inputs
            input_idx = self.trt_engine.get_binding_index(input_name)
            self.trt_context.set_binding_shape(input_idx, input_array.shape)
            bindings[input_idx] = input_array.data_ptr()

        assert self.trt_context.all_binding_shapes_specified

        for output_name, output_array in output_dict.items():
            # Output shape should be allocated from context size
            output_idx = self.trt_engine.get_binding_index(output_name)
            bindings[output_idx] = output_array.data_ptr()

        return bindings

    def set_return_device(self, return_device):
        """
        Sets the return device of the return via to(). Device name should be the same as torch devices: cuda, cpu, etc.
        This is used in our measurement code.
        """
        self.return_device = return_device

    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1
    ):
        super().__init__(trt_engine_file, network_metadata)
        self.config = hf_config
        self.batch_size = batch_size
        self.return_device = "cuda"

class GPT2TRTDecoder(TRTHFRunner):
    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1,
        max_sequence_length: int = 0,
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config, batch_size)
        self.max_sequence_length = max_sequence_length if max_sequence_length > 0 \
            else GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[network_metadata.variant]

        # We only have one profile to select so we can just grab the profile at the start of the class
        self.profile_idx = self.get_optimization_profile(batch_size=batch_size, sequence_length=1)
        self.inputs = {
            "input_ids": torch.zeros(self.batch_size, self.max_sequence_length, dtype=torch.int32).cuda(),
        }
        self.outputs = {
            "logits": torch.zeros(self.batch_size, self.max_sequence_length, GPT2ModelTRTConfig.VOCAB_SIZE, dtype=torch.float32).cuda()
        }
        self.bindings = self._allocate_memory(self.inputs, self.outputs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Todo (@pchadha): add position_ids, token_type_ids support
        return {
            "input_ids": input_ids,
        }

    def forward(self, input_ids, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        self.inputs["input_ids"].view(-1)[:batch_size * seq_len] = input_ids.flatten()
        self.trt_context.set_binding_shape(0, input_ids.shape)
        self.trt_context.execute_v2(bindings=self.bindings)
        vocab_size = self.outputs["logits"].shape[2]
        logits = self.outputs["logits"].view(-1)[:batch_size * seq_len * vocab_size].view(batch_size, seq_len, vocab_size)
        return CausalLMOutputWithCrossAttentions(logits=logits.to(self.return_device))

class GPT2Polygraphy(TRTInferenceCommand):
    def __init__(self):
        super().__init__(
            GPT2ModelTRTConfig, "Runs polygraphy results for GPT2 model.", GPT2HuggingFace
        )
        self.gpt2_trt = None

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        keep_trt_engine: bool = False,
        keep_onnx_model: bool = False,
        keep_torch_model: bool = False,
    ) -> None:
        # Deactivates context
        if self.gpt2_trt is not None:
            self.gpt2_trt.release()

        if not keep_trt_engine:
            self.gpt2_engine.cleanup()

        self.frameworks_cmd.cleanup(workspace, keep_onnx_model, keep_torch_model)

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Dict[str, NetworkModel],
        inference_input: str,
        timing_profile: TimingProfile,
        batch_size: int = 1,
        benchmarking_mode: bool = False,
        benchmarking_args: GPT2BenchmarkingArgs = None,
    ) -> Union[NetworkResult, BenchmarkingResult]:

        tokenizer = GPT2Tokenizer.from_pretrained(metadata.variant)

        # GPT2 has no proper token set. Use custom token. Only "generate()" will auto
        # replace with EOS token when using generating mode
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Prepare the input tokens and find out output sequence length.
        if not benchmarking_mode:
            output_seq_len = GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
            input_ids = tokenizer([inference_input] * batch_size, return_tensors="pt").input_ids
        else:
            input_seq_len = benchmarking_args.input_seq_len
            output_seq_len = benchmarking_args.output_seq_len
            input_ids = torch.randint(0, GPT2ModelTRTConfig.VOCAB_SIZE, (batch_size, input_seq_len))

        # get single decoder iteration inference timing profile
        _, decoder_e2e_time = gpt2_inference(
            self.gpt2_trt, input_ids, timing_profile,
        )

        # get complete decoder inference result and its timing profile
        sample_output, full_e2e_runtime = full_inference_greedy(
            self.gpt2_trt,
            input_ids,
            timing_profile,
            max_length=output_seq_len,
            batch_size=batch_size,
            early_stopping=(not benchmarking_mode),
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
        models = NetworkModels(
            torch=None,
            onnx=list(onnx_fpaths.values()),
            trt=[
                NetworkModel(
                    name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    fpath=self.gpt2_engine.fpath,
                ),
            ],
        )

        # Skip result checking in benchmarking mode since the input data is random.
        if benchmarking_mode:
            return BenchmarkingResult(median_runtime=runtime, models=models)

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
            models=models,
        )

    def run_trt(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Tuple[NetworkModel],
        network_input: List[str],
        working_directory: str,
        keep_trt_engine: bool,
        keep_onnx_model: bool,
        keep_torch_model: bool,
        timing_profile: TimingProfile,
        batch_size: int = 1,
        args: object = None,
        benchmarking_mode: bool = False,
        preview_dynamic_shapes: bool = False,
    ) -> Union[List[NetworkResult], BenchmarkingResult]:

        workspace = NNFolderWorkspace(
            self.frameworks_cmd.config.network_name, metadata, working_directory
        )

        results = []
        try:
            # no fpath provided for onnx files, download them
            if len(onnx_fpaths) == 0:
                onnx_fpaths = self.frameworks_cmd.generate_and_download_framework(
                    metadata, workspace
                ).onnx
            else:
                keep_onnx_model = True
                keep_torch_model = True

            # Output networks shall not exceed number of network segments explicitly defined by configuraiton file.
            assert len(onnx_fpaths) == len(
                GPT2ModelTRTConfig.NETWORK_SEGMENTS
            ), "There should only be {} exported ONNX segments in GPT2 model."

            hash_onnx_fpath = {v.name: v for v in onnx_fpaths}

            gpt2_onnx_fpath = hash_onnx_fpath[
                GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME
            ].fpath

            # Generate optimization profiles.
            if not benchmarking_mode:
                max_sequence_length = GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
            else:
                max_sequence_length = args.output_seq_len
            profiles = [Profile().add(
                "input_ids",
                min=(batch_size, 1),
                opt=(batch_size, max_sequence_length // 2),
                max=(batch_size, max_sequence_length),
            )]

            # Build the engine.
            if not benchmarking_mode:
                engine_tag = "bs{}".format(batch_size)
            else:
                engine_tag = "bs{}-inseq{}-outseq{}".format(batch_size, args.input_seq_len, args.output_seq_len)

            preview_features = []
            if preview_dynamic_shapes:
                preview_features = [PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]
                engine_tag += "-previewFasterDynamicShapes"

            self.gpt2_engine = GPT2ONNXFile(gpt2_onnx_fpath, metadata).as_trt_engine(
                gpt2_onnx_fpath + engine_tag + ".engine",
                profiles=profiles,
                preview_features=preview_features
            )
            tfm_config = GPT2Config(
                use_cache=metadata.other.kv_cache,
            )
            self.gpt2_trt = GPT2TRTDecoder(self.gpt2_engine, metadata, tfm_config, batch_size, max_sequence_length)

            if not benchmarking_mode:
                for ninput in network_input:
                    results.append(
                        self.execute_inference(
                            metadata, hash_onnx_fpath, ninput, timing_profile, batch_size
                        )
                    )
            else:
                benchmarking_args = GPT2BenchmarkingArgs(args.input_seq_len, args.output_seq_len)
                results = self.execute_inference(
                    metadata, hash_onnx_fpath, None, timing_profile, batch_size, True, benchmarking_args
                )

        finally:
            self.cleanup(workspace, keep_trt_engine, keep_onnx_model, keep_torch_model)

        return results

    def add_args(self, parser) -> None:
        super().add_args(parser)

        # use the same args as frameworks.py
        self.frameworks_cmd.add_args(parser)
        polygraphy_group = parser.add_argument_group("polygraphy")
        polygraphy_group.add_argument(
            "--onnx-fpath",
            default=None,
            help="Path to GPT2 ONNX model. If None is supplied, scripts will generate them from HuggingFace.",
        )
        polygraphy_group.add_argument(
            "--fp16", action="store_true", help="Enables fp16 TensorRT tactics."
        )
        polygraphy_group.add_argument(
            "--save-trt-engine",
            action="store_true",
            help="Saves TensorRT runtime engine in working directory.",
        )

    def args_to_network_models(self, args) -> List[NetworkModel]:
        gpt2_fpath_check = args.onnx_fpath is None

        network_models = None
        if gpt2_fpath_check:
            network_models = tuple()
        else:
            onnx_decoder = NetworkModel(
                name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=args.onnx_fpath,
            )
            network_models = (onnx_decoder)

        return network_models

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        frameworks_parsed_metadata = self.frameworks_cmd.args_to_network_metadata(args)

        return NetworkMetadata(
            variant=frameworks_parsed_metadata.variant,
            precision=Precision(fp16=args.fp16),
            other=frameworks_parsed_metadata.other,
        )


RUN_CMD = GPT2Polygraphy()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
