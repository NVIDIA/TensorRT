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
from typing import Dict, List, Tuple

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)


# numpy
import numpy as np

# torch
import torch

# huggingface
from transformers import GPT2Tokenizer, GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin

# TRT-HuggingFace
from NNDF.interface import TRTInferenceCommand
from NNDF.networks import (
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
from GPT2.GPT2ModelConfig import GPT2ModelTRTConfig
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

    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
    ):
        super().__init__(trt_engine_file, network_metadata)
        self.config = hf_config

class GPT2TRTDecoder(TRTHFRunner):
    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config)
        self.max_sequence_length = GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[network_metadata.variant]
        assert len(trt_engine_file.get_dynamic_shape_profiles()) == 1, "GPT2 should only have one dynamic shapes profile."

        # We only have one profile to select so we can just grab the profile at the start of the class
        self.profile_idx = self.get_optimization_profile(batch_size=1, sequence_length=1)
        self.inputs = {
            "input_ids": torch.zeros(1, self.max_sequence_length, dtype=torch.int32).cuda(),
        }
        self.outputs = {
            "logits": torch.zeros(1, self.max_sequence_length, GPT2ModelTRTConfig.VOCAB_SIZE, dtype=torch.float32).cuda()
        }
        self.bindings = self._allocate_memory(self.inputs, self.outputs)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Todo (@pchadha): add position_ids, token_type_ids support
        return {
            "input_ids": input_ids,
        }

    def forward(self, input_ids, **kwargs):
        self.inputs["input_ids"][:, :input_ids.shape[1]] = input_ids
        self.trt_context.set_binding_shape(0, input_ids.shape)
        self.trt_context.execute_v2(bindings=self.bindings)
        return CausalLMOutputWithCrossAttentions(logits=self.outputs["logits"][:, :input_ids.shape[1], :])

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
    ) -> NetworkResult:

        tokenizer = GPT2Tokenizer.from_pretrained(metadata.variant)
        input_ids = tokenizer(inference_input, return_tensors="pt").input_ids

        # get single decoder iteration inference timing profile
        _, decoder_e2e_median_time = gpt2_inference(
            self.gpt2_trt, input_ids, timing_profile
        )

        # get complete decoder inference result and its timing profile
        sample_output, full_e2e_median_runtime = full_inference_greedy(
            self.gpt2_trt, input_ids, timing_profile,
            max_length=GPT2ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
        )

        semantic_outputs = []
        for i, sample_output in enumerate(sample_output):
            semantic_outputs.append(tokenizer.decode(sample_output, skip_special_tokens=True))

        return NetworkResult(
            input=inference_input,
            output_tensor=sample_output,
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
            models=NetworkModels(
                torch=None,
                onnx=list(onnx_fpaths.values()),
                trt=[
                    NetworkModel(
                        name=GPT2ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                        fpath=self.gpt2_engine.fpath,
                    ),
                ],
            ),
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
    ) -> List[NetworkResult]:
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

            self.gpt2_engine = GPT2ONNXFile(gpt2_onnx_fpath, metadata).as_trt_engine(gpt2_onnx_fpath + ".engine")
            tfm_config = GPT2Config(
                use_cache=metadata.other.kv_cache,
            )
            self.gpt2_trt = GPT2TRTDecoder(self.gpt2_engine, metadata, tfm_config)

            for ninput in network_input:
                results.append(
                    self.execute_inference(
                        metadata, hash_onnx_fpath, ninput, timing_profile
                    )
                )

        finally:
            self.cleanup(workspace, keep_trt_engine, keep_onnx_model, keep_torch_model)

        return results

    def add_args(self, parser) -> None:
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
