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

"""
Executes ONNX Runtime framework code. See README.md for more information.
"""

import os
import sys
from typing import Dict, List, Tuple

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# huggingface
from transformers import T5Tokenizer, T5Config, PretrainedConfig
from transformers.generation_utils import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput

# torch
import torch

# TRT-HuggingFace
from NNDF.interface import OnnxRTCommand
from NNDF.torch_utils import expand_inputs_for_beam_search
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

from NNDF.general_utils import NNFolderWorkspace
from NNDF.tensorrt_utils import PolygraphyOnnxRunner
from T5.frameworks import T5FHuggingFace
from T5.T5ModelConfig import T5ModelTRTConfig, T5BenchmarkingArgs
from T5.measurements import decoder_inference, encoder_inference, full_inference
from NNDF.logger import G_LOGGER

class OnnxHFRunner(PolygraphyOnnxRunner, GenerationMixin):
    """Runner that adds interop support for HF and HF provided greedy_search functions."""

    def __init__(self, engine_fpath: str, network_metadata: NetworkMetadata, hf_config: PretrainedConfig):
        super().__init__(engine_fpath, network_metadata)
        # required for greedy search used by generation mixin
        self.main_input_name = "input_ids"
        self.config = hf_config

class T5OnnxEncoder(OnnxHFRunner):
    """OnnxRT implemented network interface that is mainly to check correctness."""

    def forward(self, input_ids, *args, **kwargs):
        # Unoptimized unconditional transfer to numpy for interfacing with polygraphy
        input_ids = input_ids.cpu().numpy().astype("int64")
        return torch.from_numpy(self.trt_context.infer({"input_ids": input_ids})["hidden_states"])

class T5OnnxDecoder(OnnxHFRunner):
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "encoder_hidden_states": kwargs["encoder_outputs"].last_hidden_state,
        }

    def forward(self, input_ids, encoder_hidden_states, *args, **kwargs):
        # Unoptimized unconditional transfer to numpy for interfacing with polygraphy
        input_ids = input_ids.cpu().numpy().astype("int64")
        encoder_hidden_states = encoder_hidden_states.cpu().numpy().astype("float32")

        logits = self.trt_context.infer(
            {"input_ids": input_ids, "encoder_hidden_states": encoder_hidden_states}
        )["hidden_states"]

        return Seq2SeqLMOutput(logits=torch.from_numpy(logits))

class T5ONNXRT(OnnxRTCommand):
    def __init__(self):
        super().__init__(
            T5ModelTRTConfig,
            "Runs polygraphy results for T5 model.",
            T5FHuggingFace,
        )
        self.t5_ort_decoder = None
        self.t5_ort_encoder = None

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        keep_onnx_model: bool = False,
        keep_torch_model: bool = False,
    ) -> None:
        # Deactivates context
        if self.t5_ort_encoder:
            self.t5_ort_encoder.release()
        if self.t5_ort_decoder:
            self.t5_ort_decoder.release()

        self.frameworks_cmd.cleanup(workspace, keep_onnx_model, keep_torch_model)

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Dict[str, NetworkModel],
        inference_input: str,
        timing_profile: TimingProfile,
        batch_size: int = 1,
        num_beams: int = 1,
        benchmarking_mode: bool = False,
        benchmarking_args: T5BenchmarkingArgs = None,
    ) -> NetworkResult:

        hf_config = T5Config.from_pretrained(metadata.variant)
        tokenizer = T5Tokenizer.from_pretrained(metadata.variant)
        # Prepare the input tokens and find out output sequence length.
        if not benchmarking_mode:
            output_seq_len = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
            input_ids = tokenizer([inference_input] * batch_size, padding=True, return_tensors="pt").input_ids
        else:
            max_seq_len = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
            input_seq_len = benchmarking_args.input_seq_len if benchmarking_args.input_seq_len > 0 else max_seq_len
            output_seq_len = benchmarking_args.output_seq_len if benchmarking_args.output_seq_len > 0 else max_seq_len
            input_ids = torch.randint(0, hf_config.vocab_size, (batch_size, input_seq_len))

        encoder_last_hidden_state, encoder_e2e_time = encoder_inference(
            self.t5_ort_encoder, input_ids, timing_profile
        )

        # Need to feed the decoder a new empty input_ids for text generation. 
        decoder_output_len = output_seq_len // 2

        decoder_input_ids = torch.full(
            (batch_size, decoder_output_len), tokenizer.convert_tokens_to_ids(tokenizer.pad_token), dtype=torch.int32
        )
        # OnnxRT currently does not enable kv cache
        _, decoder_e2e_time = decoder_inference(
            self.t5_ort_decoder,
            expand_inputs_for_beam_search(decoder_input_ids, num_beams) if num_beams > 1 else decoder_input_ids,
            expand_inputs_for_beam_search(encoder_last_hidden_state, num_beams) if num_beams > 1 else encoder_last_hidden_state,
            timing_profile,
            use_cache=metadata.other.kv_cache,
        )

        decoder_output, full_e2e_runtime = full_inference(
            self.t5_ort_encoder,
            self.t5_ort_decoder,
            input_ids,
            tokenizer,
            timing_profile,
            max_length=output_seq_len,
            min_length=T5ModelTRTConfig.MIN_OUTPUT_LENGTH[metadata.variant] if not benchmarking_mode else output_seq_len,
            use_cuda=False,
            num_beams=num_beams,
            batch_size=batch_size,
            use_cache=metadata.other.kv_cache,
        )

        # Prepare runtime results.
        runtime = [
            NetworkRuntime(
                name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                runtime=decoder_e2e_time,
            ),
            NetworkRuntime(
                name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                runtime=encoder_e2e_time,
            ),
            NetworkRuntime(
                name=T5ModelTRTConfig.NETWORK_FULL_NAME,
                runtime=full_e2e_runtime,
            ),
        ]
        models=NetworkModels(
            torch=None,
            onnx=list(onnx_fpaths.values()),
            trt=None
        )

        # Skip result checking in benchmarking mode since the input data is random.
        if benchmarking_mode:
            return BenchmarkingResult(median_runtime=runtime, models=models)

        # Remove the padding and end tokens.
        semantic_outputs = tokenizer.decode(
            decoder_output[-1, :], skip_special_tokens=True
        )

        if isinstance(semantic_outputs, list):
            semantic_outputs = " ".join(semantic_outputs).strip()

        return NetworkResult(
            input=inference_input,
            output_tensor=decoder_output,
            semantic_output=semantic_outputs,
            median_runtime=runtime,
            models=models,
        )

    def run_onnxrt(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Tuple[NetworkModel],
        network_input: List[str],
        working_directory: str,
        keep_onnx_model: bool,
        keep_torch_model: bool,
        timing_profile: TimingProfile,
        batch_size: int = 1,
        args: object = None,
        benchmarking_mode: bool = False,
    ) -> List[NetworkResult]:
        workspace = NNFolderWorkspace(
            self.frameworks_cmd.config.network_name, metadata, working_directory
        )

        results = []
        try:
            if metadata.other.kv_cache:
                assert False, "OnnxRT currently does not support kv cache."
            # no fpath provided for onnx files, download them
            if len(onnx_fpaths) == 0:
                onnx_fpaths = self.frameworks_cmd.generate_and_download_framework(
                    metadata, workspace
                ).onnx
            else:
                keep_onnx_model = True
                keep_torch_model = True

            # Output networks shall not exceed number of network segments explicitly defined by configuration file.
            assert len(onnx_fpaths) == len(
                T5ModelTRTConfig.NETWORK_SEGMENTS
            ), "There should only be {} exported ONNX segments in T5 model.".format(
                len(T5ModelTRTConfig.NETWORK_SEGMENTS)
            )

            lookup_onnx_table = {v.name: v for v in onnx_fpaths}

            hf_config = T5Config.from_pretrained(
                metadata.variant,
                use_cache=metadata.other.kv_cache
            )
            self.t5_ort_encoder = T5OnnxEncoder(
                lookup_onnx_table["encoder"].fpath, metadata, hf_config
            )
            self.t5_ort_decoder = T5OnnxDecoder(
                lookup_onnx_table["decoder"].fpath, metadata, hf_config
            )

            if not benchmarking_mode:
                for ninput in network_input:
                    results.append(
                        self.execute_inference(
                            metadata, lookup_onnx_table, ninput, timing_profile, batch_size, args.num_beams
                        )
                    )
            else:
                benchmarking_args = T5BenchmarkingArgs(args.input_seq_len, args.output_seq_len)
                results = self.execute_inference(
                    metadata, lookup_onnx_table, None, timing_profile, batch_size, args.num_beams, True, benchmarking_args
                )

        finally:
            self.cleanup(workspace, keep_onnx_model, keep_torch_model)
        # TODO: Add perplexity calculation for OnnxRT
        G_LOGGER.warning("perplexity calculation is disabled for OnnxRT.")
        return results

    def add_args(self, parser) -> None:
        super().add_args(parser)
        onnx_group = parser.add_argument_group("onnx models")
        onnx_group.add_argument(
            "--onnx-decoder-fpath",
            default=None,
            help="Path to ONNX decoder. If None is supplied, scripts will generate them from HuggingFace.",
        )
        onnx_group.add_argument(
            "--onnx-encoder-fpath",
            default=None,
            help="Path to ONNX encoder. If None is supplied, scripts will generate them from HuggingFace.",
        )

    def args_to_network_models(self, args) -> List[NetworkModel]:
        # Check if both flags are given otherwise error out
        decoder_fpath_check = args.onnx_decoder_fpath is None
        encoder_fpath_check = args.onnx_encoder_fpath is None

        network_models = None
        if decoder_fpath_check and encoder_fpath_check:
            network_models = tuple()
        elif decoder_fpath_check or encoder_fpath_check:
            raise self._parser.error(
                "Both --onnx-decoder-fpath and --onnx-encoder-fpath must be given. Otherwise neither should be provided for script to download them."
            )
        else:
            onnx_decoder = NetworkModel(
                name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=args.onnx_decoder_fpath,
            )
            onnx_encoder = NetworkModel(
                name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                fpath=args.onnx_encoder_fpath,
            )
            network_models = (onnx_decoder, onnx_encoder)

        return network_models

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        """Override args to metadata to use export subroutine."""
        frameworks_parsed_metadata = self.frameworks_cmd.args_to_network_metadata(args)

        return NetworkMetadata(
            variant=frameworks_parsed_metadata.variant,
            precision=Precision(fp16=args.fp16),
            other=frameworks_parsed_metadata.other,
        )


RUN_CMD = T5ONNXRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
