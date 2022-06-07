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
from functools import reduce

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# polygraphy
from polygraphy.backend.trt import Profile

# torch
import torch

# huggingface
from transformers import BartTokenizer, BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GenerationMixin

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

from NNDF.tensorrt_utils import TRTNativeRunner
from NNDF.general_utils import NNFolderWorkspace
from BART.frameworks import BARTHuggingFace
from BART.BARTModelConfig import BARTModelTRTConfig, BARTBenchmarkingArgs
from BART.measurements import decoder_inference, encoder_inference, full_inference_greedy
from BART.export import BARTDecoderONNXFile, BARTEncoderONNXFile
from NNDF.models import TRTEngineFile

class TRTHFRunner(TRTNativeRunner, GenerationMixin):
    """Runner that adds interop support for HF and HF provided greedy_search functions."""

    # Stores the encoder input length received at runtime, which is used to slice decoder inputs.
    ENCODER_LENGTH = 0
    def _allocate_memory(self,
                         input_shapes: Dict[str, tuple],
                         input_types: Dict[str, torch.dtype],
                         output_shapes: Dict[str, tuple],
                         output_types: Dict[str, torch.dtype]):
        """Helper function for binding several inputs at once and pre-allocating the results."""
        # Allocate memories as 1D linear buffers for simpler handling of dynamic shapes.
        self.inputs = {
            k: torch.zeros(reduce(lambda v, a: v*a, shape), dtype=input_types[k]).cuda()
            for k, shape in input_shapes.items()
        }

        self.outputs = {
            k: torch.zeros(reduce(lambda v, a: v*a, shape), dtype=output_types[k]).cuda()
            for k, shape in output_shapes.items()
        }

        bindings = [None] * self.trt_engine.num_bindings

        for input_name, input_array in self.inputs.items():
            # Allocate memory for inputs
            input_idx = self.trt_engine.get_binding_index(input_name)
            self.trt_context.set_binding_shape(input_idx, input_shapes[input_name])
            bindings[input_idx] = input_array.data_ptr()

        assert self.trt_context.all_binding_shapes_specified

        for output_name, output_array in self.outputs.items():
            # Output shape should be allocated from context size
            output_idx = self.trt_engine.get_binding_index(output_name)
            bindings[output_idx] = output_array.data_ptr()

        return bindings

    def __init__(
        self,
        trt_engine_file: TRTEngineFile,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1
    ):
        super().__init__(trt_engine_file, network_metadata)
        self.config = hf_config
        self.batch_size = batch_size

class BARTTRTEncoder(TRTHFRunner):
    """TRT implemented network interface that can be used to measure inference time."""

    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config, batch_size = batch_size)
        self.max_sequence_length = BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[network_metadata.variant]

        # We only have one profile to select so we can just grab the profile at the start of the class
        self.profile_idx = self.get_optimization_profile(batch_size=self.batch_size, sequence_length=1)

        self.input_shapes = {
            "input_ids": (self.batch_size, self.max_sequence_length)
        }
        self.input_types = {
            "input_ids": torch.int32
        }
        self.output_shapes = {
            "hidden_states": (self.batch_size, self.max_sequence_length, self.max_sequence_length)
        }
        self.output_types = {
            "hidden_states": torch.float32
        }

        self.bindings = self._allocate_memory(self.input_shapes, self.input_types, self.output_shapes, self.output_types)

    def forward(self, input_ids, *args, **kwargs):
        bs = self.batch_size
        max_length = self.max_sequence_length
        TRTHFRunner.ENCODER_LENGTH = input_ids.shape[1]
        input_length = input_ids.shape[1]

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (input_ids.device == torch.device("cpu"))

        # We allocate the buffers using max_length, but we only need to first portion of it, so copy the data into the
        # first portion of the input buffer.
        # TODO: Could we just reuse input_ids' data_ptr() as the first binding when input_ids is already contiguous to
        # avoid an additional D2D?
        if is_cpu_mode:
            self.inputs["input_ids"] = input_ids.int().flatten().contiguous().cuda()
            self.bindings[0] = self.inputs["input_ids"].data_ptr()
        else:
            self.inputs["input_ids"][:bs * input_length] = input_ids.flatten()

        # Set the binding shape of input_ids, which should be (bs, input_length).
        self.trt_context.set_binding_shape(0, input_ids.shape)

        # Launch TRT inference.
        # TODO: Could we use execute_v2_async() instead of execute_v2()?
        self.trt_context.execute_v2(bindings=self.bindings)

        # We allocate the buffers using max_length, but we only need to first portion of it, so get only the first
        # portion of the output buffer and return that.
        # TODO: Could we construct a Torch tensor using given data_ptr() to avoid this D2D copy?
        if is_cpu_mode:
            folded = self.outputs["hidden_states"].cpu()[:bs * input_length * max_length].view(bs, input_length, max_length)
        else:
            folded = self.outputs["hidden_states"][:bs * input_length * max_length].view(bs, input_length, max_length)

        return folded

class BARTTRTDecoder(TRTHFRunner):

    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        hf_config: PretrainedConfig,
        batch_size: int = 1
    ):
        super().__init__(trt_engine_file, network_metadata, hf_config, batch_size = batch_size)
        self.max_sequence_length = BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[network_metadata.variant]

        # We only have one profile to select so we can just grab the profile at the start of the class
        self.profile_idx = self.get_optimization_profile(batch_size=self.batch_size, sequence_length=1)

        self.input_types = {
            "input_ids": torch.int32,
            "encoder_hidden_states": torch.float32
        }
        self.input_shapes = {
            "input_ids": (self.batch_size, self.max_sequence_length),
            "encoder_hidden_states": (self.batch_size, self.max_sequence_length, self.max_sequence_length)
        }

        self.output_shapes = {
            "hidden_states": (self.batch_size, self.max_sequence_length, BARTModelTRTConfig.VOCAB_SIZE[network_metadata.variant])
        }
        self.output_types = {
            "hidden_states": torch.float32
        }
        self.bindings = self._allocate_memory(self.input_shapes, self.input_types, self.output_shapes, self.output_types)

        # Optimization bit
        self.persist_encoder_hidden_states = False
        self.return_device = "cuda"

        self.variant = network_metadata.variant # record variant name to later index the vocab_size in forward()

    def set_encoder_hidden_states_for_inference_cycle(self, encoder_hidden_states):
        """Used to cache encoder hidden state runs across same encoder sessions"""
        self.persist_encoder_hidden_states = True

        bs = self.batch_size
        max_length = self.max_sequence_length
        encoder_length = TRTHFRunner.ENCODER_LENGTH
        if encoder_hidden_states.device == torch.device("cpu"):
            self.inputs["encoder_hidden_states"] = encoder_hidden_states.flatten().contiguous().cuda()
            self.bindings[1] = self.inputs["encoder_hidden_states"].data_ptr()
        else:
            self.inputs["encoder_hidden_states"][:bs * encoder_length * max_length] = encoder_hidden_states.flatten()

    def set_return_device(self, return_device):
        """
        Sets the return device of the return via to(). Device name should be the same as torch devices: cuda, cpu, etc.
        This is used in our measurement code.
        """
        self.return_device = return_device

    def forward(self, input_ids, encoder_hidden_states, *args, **kwargs):
        # Get the batch size.
        bs = self.batch_size

        # Get the maximum sequence length.
        max_length = self.max_sequence_length

        # Get the vocab size.
        vocab_size = BARTModelTRTConfig.VOCAB_SIZE[self.variant]

        # Actual sequence length of the input_ids and the output hidden_states.
        input_length = input_ids.shape[1]

        # The sequence length of the encoder_hidden_states.
        encoder_length = TRTHFRunner.ENCODER_LENGTH

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (input_ids.device == torch.device("cpu")) or (self.return_device == "cpu")

        # We allocate the buffers using max_length, but we only need to first portion of it, so copy the data into the
        # first portion of the input buffer.
        # TODO: Could we just reuse input_ids' data_ptr() as the first binding when input_ids is already contiguous to
        # avoid an additional D2D?
        if is_cpu_mode:
            self.inputs["input_ids"] = input_ids.int().flatten().contiguous().cuda()
            self.bindings[0] = self.inputs["input_ids"].data_ptr()
        else:
            self.inputs["input_ids"][:bs * input_length] = input_ids.flatten()

        # Set the binding shape of input_ids, which should be (bs, input_length).
        self.trt_context.set_binding_shape(0, input_ids.shape)

        # If encoder hidden states have not been copied yet, copy the hidden states to the input buffer.
        if not self.persist_encoder_hidden_states:
            if is_cpu_mode:
                self.inputs["encoder_hidden_states"] = encoder_hidden_states.flatten().contiguous().cuda()
                self.bindings[1] = self.inputs["encoder_hidden_states"].data_ptr()
            else:
                self.inputs["encoder_hidden_states"][:bs * encoder_length * max_length] = encoder_hidden_states.flatten()

        # Set the binding shape of encoder_hidden_states, which should be (bs, encoder_length, max_length).
        self.trt_context.set_binding_shape(1, (bs, encoder_length, max_length))

        # Launch TRT inference.
        # TODO: Could we use execute_v2_async() instead of execute_v2()? Current profiling shows that there is a
        # synchronization inside TRT's inference body, so this change may not be needed.
        self.trt_context.execute_v2(bindings=self.bindings)

        # We allocate the buffers using max_length, but we only need to first portion of it, so get only the first
        # portion of the output buffer and return that.
        # TODO: Could we construct a Torch tensor using given data_ptr() to avoid this D2D copy?
        if is_cpu_mode:
            folded = self.outputs["hidden_states"].cpu()[:bs * input_length * vocab_size].view(bs, input_length, vocab_size)
        else:
            folded = self.outputs["hidden_states"][:bs * input_length * vocab_size].view(bs, input_length, vocab_size)

        # Transfer predictions back from GPU to do greedy search
        return Seq2SeqLMOutput(logits=folded.to(self.return_device))

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {
            "input_ids": input_ids,
            "encoder_hidden_states": kwargs["encoder_hidden_states"],
        }


class BARTTRT(TRTInferenceCommand):
    def __init__(self):
        super().__init__(
            BARTModelTRTConfig,
            "Runs trt results for BART model.",
            BARTHuggingFace,
        )
        self.BART_trt_decoder = None
        self.BART_trt_encoder = None

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        keep_trt_engine: bool = False,
        keep_onnx_model: bool = False,
        keep_torch_model: bool = False,
    ) -> None:
        # Deactivates context
        if self.BART_trt_encoder:
            self.BART_trt_encoder.release()
        if self.BART_trt_decoder:
            self.BART_trt_decoder.release()

        if not keep_trt_engine:
            self.BART_trt_encoder_engine.cleanup()
            self.BART_trt_decoder_engine.cleanup()

        self.frameworks_cmd.cleanup(workspace, keep_onnx_model, keep_torch_model)

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Dict[str, NetworkModel],
        inference_input: str,
        timing_profile: TimingProfile,
        batch_size: int = 1,
        benchmarking_mode: bool = False,
        benchmarking_args: BARTBenchmarkingArgs = None,
    ) -> Union[NetworkResult, BenchmarkingResult]:

        tokenizer = BartTokenizer.from_pretrained(metadata.variant)
        # Prepare the input tokens and find output sequence length.
        if not benchmarking_mode:
            output_seq_len = BARTModelTRTConfig.MAX_OUTPUT_LENGTH[metadata.variant] # note: T5 uses XXModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant] which is the max input length. Here should rather be the max output length for generation
            input_ids = tokenizer([inference_input] * batch_size, padding=True, return_tensors="pt").input_ids
        else:
            max_seq_len = BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
            input_seq_len = benchmarking_args.input_seq_len if benchmarking_args.input_seq_len > 0 else max_seq_len
            output_seq_len = benchmarking_args.output_seq_len if benchmarking_args.output_seq_len > 0 else max_seq_len
            input_ids = torch.randint(0, BARTModelTRTConfig.VOCAB_SIZE[metadata.variant], (batch_size, input_seq_len))

        encoder_last_hidden_state, encoder_e2e_median_time = encoder_inference(
            self.BART_trt_encoder, input_ids, timing_profile
        )
        _, decoder_e2e_median_time = decoder_inference(
            self.BART_trt_decoder,
            input_ids,
            encoder_last_hidden_state,
            timing_profile,
        )
        decoder_output_greedy, full_e2e_median_runtime = full_inference_greedy(
            self.BART_trt_encoder,
            self.BART_trt_decoder,
            input_ids,
            tokenizer,
            timing_profile,
            max_length=output_seq_len,
            batch_size=batch_size,
            early_stopping=(not benchmarking_mode),
        )

        # Prepare runtime results.
        median_runtime=[
            NetworkRuntime(
                name=BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                runtime=decoder_e2e_median_time,
            ),
            NetworkRuntime(
                name=BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                runtime=encoder_e2e_median_time,
            ),
            NetworkRuntime(
                name=BARTModelTRTConfig.NETWORK_FULL_NAME,
                runtime=full_e2e_median_runtime,
            ),
        ],
        models=NetworkModels(
            torch=None,
            onnx=list(onnx_fpaths.values()),
            trt=[
                NetworkModel(
                    name=BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    fpath=self.BART_trt_decoder_engine.fpath,
                ),
                NetworkModel(
                    name=BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                    fpath=self.BART_trt_encoder_engine.fpath,
                ),
            ],
        ),

        # Skip result checking in benchmarking mode since the input data is random.
        if benchmarking_mode:
            return BenchmarkingResult(median_runtime=median_runtime, models=models)

        # Remove the padding and end tokens.
        semantic_outputs = tokenizer.decode(
            decoder_output_greedy[-1, :], skip_special_tokens=True
        )

        if isinstance(semantic_outputs, list):
            semantic_outputs = " ".join(semantic_outputs).strip()

        return NetworkResult(
            input=inference_input,
            output_tensor=encoder_last_hidden_state,
            semantic_output=semantic_outputs,
            median_runtime=median_runtime,
            models=models,
        )

    def _setup_workspace(self, metadata: NetworkMetadata, working_directory: str) -> NNFolderWorkspace:
        return NNFolderWorkspace(
            self.frameworks_cmd.config.network_name, metadata, working_directory
        )

    def _download_models(
        self,
        workspace: NNFolderWorkspace,
        metadata: NetworkMetadata,
    ) -> Tuple[NetworkModel]:
        # No fpath provided for onnx files, download them from HuggingFace repo.
        return self.frameworks_cmd.generate_and_download_framework(
            metadata, workspace
        ).onnx

    def _setup_engines(
        self,
        metadata: NetworkMetadata,
        hash_onnx_fpath: Dict[str, NetworkModel],
        batch_size: int,
        benchmarking_args: BARTBenchmarkingArgs = None,
    ) -> None:

        # Output networks shall not exceed number of network segments explicitly defined by configuration file.
        assert len(hash_onnx_fpath) == len(
            BARTModelTRTConfig.NETWORK_SEGMENTS
        ), "There should only be {} exported ONNX segments in BART model.".format(
            len(BARTModelTRTConfig.NETWORK_SEGMENTS)
        )

        decoder_onnx_fpath = hash_onnx_fpath[
            BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME
        ].fpath
        encoder_onnx_fpath = hash_onnx_fpath[
            BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME
        ].fpath

        # Generate optimization profiles.
        max_sequence_length = BARTModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]

        if benchmarking_args is None or benchmarking_args.input_seq_len is None:
            opt_input_seq_len = max_sequence_length // 2
        else:
            opt_input_seq_len = benchmarking_args.input_seq_len

        if benchmarking_args is None or benchmarking_args.output_seq_len is None:
            opt_output_seq_len = max_sequence_length // 2
        else:
            opt_output_seq_len = benchmarking_args.output_seq_len

        encoder_profiles = [
            Profile().add(
                "input_ids",
                min=(batch_size, 1),
                opt=(batch_size, opt_input_seq_len),
                max=(batch_size, max_sequence_length),
            )
        ]

        decoder_profiles = [
            Profile().add(
                "input_ids",
                min=(batch_size, 1),
                opt=(batch_size, opt_output_seq_len),
                max=(batch_size, max_sequence_length),
            ).add(
                "encoder_hidden_states",
                min=(batch_size, 1, max_sequence_length),
                opt=(batch_size, opt_input_seq_len, max_sequence_length),
                max=(batch_size, max_sequence_length, max_sequence_length),
            )
        ]

        # Convert ONNX models to TRT engines.
        if benchmarking_args is None:
            engine_tag = "bs{}".format(batch_size)
        else:
            engine_tag = "bs{}-inseq{}-outseq{}".format(batch_size, benchmarking_args.input_seq_len, benchmarking_args.output_seq_len)

        self.BART_trt_encoder_engine = BARTEncoderONNXFile(
            encoder_onnx_fpath, metadata
        ).as_trt_engine(
            encoder_onnx_fpath + "-{}.engine".format(engine_tag),
            profiles=encoder_profiles,
        )
        self.BART_trt_decoder_engine = BARTDecoderONNXFile(
            decoder_onnx_fpath, metadata
        ).as_trt_engine(
            decoder_onnx_fpath + "-{}.engine".format(engine_tag),
            profiles=decoder_profiles,
        )

        # Create BARTTRTEncoder and BARTTRTDecoder instances.
        tfm_config = BartConfig(
            use_cache=metadata.other.kv_cache,
            num_layers=BARTModelTRTConfig.NUMBER_OF_LAYERS[metadata.variant],
        )
        self.BART_trt_encoder = BARTTRTEncoder(
            self.BART_trt_encoder_engine, metadata, tfm_config, batch_size=batch_size
        )
        self.BART_trt_decoder = BARTTRTDecoder(
            self.BART_trt_decoder_engine, metadata, tfm_config, batch_size=batch_size
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
    ) -> Union[List[NetworkResult], BenchmarkingResult] :

        workspace = self._setup_workspace(metadata, working_directory)

        # Keep onnx and Torch models if they are provided by users.
        if len(onnx_fpaths) == 0:
            onnx_fpaths = self._download_models(workspace, metadata)
        else:
            keep_onnx_model = True
            keep_torch_model = True

        hash_onnx_fpath = {v.name: v for v in onnx_fpaths}

        results = []
        try:
            if not benchmarking_mode:
                self._setup_engines(metadata, hash_onnx_fpath, batch_size)
                for ninput in network_input:
                    results.append(
                        self.execute_inference(
                            metadata, hash_onnx_fpath, ninput, timing_profile, batch_size
                        )
                    )
            else:
                benchmarking_args = BARTBenchmarkingArgs(args.input_seq_len, args.output_seq_len)
                self._setup_engines(metadata, hash_onnx_fpath, batch_size, benchmarking_args)
                results = self.execute_inference(
                    metadata, hash_onnx_fpath, None, timing_profile, batch_size, True, benchmarking_args
                )

        finally:
            self.cleanup(workspace, keep_trt_engine, keep_onnx_model, keep_torch_model)

        return results

    def add_args(self, parser) -> None:
        super().add_args(parser)
        polygraphy_group = parser.add_argument_group("polygraphy models")
        polygraphy_group.add_argument(
            "--onnx-decoder-fpath",
            default=None,
            help="Path to ONNX decoder. If None is supplied, scripts will generate them from HuggingFace.",
        )
        polygraphy_group.add_argument(
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
                name=BARTModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=args.onnx_decoder_fpath,
            )
            onnx_encoder = NetworkModel(
                name=BARTModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                fpath=args.onnx_encoder_fpath,
            )
            network_models = (onnx_decoder, onnx_encoder)

        return network_models

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        frameworks_parsed_metadata = self.frameworks_cmd.args_to_network_metadata(args)

        return NetworkMetadata(
            variant=frameworks_parsed_metadata.variant,
            precision=Precision(fp16=args.fp16),
            other=frameworks_parsed_metadata.other,
        )


RUN_CMD = BARTTRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
