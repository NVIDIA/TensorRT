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

from typing import List, Union

# huggingface
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)

# torch
import torch

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# TRT-HuggingFace
from NNDF.interface import FrameworkCommand
from NNDF.torch_utils import expand_inputs_for_beam_search
from NNDF.networks import (
    BenchmarkingResult,
    NetworkResult,
    NetworkMetadata,
    NetworkRuntime,
    NetworkModels,
    NetworkModel,
    TimingProfile,
)
from T5.export import T5EncoderTorchFile, T5DecoderTorchFile
from T5.T5ModelConfig import T5ModelTRTConfig, T5BenchmarkingArgs
from T5.measurements import decoder_inference, encoder_inference, full_inference_greedy, full_inference_beam, calculate_perplexity
from NNDF.general_utils import confirm_folder_delete, NNFolderWorkspace


class T5FHuggingFace(FrameworkCommand):
    def __init__(self):
        super().__init__(
            T5ModelTRTConfig, description="Runs framework results for T5 model."
        )

        self.onnx_t5_encoder = None
        self.onnx_t5_decoder = None
        self.torch_t5_dir = None

    def generate_and_download_framework(
        self, metadata: NetworkMetadata, workspace: NNFolderWorkspace
    ) -> NetworkModels:

        cache_variant = False
        if metadata.other.kv_cache:
            cache_variant = True

        trt_t5_config = self.config
        metadata_serialized = trt_t5_config.get_metadata_string(metadata)
        workspace_dir = workspace.get_path()

        pytorch_model_dir = os.path.join(workspace_dir, metadata_serialized)
        # We keep track of the generated torch location for cleanup later
        self.torch_t5_dir = pytorch_model_dir

        model = None
        tfm_config = T5Config(
            use_cache=cache_variant,
            num_layers=T5ModelTRTConfig.NUMBER_OF_LAYERS[metadata.variant],
        )
        if not os.path.exists(pytorch_model_dir):
            # Generate the pre-trained weights
            model = T5ForConditionalGeneration(tfm_config).from_pretrained(
                metadata.variant
            )
            model.save_pretrained(pytorch_model_dir)
            print("Pytorch Model saved to {}".format(pytorch_model_dir))
        else:
            print(
                "Frameworks file already exists, skipping generation and loading from file instead."
            )
            model = T5ForConditionalGeneration(tfm_config).from_pretrained(
                pytorch_model_dir
            )

        # These ONNX models can be converted using special encoder and decoder classes.
        root_onnx_model_name = "{}.onnx".format(metadata_serialized)
        root_onnx_model_fpath = os.path.join(
            os.getcwd(), workspace_dir, root_onnx_model_name
        )
        encoder_onnx_model_fpath = root_onnx_model_fpath + "-encoder.onnx"
        decoder_onnx_model_fpath = root_onnx_model_fpath + "-decoder-with-lm-head.onnx"

        t5_encoder = T5EncoderTorchFile(model, metadata)
        t5_decoder = T5DecoderTorchFile(model, metadata)
        self.onnx_t5_encoder = t5_encoder.as_onnx_model(
            encoder_onnx_model_fpath, force_overwrite=False
        )
        self.onnx_t5_decoder = t5_decoder.as_onnx_model(
            decoder_onnx_model_fpath, force_overwrite=False
        )

        onnx_models = [
            NetworkModel(
                name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                fpath=self.onnx_t5_decoder.fpath,
            ),
            NetworkModel(
                name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                fpath=self.onnx_t5_encoder.fpath,
            ),
        ]
        torch_models = [
            NetworkModel(
                name=T5ModelTRTConfig.NETWORK_FULL_NAME, fpath=pytorch_model_dir
            )
        ]

        return NetworkModels(torch=torch_models, onnx=onnx_models, trt=None)

    def cleanup(
        self,
        workspace: NNFolderWorkspace,
        keep_onnx_model: bool = True,
        keep_pytorch_model: bool = True,
    ) -> None:
        """
        Cleans up the working directory and leaves models if available.
        Should not assume any functions from the framework class has been called.
        Return:
            None
        """
        # Clean-up generated files
        if not keep_onnx_model:
            if self.onnx_t5_decoder is not None:
                self.onnx_t5_decoder.cleanup()
            if self.onnx_t5_encoder is not None:
                self.onnx_t5_encoder.cleanup()

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
                self.torch_t5_dir,
                prompt="Confirm you want to delete downloaded pytorch model folder?",
            )

        if not keep_pytorch_model and not keep_onnx_model:
            workspace.cleanup(force_remove=False)

    def setup_tokenizer_and_model(
        self,
        metadata: NetworkMetadata,
        network_fpaths: NetworkModels,
    ):
        tokenizer = T5Tokenizer.from_pretrained(metadata.variant)

        # By default, huggingface model structure is one giant file.
        t5_torch_fpath = network_fpaths.torch[0].fpath
        config = T5Config(
            use_cache=metadata.other.kv_cache,
            num_layers=T5ModelTRTConfig.NUMBER_OF_LAYERS[metadata.variant],
        )
        t5_model = T5ForConditionalGeneration(config).from_pretrained(t5_torch_fpath)

        t5_torch_encoder = T5EncoderTorchFile.TorchModule(t5_model.encoder)
        t5_torch_decoder = T5DecoderTorchFile.TorchModule(
            t5_model.decoder, t5_model.lm_head, t5_model.config
        )

        return tokenizer, t5_torch_encoder, t5_torch_decoder

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
        benchmarking_args: T5BenchmarkingArgs = None,
    ) -> Union[NetworkResult, BenchmarkingResult]:

        tokenizer, t5_torch_encoder, t5_torch_decoder = self.setup_tokenizer_and_model(metadata, network_fpaths)

        # Prepare the input tokens and find out output sequence length..
        if not benchmarking_mode:
            output_seq_len = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
            input_ids = tokenizer([inference_input] * batch_size, padding=True, return_tensors="pt").input_ids
        else:
            max_seq_len = T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant]
            input_seq_len = benchmarking_args.input_seq_len if benchmarking_args.input_seq_len > 0 else max_seq_len
            output_seq_len = benchmarking_args.output_seq_len if benchmarking_args.output_seq_len > 0 else max_seq_len
            input_ids = torch.randint(0, T5ModelTRTConfig.VOCAB_SIZE, (batch_size, input_seq_len))

        encoder_last_hidden_state, encoder_e2e_time = encoder_inference(
            t5_torch_encoder, input_ids, timing_profile, use_cuda=(not use_cpu)
        )

        # Need to feed the decoder a new empty input_ids for text generation. 
        decoder_output_len = output_seq_len // 2 if (not metadata.other.kv_cache) else 1

        decoder_input_ids = torch.full(
            (batch_size, decoder_output_len), tokenizer.convert_tokens_to_ids(tokenizer.pad_token), dtype=torch.int32
        )

        _, decoder_e2e_time = decoder_inference(
            t5_torch_decoder,
            expand_inputs_for_beam_search(decoder_input_ids, num_beams) if num_beams > 1 else decoder_input_ids,
            expand_inputs_for_beam_search(encoder_last_hidden_state, num_beams) if num_beams > 1 else encoder_last_hidden_state,
            timing_profile,
            use_cache=metadata.other.kv_cache,
        )
        
        if num_beams == 1:
            decoder_output, full_e2e_runtime = full_inference_greedy(
                t5_torch_encoder,
                t5_torch_decoder,
                input_ids,
                tokenizer,
                timing_profile,
                max_length=output_seq_len,
                min_length=T5ModelTRTConfig.MIN_OUTPUT_LENGTH[metadata.variant] if not benchmarking_mode else output_seq_len,
                use_cuda=(not use_cpu),
                batch_size=batch_size,
                use_cache=metadata.other.kv_cache,
            )
        else:
            decoder_output, full_e2e_runtime = full_inference_beam(
                t5_torch_encoder,
                t5_torch_decoder,
                input_ids,
                tokenizer,
                timing_profile,
                num_beams=num_beams,
                max_length=output_seq_len,
                min_length=T5ModelTRTConfig.MIN_OUTPUT_LENGTH[metadata.variant] if not benchmarking_mode else output_seq_len,
                use_cuda=(not use_cpu),
                batch_size=batch_size,
                use_cache=metadata.other.kv_cache,
            )

        # Prepare runtime results.
        runtime=[
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

        # Skip result checking in benchmarking mode since the input data is random.
        if benchmarking_mode:
            return BenchmarkingResult(median_runtime=runtime, models=network_fpaths)

        # Remove the padding and end tokens.
        semantic_outputs = tokenizer.decode(
            decoder_output[-1, :], skip_special_tokens=True
        )

        if isinstance(semantic_outputs, list):
            semantic_outputs = " ".join(semantic_outputs).strip()

        return NetworkResult(
            input=inference_input,
            output_tensor=encoder_last_hidden_state,
            semantic_output=semantic_outputs,
            median_runtime=runtime,
            models=network_fpaths,
        )

    def execute_calculate_perplexity(
        self,
        metadata: NetworkMetadata,
        network_fpaths: NetworkModels,
        encoder_input: str,
        decoder_input: str,
    ):
        tokenizer, t5_torch_encoder, t5_torch_decoder = self.setup_tokenizer_and_model(metadata, network_fpaths)
        encoder_input_ids = tokenizer([encoder_input], padding=True, return_tensors="pt").input_ids
        decoder_input_ids = tokenizer([decoder_input], padding=True, return_tensors="pt").input_ids
        perplexity = calculate_perplexity(
            t5_torch_encoder, t5_torch_decoder, tokenizer, encoder_input_ids, decoder_input_ids,
            T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
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
                    assert len(network_input) == len(perplexity_reference), "Encoder and decoder inputs must pair up"
                    for ei, di in zip(network_input, perplexity_reference):
                        ppl_results.append(
                            self.execute_calculate_perplexity(
                                metadata, network_fpaths, ei, di
                            )
                        )
            else:
                benchmarking_args = T5BenchmarkingArgs(args.input_seq_len, args.output_seq_len)
                inference_results = self.execute_inference(
                    metadata, network_fpaths, None, timing_profile, use_cpu, batch_size, args.num_beams, True, benchmarking_args
                )
        finally:
            self.cleanup(workspace, keep_onnx_model, keep_pytorch_model)

        return inference_results, ppl_results


# Entry point
RUN_CMD = T5FHuggingFace()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
