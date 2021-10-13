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

from typing import List

# huggingface
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# TRT-HuggingFace
from NNDF.interface import FrameworkCommand
from NNDF.networks import (
    NetworkResult,
    NetworkMetadata,
    NetworkRuntime,
    NetworkModels,
    NetworkModel,
    TimingProfile,
)
from T5.export import T5EncoderTorchFile, T5DecoderTorchFile
from T5.T5ModelConfig import T5ModelTRTConfig
from T5.measurements import decoder_inference, encoder_inference, full_inference_greedy
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

    def execute_inference(
        self,
        metadata: NetworkMetadata,
        network_fpaths: NetworkModels,
        inference_input: str,
        timing_profile: TimingProfile,
    ) -> NetworkResult:

        # Execute some tests
        tokenizer = T5Tokenizer.from_pretrained(metadata.variant)
        input_ids = tokenizer(inference_input, return_tensors="pt").input_ids

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

        encoder_last_hidden_state, encoder_e2e_median_time = encoder_inference(
            t5_torch_encoder, input_ids, timing_profile
        )
        _, decoder_e2e_median_time = decoder_inference(
            t5_torch_decoder, input_ids, encoder_last_hidden_state, timing_profile
        )
        decoder_output_greedy, full_e2e_median_runtime = full_inference_greedy(
            t5_torch_encoder,
            t5_torch_decoder,
            input_ids,
            tokenizer,
            timing_profile,
            max_length=T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
        )

        # Remove the padding and end tokens.
        semantic_outputs = tokenizer.convert_ids_to_tokens(
            decoder_output_greedy.tolist()[0]
        )[1:-1]
        remove_underscore = "".join(
            [s.replace("\u2581", " ") for s in semantic_outputs]
        )

        return NetworkResult(
            input=inference_input,
            output_tensor=encoder_last_hidden_state,
            semantic_output=remove_underscore.strip(),
            median_runtime=[
                NetworkRuntime(
                    name=T5ModelTRTConfig.NETWORK_DECODER_SEGMENT_NAME,
                    runtime=decoder_e2e_median_time,
                ),
                NetworkRuntime(
                    name=T5ModelTRTConfig.NETWORK_ENCODER_SEGMENT_NAME,
                    runtime=encoder_e2e_median_time,
                ),
                NetworkRuntime(
                    name=T5ModelTRTConfig.NETWORK_FULL_NAME,
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


# Entry point
RUN_CMD = T5FHuggingFace()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
