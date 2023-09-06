#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time

# Add syspath for custom library
if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(filepath, os.pardir)
    sys.path.append(project_root)

# polygraphy
from polygraphy.backend.trt import Profile

# tensorrt
import tensorrt as trt

# torch
import torch

# huggingface
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoProcessor

# tensorrt
from tensorrt import PreviewFeature

# TRT-HuggingFace
from NNDF.interface import TRTInferenceCommand
from NNDF.networks import (
    NetworkMetadata,
    NetworkModels,
    NetworkModel,
    Precision,
)

from NNDF.general_utils import confirm_folder_delete
from NNDF.tensorrt_utils import TRTNativeRunner, setup_benchmark_arg, CUASSERT
from NNDF.logger import G_LOGGER

from Seq2Seq.trt import Seq2SeqTRTDecoder
from Vision2Seq.Vision2SeqModelConfig import Vision2SeqModelTRTConfig
from Vision2Seq.export import Vision2SeqModelClass
from cuda import cudart

class Vision2SeqTRTEncoder(TRTNativeRunner):
    """TRT implemented network interface that can be used to measure inference time."""

    def __init__(
        self,
        trt_engine_file: str,
        network_metadata: NetworkMetadata,
        config: Vision2SeqModelTRTConfig,
        nvtx_verbose: bool,
    ):
        super().__init__(trt_engine_file, network_metadata, config, nvtx_verbose)

        self.data_type = torch.float16 if network_metadata.precision.fp16 else torch.float32
        self.main_input_name = "pixel_values"
        self.device = torch.device("cuda")

        self.image_embeds = torch.zeros(
            config.batch_size*config.num_positions*config.hidden_size,
            dtype=self.data_type,
            device=self.device
        )
        self.trt_context.set_tensor_address("image_embeds", self.image_embeds.data_ptr())

    def forward(self, pixel_values: torch.Tensor):

        # Check if the input data is on CPU (which usually means the PyTorch does not support current GPU).
        is_cpu_mode = (pixel_values.device == torch.device("cpu"))

        if is_cpu_mode:
            pixel_values = pixel_values.flatten().contiguous().cuda()

        if self.data_type == torch.float16:
            pixel_values = pixel_values.half()

        self.trt_context.set_tensor_address(self.main_input_name, pixel_values.data_ptr())
        self.trt_context.set_input_shape(self.main_input_name, pixel_values.shape)

        assert self.trt_context.all_shape_inputs_specified

        # Launch TRT inference.
        self.trt_context.execute_async_v3(self.stream)
        CUASSERT(cudart.cudaStreamSynchronize(self.stream))

        last_hidden_state = self.image_embeds[:self.config.batch_size * self.config.num_positions * self.config.hidden_size].view(self.config.batch_size, self.config.num_positions, self.config.hidden_size)

        if is_cpu_mode:
            last_hidden_state = last_hidden_state.cpu()

        return BaseModelOutput(last_hidden_state = last_hidden_state)

class Vision2SeqTRT(TRTInferenceCommand):
    def __init__(
        self,
        config_class=Vision2SeqModelTRTConfig,
        description="Runs trt results for Vision2Seq model.",
        model_classes=Vision2SeqModelClass,
        **kwargs
    ):
        super().__init__(
            network_config=config_class,
            description=description,
            model_classes=model_classes,
            **kwargs
        )
        self.onnx_encoder = None
        self.onnx_decoder = None
        self.encoder_class = Vision2SeqTRTEncoder
        self.decoder_class = Seq2SeqTRTDecoder # The decoder part (the text model) could be reusing Seq2Seq

    def process_framework_specific_arguments(
        self,
        disable_preview_dynamic_shapes: bool = False,
        dynamic_batch: bool = False,
        min_dynamic_batch: int = None,
        max_dynamic_batch: int = None,
        encoder_engine: str = None,
        decoder_engine: str = None,
        use_timing_cache: bool = False,
        nvtx_verbose: bool = False,
        **kwargs
    ):
        self.encoder_hidden_size = self.config.hidden_size
        self.disable_preview_dynamic_shapes = disable_preview_dynamic_shapes
        self.dynamic_batch = dynamic_batch
        self.opt_input_seq_len = self.config.opt_input_length
        self.opt_output_seq_len = self.config.opt_output_length

        # Ensure validity of batch size being built
        if self.dynamic_batch:
            min_dynamic_batch = int(setup_benchmark_arg(min_dynamic_batch, "min_dynamic_batch", 1))
            assert min_dynamic_batch <= self.config.batch_size, \
                "min_dynamic_batch {} should be <= batch_size {}".format(min_dynamic_batch, self.config.batch_size)
            self.min_dynamic_batch = min_dynamic_batch

            max_dynamic_batch = int(setup_benchmark_arg(max_dynamic_batch, "max_dynamic_batch", self.config.batch_size))
            assert self.config.batch_size <= max_dynamic_batch, \
                "max_dynamic_batch {} should be >= batch_size {}".format(max_dynamic_batch, self.config.batch_size)
            self.max_dynamic_batch = max_dynamic_batch

        self.min_batch_size = self.min_dynamic_batch if self.dynamic_batch else self.config.batch_size
        self.max_batch_size = self.max_dynamic_batch if self.dynamic_batch else self.config.batch_size

        self.min_expand_size = self.config._compute_expand_size(self.min_batch_size, self.config.num_beams)
        self.opt_expand_size = self.config._compute_expand_size(self.config.batch_size, self.config.num_beams)
        self.max_expand_size = self.config._compute_expand_size(self.max_batch_size, self.config.num_beams)

        self.use_generator = False

        self.workspace.set_encoder_engine_path(encoder_engine)
        self.workspace.set_decoder_engine_path(decoder_engine)

        self.use_timing_cache = use_timing_cache
        self.timing_cache = self.workspace.get_timing_cache() if self.use_timing_cache else None

        self.embedding_size_per_head = self.config.d_kv

        # In building the engine, setting nvtx verbose level does not affect performance, so always set to True.
        self.nvtx_verbose_build = True

        self.nvtx_verbose_inference = nvtx_verbose # In inference, nvtx verbose level may affect performance.

        return kwargs

    def setup_tokenizer_and_model(self):
        """
        Set up tokenizer and TRT engines for TRT Runner.
        """
        # self.tokenizer = self.download_tokenizer()
        processor = AutoProcessor.from_pretrained(self.config.metadata.variant)
        self.tokenizer = processor.tokenizer
        
        t0 = time.time()
        # Check whether user passed engine
        if self.check_engine_inputs_valid() and self.setup_engines_from_path(
            encoder_engine_fpath=self.workspace.encoder_engine_path,
            decoder_engine_fpath=self.workspace.decoder_engine_path,
        ):
            G_LOGGER.info("TRT engine loaded successful from arguments. Engine loading time: {:.4f}s".format(time.time() - t0))
        else:
            G_LOGGER.info("Cannot load existing TRT engines from arguments. Attempt to obtain from onnx model.")
            self.workspace.create_onnx_folders()
            self.load_onnx_model()
            self.setup_engines_from_onnx()
            G_LOGGER.info("TRT engine successfully obtained from onnx models. Total engine loading/building time: {:.4f}s".format(time.time() - t0))

        trt_models = [
            NetworkModel(
                name=self.config.NETWORK_DECODER_SEGMENT_NAME,
                fpath=self.decoder_engine.fpath,
            )
        ]

        trt_models.append(
            NetworkModel(
                name=self.config.NETWORK_ENCODER_SEGMENT_NAME,
                fpath=self.encoder_engine.fpath,
            )
        )

        return NetworkModels(torch=None, onnx=None, trt=trt_models)

    def check_engine_inputs_valid(self):
        """
        Check whether all engines are valid.
        """
        encoder_engine_fpath = self.workspace.encoder_engine_path
        decoder_engine_fpath = self.workspace.decoder_engine_path
        is_encoder_valid = encoder_engine_fpath is not None and os.path.exists(encoder_engine_fpath)
        is_decoder_valid = decoder_engine_fpath is not None and os.path.exists(decoder_engine_fpath)

        return is_encoder_valid and is_decoder_valid

    def setup_engines_from_path(
        self,
        encoder_engine_fpath = None,
        decoder_engine_fpath = None,
    ):
        """
        Check whether user has passed in all required TRT engine name.
        If user passed valid TRT engines, will skip onnx export and use engine directly.
        """
        if decoder_engine_fpath is None:
            return False
        if encoder_engine_fpath is None:
            return False

        try:
            self.decoder_engine = self.config.decoder_classes["engine"](decoder_engine_fpath, self.metadata)
            self.decoder = self.decoder_class(
                self.decoder_engine, self.metadata, self.config.text_config, self.nvtx_verbose_inference
            )

            self.encoder_engine = self.config.encoder_classes["engine"](encoder_engine_fpath, self.metadata)
            if self.config.use_fp32_encoder:
                encoder_metadata = self.metadata._replace(precision=Precision(fp16=False))
            else:
                encoder_metadata = self.metadata

            self.encoder = self.encoder_class(
                self.encoder_engine, encoder_metadata, self.config, self.nvtx_verbose_inference
            )

            return True
        except Exception as e:
            G_LOGGER.error("Cannot proceed with the provided engine. Attempt to generate from onnx. Reason is: {}".format(str(e)))

        return False

    def setup_engines_from_onnx(self) -> None:
        """
        Set up TRT engines from onnx files.
        """

        # Generate optimization profiles.
        # non-benchmarking mode: opt profile length is by default half of the max profile
        # benchmarking mode: user can specify opt and max profile by flags.
        #                    If no additional benchmarking flags are provided, it will just use n_positions for max coverage

        # Convert ONNX models to TRT engines.
        if not self.benchmarking_mode:
            engine_tag = "bs{}".format(self.config.batch_size)
        # When user does not input any profile_max_len, use seq as tag, both max are config max
        elif self.seq_tag:
            # When user inputs dynamic batch, enable engine reuse in future with different batch size.
            if self.dynamic_batch:
                engine_tag = "minbs{}-maxbs{}-inseq{}-outseq{}".format(self.min_dynamic_batch,
                                                                       self.max_dynamic_batch,
                                                                       self.opt_input_seq_len,
                                                                       self.opt_output_seq_len)
            else:
                engine_tag = "bs{}-inseq{}-outseq{}".format(self.config.batch_size,
                                                            self.opt_input_seq_len,
                                                            self.opt_output_seq_len)
        # When user input profile_max_len, reuse the engine for future use with different seq_len
        else:
            # When user inputs dynamic batch, enable engine reuse in future with different batch size.
            if self.dynamic_batch:
                engine_tag = "minbs{}-maxbs{}-inmax{}-outmax{}".format(self.min_dynamic_batch,
                                                                       self.max_dynamic_batch,
                                                                       self.config.max_input_profile_length,
                                                                       self.config.max_output_profile_length)
            else:
                engine_tag = "bs{}-inmax{}-outmax{}".format(self.config.batch_size,
                                                            self.config.max_input_profile_length,
                                                            self.config.max_output_profile_length)
                
        if self.config.num_beams > 1:
            engine_tag += "-beam{}".format(self.config.num_beams)

        preview_features = [PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]

        if self.disable_preview_dynamic_shapes:
            engine_tag += "-noPreviewFasterDynamicShapes"
        else:
            preview_features.append(PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)

        # Set up decoder engine
        decoder_profiles = self._setup_decoder_profiles()

        decoder_engine_path = self.workspace.get_engine_fpath_from_onnx(self.onnx_decoder.fpath, engine_tag, self.engine_postfix)

        G_LOGGER.info("Setting up decoder engine in {}...".format(decoder_engine_path))

        # Print trtexec command for decoder
        G_LOGGER.debug(self._decoder_trtexec_command(decoder_engine_path))

        self.decoder_engine = self.onnx_decoder.as_trt_engine(
            decoder_engine_path,
            profiles=decoder_profiles,
            preview_features=preview_features,
            nvtx_verbose=self.nvtx_verbose_build,
            timing_cache=self.timing_cache,
        )

        self.decoder = self.decoder_class(
            self.decoder_engine, self.metadata, self.config.text_config, self.nvtx_verbose_inference,
        )
        self.workspace.set_decoder_engine_path(decoder_engine_path)

        # Set up encoder if needed
        encoder_profiles = self._setup_encoder_profiles()
        encoder_engine_path = self.workspace.get_engine_fpath_from_onnx(self.onnx_encoder.fpath, engine_tag, self.engine_postfix).replace(f"-beam{self.config.num_beams}", "")

        G_LOGGER.info("Setting up encoder engine in {}...".format(encoder_engine_path))

        if self.config.use_fp32_encoder:
            encoder_metadata = self.metadata._replace(precision=Precision(fp16=False))
        else:
            encoder_metadata = self.metadata

        # Print trtexec command for encoder
        G_LOGGER.debug(self._encoder_trtexec_command(encoder_engine_path, encoder_metadata))

        self.encoder_engine = self.onnx_encoder.as_trt_engine(
            encoder_engine_path, # encoder engine name not affected by beam search
            profiles=encoder_profiles,
            preview_features=preview_features,
            nvtx_verbose=self.nvtx_verbose_build,
            timing_cache=self.timing_cache,
        )

        self.encoder = self.encoder_class(
            self.encoder_engine, encoder_metadata, self.config, self.nvtx_verbose_inference,
        )
        self.workspace.set_encoder_engine_path(encoder_engine_path)

    def _setup_decoder_profiles(self):
        if not self.config.use_cache:
            decoder_profile = Profile().add(
                "input_ids",
                min=(self.min_expand_size, 1),
                opt=(self.opt_expand_size, self.opt_output_seq_len),
                max=(self.max_expand_size, self.config.max_output_profile_length),
            )
            if self.config.use_mask:
                decoder_profile = decoder_profile.add(
                    "attention_mask",
                    min=(self.min_expand_size, 1),
                    opt=(self.opt_expand_size, self.opt_output_seq_len),
                    max=(self.max_expand_size, self.config.max_output_profile_length),
                )
            decoder_profile.add(
                "image_embeds",
                min=(self.min_expand_size, self.config.num_positions, self.encoder_hidden_size),
                opt=(self.opt_expand_size, self.config.num_positions, self.encoder_hidden_size),
                max=(self.max_expand_size, self.config.num_positions, self.encoder_hidden_size),
            )

            decoder_profiles = [decoder_profile]
        else:
            self_attn_profile = {
                "min": (self.min_expand_size, self.config.num_heads, 0, self.embedding_size_per_head),
                "opt": (self.opt_expand_size, self.config.num_heads, self.opt_output_seq_len - 1, self.embedding_size_per_head),
                "max": (self.max_expand_size, self.config.num_heads, self.config.max_output_profile_length - 1, self.embedding_size_per_head),
            }

            decoder_profile_generation = Profile().add(
                "input_ids",
                min=(self.min_expand_size, 1),
                opt=(self.opt_expand_size, 1),
                max=(self.max_expand_size, 1),
            )

            if self.config.use_mask:
                decoder_profile_generation = decoder_profile_generation.add(
                    "attention_mask",
                    min=(self.min_expand_size, 1),
                    opt=(self.opt_expand_size, self.opt_output_seq_len),
                    max=(self.max_expand_size, self.config.max_output_profile_length),
                )

            decoder_profile_generation.add(
                "image_embeds",
                min=(self.min_expand_size, self.config.num_positions, self.encoder_hidden_size),
                opt=(self.opt_expand_size, self.config.num_positions, self.encoder_hidden_size),
                max=(self.max_expand_size, self.config.num_positions, self.encoder_hidden_size),
            )


            for i in range(self.config.num_decoder_layers):
                decoder_profile_generation = decoder_profile_generation.add(
                    f"past_key_values.{i}.self.key",
                    **self_attn_profile
                ).add(
                    f"past_key_values.{i}.self.value",
                    **self_attn_profile
                )

            decoder_profiles = [decoder_profile_generation]

            # Decoder only model has "context phase" that is only used for the 1st decoder phase.
            if not self.config.text_config.is_encoder_decoder:
                # Context phase takes various-length input_ids with no kv cache and generates initial cache for subsequent decoding steps.
                decoder_profile_context = Profile().add(
                    "input_ids",
                    min=(self.min_expand_size, 1),
                    opt=(self.opt_expand_size, self.opt_input_seq_len),
                    max=(self.max_expand_size, self.config.max_input_profile_length),
                )
                if self.config.use_mask:
                    decoder_profile_context = decoder_profile_context.add(
                        "attention_mask",
                        min=(self.min_expand_size, 1),
                        opt=(self.opt_expand_size, self.opt_input_seq_len),
                        max=(self.max_expand_size, self.config.max_input_profile_length),
                    )

                self_attn_profile_context = {
                    "min": (self.min_expand_size, self.config.num_heads, 0, self.embedding_size_per_head),
                    "opt": (self.opt_expand_size, self.config.num_heads, 0, self.embedding_size_per_head),
                    "max": (self.max_expand_size, self.config.num_heads, 0, self.embedding_size_per_head),
                }
                for i in range(self.config.num_decoder_layers):
                    decoder_profile_context = decoder_profile_context.add(
                        f"past_key_values.{i}.self.key",
                        **self_attn_profile_context
                    ).add(
                        f"past_key_values.{i}.self.value",
                        **self_attn_profile_context
                    )

                decoder_profile_context.add(
                    "image_embeds",
                    min=(self.min_expand_size, self.config.num_positions, self.encoder_hidden_size),
                    opt=(self.opt_expand_size, self.config.num_positions, self.encoder_hidden_size),
                    max=(self.max_expand_size, self.config.num_positions, self.encoder_hidden_size),
                )

                decoder_profiles.append(decoder_profile_context)

        return decoder_profiles

# Used to create decoder trtexec command string for debugging accuracy/performance.
    def _decoder_trtexec_command(self, decoder_engine_path):
        command = f"trtexec --onnx={self.onnx_decoder.fpath} --verbose --saveEngine={decoder_engine_path} "
        min_shape = "--minShapes="
        opt_shape = "--optShapes="
        max_shape = "--maxShapes="
        if not self.config.use_cache:
            min_shape += f"'input_ids':{self.min_expand_size}x1"
            opt_shape += f"'input_ids':{self.opt_expand_size}x{self.opt_output_seq_len}"
            max_shape += f"'input_ids':{self.max_expand_size}x{self.config.max_output_profile_length}"
            if self.config.use_mask:
                min_shape += f",'attention_mask':{self.min_expand_size}x1"
                opt_shape += f",'attention_mask':{self.opt_expand_size}x{self.opt_output_seq_len}"
                max_shape += f",'attention_mask':{self.max_expand_size}x{self.config.max_output_profile_length}"

            min_shape += f",'image_embeds':{self.min_expand_size}x{self.config.num_positions}x{self.encoder_hidden_size}"
            opt_shape += f",'image_embeds':{self.opt_expand_size}x{self.config.num_positions}x{self.encoder_hidden_size}"
            max_shape += f",'image_embeds':{self.max_expand_size}x{self.config.num_positions}x{self.encoder_hidden_size}"

            command += f"{min_shape} {opt_shape} {max_shape} "

        else:
            min_shape += f"'input_ids':{self.min_expand_size}x1"
            opt_shape += f"'input_ids':{self.opt_expand_size}x1"
            max_shape += f"'input_ids':{self.max_expand_size}x1"
            if self.config.use_mask:
                min_shape += f",'attention_mask':{self.min_expand_size}x1"
                opt_shape += f",'attention_mask':{self.opt_expand_size}x{self.opt_output_seq_len}"
                max_shape += f",'attention_mask':{self.max_expand_size}x{self.config.max_output_profile_length}"

            min_shape += f",'image_embeds':{self.min_expand_size}x{self.config.num_positions}x{self.encoder_hidden_size}"
            opt_shape += f",'image_embeds':{self.opt_expand_size}x{self.config.num_positions}x{self.encoder_hidden_size}"
            max_shape += f",'image_embeds':{self.max_expand_size}x{self.config.num_positions}x{self.encoder_hidden_size}"

            for i in range(self.config.num_decoder_layers):
                k = f",'past_key_values.{i}.self.key'"
                v = f",'past_key_values.{i}.self.value'"
                kv_min_str = f":{self.min_expand_size}x{self.config.num_heads}x0x{self.embedding_size_per_head}"
                kv_opt_str = f":{self.opt_expand_size}x{self.config.num_heads}x{self.opt_output_seq_len-1}x{self.embedding_size_per_head}"
                kv_max_str = f":{self.max_expand_size}x{self.config.num_heads}x{self.config.max_output_profile_length-1}x{self.embedding_size_per_head}"
                min_shape += k + kv_min_str + v + kv_min_str
                opt_shape += k + kv_opt_str + v + kv_opt_str
                max_shape += k + kv_max_str + v + kv_max_str

            command += f"{min_shape} {opt_shape} {max_shape} "

        if self.metadata.precision.fp16:
            # For version compatibility. Older version of TRT does not support int64.
            int_type = "int64" if hasattr(trt,"int64") else "int32"
            inputIO = f"--inputIOFormats={int_type}:chw" # input_ids
            if self.config.use_mask:
                inputIO += f",{int_type}:chw" # attention_mask
            outputIO = "--outputIOFormats=fp16:chw" # logits
            inputIO += ",fp16:chw" # image_embeds
            if self.config.use_cache:
                for _ in range(self.config.num_decoder_layers):
                    kv_precision = ",fp16:chw,fp16:chw"
                    inputIO += kv_precision # self attention
                    outputIO += kv_precision # self attention
                    if self.config.is_encoder_decoder:
                        inputIO += kv_precision # cross attention

            command += f"--fp16 --precisionConstraints=obey {inputIO} {outputIO} "

        if self.timing_cache is not None:
            command += f"--timingCacheFile={self.timing_cache} "

        return command

    def _setup_encoder_profiles(self):
        if not self.config.is_encoder_decoder:
            raise NotImplementedError("You are setting encoder profile for a non encoder_decoder model!")
        encoder_profile = Profile().add(
            "pixel_values",
            min=(self.min_batch_size, 3, self.config.image_size, self.config.image_size),
            opt=(self.config.batch_size, 3, self.config.image_size, self.config.image_size),
            max=(self.max_batch_size, 3, self.config.image_size, self.config.image_size),
        )

        return [encoder_profile]

    # Used to create encoder trtexec command string for debugging accuracy/performance.
    def _encoder_trtexec_command(self, encoder_engine_path, encoder_metadata):
        command = f"trtexec --onnx={self.onnx_encoder.fpath} --verbose --saveEngine={encoder_engine_path} "
        min_shape = f"--minShapes='pixel_values':{self.min_batch_size}x3x{self.config.image_size}x{self.config.image_size}"
        opt_shape = f"--optShapes='pixel_values':{self.config.batch_size}x3x{self.config.image_size}x{self.config.image_size}"
        max_shape = f"--maxShapes='pixel_values':{self.max_batch_size}x3x{self.config.image_size}x{self.config.image_size}"
        command += f"{min_shape} {opt_shape} {max_shape} "

        if encoder_metadata.precision.fp16:
            inputIO = "--inputIOFormats=fp16:chw" # pixel_values
            outputIO = "--outputIOFormats=fp16:chw" # image_embeds
            command += f"--fp16 --precisionConstraints=obey {inputIO} {outputIO} "

        if self.timing_cache is not None:
            command += f"--timingCacheFile={self.timing_cache} "

        return command

    def cleanup(self) -> None:

        # Deactivates context
        if self.encoder:
            self.encoder.release()
        if self.decoder:
            self.decoder.release()

        if not self.keep_trt_engine:
            self.decoder_engine.cleanup()
            self.encoder_engine.cleanup()

        if not self.keep_onnx_model:
            if self.onnx_decoder:
                self.onnx_decoder.cleanup()
            if self.onnx_encoder:
                self.onnx_encoder.cleanup()

        if not self.keep_torch_model and self.workspace.torch_path is not None:

            confirm_folder_delete(
                self.workspace.torch_path,
                prompt="Confirm you want to delete downloaded pytorch model folder?",
            )

        if not self.keep_trt_engine:
            self.workspace.cleanup(force_remove=False)


RUN_CMD = Vision2SeqTRT()

if __name__ == "__main__":
    result = RUN_CMD()
    print("Results: {}".format(result))
