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

"""
Interface classes required for each registered network script.
"""

import argparse

from abc import ABCMeta, abstractmethod
from typing import List, Union
# For time purpose
import time

# NNDF
from NNDF.networks import (
    BenchmarkingResult,
    NetworkResult,
    Precision,
    NetworkMetadata,
    DeprecatedCache,
    NetworkCheckpointResult,
    NNConfig,
    NetworkModels,
    TimingProfile,
    NetworkRuntime,
)
from NNDF.logger import G_LOGGER
from NNDF.general_utils import NNFolderWorkspace, confirm_folder_delete, measure_python_inference_code
from NNDF.torch_utils import use_cuda
from NNDF.tensorrt_utils import TRTNativeRunner, setup_benchmark_arg

# From HuggingFace
from transformers import (
    AutoModelForCausalLM, # For decoder-only models
    AutoModelForSeq2SeqLM, # For encoder_decoder models
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    GenerationMixin,
)

import os
import torch

# Program-wide constants for passing in valid frameworks.
FRAMEWORK_NATIVE = "native"
FRAMEWORK_TENSORRT = "trt"
FRAMEWORK_ONNXRT = "onnxrt"
VALID_FRAMEWORKS = [
    FRAMEWORK_NATIVE,
    FRAMEWORK_ONNXRT,
    FRAMEWORK_TENSORRT
]

class NetworkCommand(metaclass=ABCMeta):
    """Base class that each network script's command module should inherit."""

    description = "NetworkCommand"

    DEFAULT_ITERATIONS = 10
    DEFAULT_NUMBER = 1
    DEFAULT_WARMUP = 3
    DEFAULT_DURATION = 0.0
    DEFAULT_PERCENTILE = 50

    def __init__(
        self,
        config_class: NNConfig,
        description: str,
        model_classes,
        variant: str = None,
        **model_args
    ):
        self.config_class = config_class
        self.description = description
        self.framework_name = None
        self.model_classes = model_classes
        self._parser = argparse.ArgumentParser(description=description, conflict_handler="resolve")
        self._args = None

        # These parameters need to be set by `setup_tokenizer_and_models`
        self.encoder = None
        self.decoder = None
        self.torch_model = None
        self.tokenizer = None
        self.models = None

        if variant is not None:
            self.setup_environment(variant=variant, **model_args)

    def process_framework_specific_arguments(self, **kwargs):
        pass

    def setup_environment(
        self,
        variant: str,
        working_dir: str = "temp",
        batch_size: int = 1,
        num_beams: int = 1,
        use_cache: bool = True,
        enable_kv_cache: bool = False,
        fp16: bool = True,
        verbose: bool = False,
        info: bool = False,
        iterations: int = DEFAULT_ITERATIONS,
        number: int = DEFAULT_NUMBER,
        warmup: int = DEFAULT_WARMUP,
        duration: int = DEFAULT_DURATION,
        percentile: int = DEFAULT_PERCENTILE,
        benchmarking_mode: bool = False,
        input_seq_len: int = None,
        output_seq_len: int = None,
        input_profile_max_len: int = None,
        output_profile_max_len: int = None,
        cleanup: bool = False,
        torch_dir: str = None,
        encoder_onnx: str = None,
        decoder_onnx: str = None,
        cache_generator_onnx: str = None,
        skip_checkpoint_load: bool = False,
        engine_postfix: str = "",
        use_mask: bool = False,
        **kwargs,
    ) -> None:
        """
        Uses Arguments from command line or user specified to setup config for the model.
        """

        if verbose:
            G_LOGGER.setLevel(level=G_LOGGER.DEBUG)
        elif info:
            G_LOGGER.setLevel(level=G_LOGGER.INFO)

        if variant is None:
            G_LOGGER.error("You should specify --variant to run HuggingFace demo")
            return

        if enable_kv_cache:
            G_LOGGER.warning("--enable-kv-cache has been deprecated to --use-cache to fit HuggingFace config.")
            use_cache = True

        if self._args is not None:
            G_LOGGER.info("Setting up environment with arguments: {}".format(self._args))
        else:
            G_LOGGER.info("User-customized API is called")


        self.metadata = NetworkMetadata(
            variant=variant,
            precision=Precision(fp16=fp16),
            use_cache=use_cache,
            num_beams=num_beams,
            batch_size=batch_size,
            other=DeprecatedCache(kv_cache=use_cache)
        )

        self.config = self.config_class(
            metadata = self.metadata
        )

        def get_hf_config():
            """Gets HF config with correct max sequence length limits in benchmarking mode."""
            hf_config = AutoConfig.from_pretrained(
                variant,
                use_cache=use_cache,
                trust_remote_code=True,
            )

            if benchmarking_mode:

                # Return supremum of a max value and valid user input values
                N_POSITION_EMBEDDINGS_FALLBACK = 1024

                # Different models have different keyword names for max position embeddings.
                # Loop over and see if HF consumes keyword.
                possible_position_kw = ( "n_positions", "max_position_embeddings" )

                original_n_positions = None
                for k in possible_position_kw:
                    if hasattr(hf_config, k):
                        original_n_positions = getattr(hf_config, k)

                if original_n_positions is None:
                    G_LOGGER.warning("Unable to set n_positions for the model using {} as hints. "
                                     "Overriding the field `n_positions` instead, assigning it "
                                     "to {}".format(", ".join(possible_position_kw), N_POSITION_EMBEDDINGS_FALLBACK))
                    setattr(hf_config, "n_positions", N_POSITION_EMBEDDINGS_FALLBACK)
                    original_n_positions = N_POSITION_EMBEDDINGS_FALLBACK

                def _n_positions_hint():
                    hint_cli_args = (
                        input_seq_len,
                        output_seq_len,
                        input_profile_max_len,
                        output_profile_max_len,
                        original_n_positions,
                    )
                    def consider_if_valid(x):
                        return x if x else -1

                    return max(map(lambda x : consider_if_valid(x), hint_cli_args))

                n_positions_hints = _n_positions_hint()
                n_positions = setup_benchmark_arg(
                    kwargs.get("n_positions"),
                    "n_positions",
                    n_positions_hints
                )

                if original_n_positions < n_positions:
                    setattr(hf_config, k, n_positions)
                    self.config.ignore_mismatched_sizes = True

            return hf_config

        hf_config = get_hf_config()
        self.config.from_hf_config(hf_config)
        self.benchmarking_mode = benchmarking_mode
        # Not able to set it inside config class. Therefore needs to be set up here.
        self.config.precision = torch.float16 if self.config.fp16 else torch.float32

        # Not able to specify model classes in config and therefore needs to set up here.
        self.config.set_model_classes(self.model_classes)

        if benchmarking_mode:
            self.checkpoint = None
            # Overwrite some fields for generation
            self.seq_tag = input_profile_max_len is None and output_profile_max_len is None
            self.process_benchmarking_args(
                input_seq_len=input_seq_len,
                output_seq_len=output_seq_len,
                input_profile_max_len=input_profile_max_len,
                output_profile_max_len=output_profile_max_len,
            )

        if use_mask:
            self.config.use_mask = True

        skip_checkpoint_load = skip_checkpoint_load or benchmarking_mode or (self._args is None)
        if not skip_checkpoint_load:
            self.checkpoint = self.load_nn_semantic_checkpoint()
            network_input = list(self.checkpoint.inputs())
            # If there is input which is list, using maximum input list size to batch size

            self.config.input_case_size = max([len(n) if isinstance(n, list) else 1 for n in network_input ])
            # update config batch size
            self.config.batch_size = self.config.input_case_size * self.config.batch_size
            self.config.expand_size = self.config._compute_expand_size(self.config.batch_size, self.config.num_beams)

        # User defined variables for generation
        generation_config = GenerationConfig.from_model_config(hf_config)
        generation_config.max_length = self.config.max_output_length
        generation_config.min_length = self.config.min_output_length
        generation_config.num_beams = num_beams
        generation_config.use_cache = use_cache

        self.config.set_generation_config(generation_config)

        self.workspace = NNFolderWorkspace(
            self.config, working_dir
        )

        self.timing_profile = TimingProfile(
            iterations=iterations,
            number=number,
            warmup=warmup,
            duration=duration,
            percentile=percentile,
        )

        self.keep_torch_model = not cleanup
        self.keep_onnx_model = not cleanup
        self.keep_trt_engine = not cleanup

        # If user specifies location, uses user-specified paths
        if torch_dir is not None:
            self.workspace.set_torch_path(torch_dir)
        if encoder_onnx is not None:
            self.workspace.set_encoder_onnx_path(encoder_onnx)
        if decoder_onnx is not None:
            self.workspace.set_decoder_onnx_path(decoder_onnx)
        if cache_generator_onnx is not None:
            self.workspace.set_cross_attn_generator_onnx_path(cache_generator_onnx)

        # Some user-defined engine postfix.
        self.engine_postfix = engine_postfix

        self.model_path_args = self.process_framework_specific_arguments(**kwargs)

    def process_benchmarking_args(
        self,
        input_seq_len,
        output_seq_len,
        # For TRT only
        input_profile_max_len = None,
        output_profile_max_len = None,
    ):
        # This is the largest seq len that the model could ever been used
        n_positions = self.config.n_positions
        # User must provide either a pair of profile_max_len or a profile of seq_len for input/output
        if input_profile_max_len is None or output_profile_max_len is None:
            if input_seq_len is None or output_seq_len is None:
                raise RuntimeError("Please provide [input/output]_seq_len or provide [input/output]_profile_max_len for TRT")

        input_profile_max_len = setup_benchmark_arg(input_profile_max_len, "input_profile_max_len", n_positions)
        output_profile_max_len = setup_benchmark_arg(output_profile_max_len, "output_profile_max_len", n_positions)
        input_seq_len = setup_benchmark_arg(input_seq_len, "input_seq_len", input_profile_max_len // 2)
        output_seq_len = setup_benchmark_arg(output_seq_len, "output_seq_len", output_profile_max_len // 2)

        # Assert to ensure the validity of benchmarking arguments
        assert input_seq_len <= input_profile_max_len, "input_seq_len should <= input_profile_max_len = {} for benchmarking mode".format(input_profile_max_len)
        assert output_seq_len <= output_profile_max_len, "output_seq_len should <= output_profile_max_len = {} for benchmarking mode".format(output_profile_max_len)
        assert input_profile_max_len <= n_positions, "Model n_positions restrict input_profile_max_len <= {} for benchmark mode".format(n_positions)
        assert output_profile_max_len <= n_positions, "Model n_positions restrict output_profile_max_len <= {} for benchmark mode".format(n_positions)

        self.config.max_input_length = input_seq_len
        self.config.opt_input_length = input_seq_len
        self.config.max_input_profile_length = input_profile_max_len

        self.config.min_output_length = output_seq_len
        self.config.max_output_length = output_seq_len
        self.config.max_decoder_length = 1 if (self.config.use_cache and self.config.is_encoder_decoder) else self.config.max_output_length
        self.config.opt_output_length = output_seq_len
        self.config.max_output_profile_length = output_profile_max_len


    def add_args(self) -> None:
        general_group = self._parser.add_argument_group("general")
        general_group.add_argument(
            "--verbose", "-v",
            help="Display verbose logs.",
            action="store_true"
        )
        general_group.add_argument(
            "--info", help="Display info logs.", action="store_true"
        )
        general_group.add_argument(
            "--cleanup",
            help="Cleans up user-specified workspace. Can not be cleaned if external files exist in workspace.",
            action="store_true",
        )
        general_group.add_argument(
            "--working-dir", "-wd",
            help="Location of where to save the model and other downloaded files.",
            required=True,
        )

        model_config_group = self._parser.add_argument_group("model")
        model_config_group.add_argument(
            "--batch-size", "-b",
            help="Chosen batch size for given network",
            required=False,
            type=int,
            default=1
        )

        model_config_group.add_argument(
            "--variant", "-m",
            help="model to generate",
            required=True,
        )
        model_config_group.add_argument(
            "--use-cache",
            "-kv",
            help="Enable KV cache",
            action="store_true",
            default=False,
        )
        model_config_group.add_argument(
            "--enable-kv-cache",
            help="Deprecated: Please use --use-cache.",
            action="store_true",
            default=False,
        )
        model_config_group.add_argument(
            "--num-beams",
            "-nb",
            type=int,
            default=1,
            help="Enables beam search during decoding."
        )

        model_config_group.add_argument(
            "--fp16",
            action="store_true",
            help="Uses fp16 tactics.",
            default=False,
        )

        model_config_group.add_argument(
            "--use-mask",
            action="store_true",
            help="Pass attention_mask as external inputs instead of auto generated for better multi-batch accuracy.",
            default=False,
        )

        timing_group = self._parser.add_argument_group("inference measurement")
        timing_group.add_argument(
            "--iterations",
            type=int,
            help="Number of iterations to measure.",
            default=self.DEFAULT_ITERATIONS,
        )
        timing_group.add_argument(
            "--number",
            type=int,
            help="Number of actual inference cycles per iterations.",
            default=self.DEFAULT_NUMBER,
        )
        timing_group.add_argument(
            "--warmup",
            type=int,
            help="Number of warmup iterations before actual measurement occurs.",
            default=self.DEFAULT_WARMUP,
        )
        timing_group.add_argument(
            "--duration",
            type=float,
            help="Minimal duration of inference iterations to measure.",
            default=self.DEFAULT_DURATION,
        )
        timing_group.add_argument(
            "--percentile",
            type=int,
            help="Key percentile number for time measurement.",
            default=self.DEFAULT_PERCENTILE,
        )

        torch_group = self._parser.add_argument_group("torch model path")
        torch_group.add_argument(
            "--torch-dir",
            default=None,
            help="Path to PyTorch model. If None is supplied, will attempt to pull from HuggingFace",
        )

        onnx_group = self._parser.add_argument_group("onnx models path")
        onnx_group.add_argument(
            "--encoder-onnx",
            default=None,
            help="Path to ONNX encoder. Only use for encoder-decoder models. If None is supplied, scripts will generate them from HuggingFace.",
        )

        onnx_group.add_argument(
            "--decoder-onnx",
            default=None,
            help="Path to ONNX decoder. If None is supplied, scripts will generate them from HuggingFace.",
        )

        onnx_group.add_argument(
            "--cache-generator-onnx",
            default=None,
            help="Path to ONNX cross-attention cache generator. Only use with use-cache mode. If None is supplied, scripts will generate them from HuggingFace.",
        )

    @abstractmethod
    def setup_tokenizer_and_model(self) -> NetworkModels:
        """
        This function is required for every subclass to setup proper tokenizer and models to execute inference.
        """
        raise NotImplementedError(
            f"Make sure that a `setup_tokenizer_and_model` function is correctly implemented in {self.__class__.__module__} to"
            f" enable accuracy check for {self.__class__}"
        )

    def download_tokenizer(self):
        """
        This function is a helper function to download tokenizer from HuggingFace

        Returns:
            tokenizer (transformers.AutoTokenizer): tokenizer for the model.
        """
        if self.tokenizer:
            return self.tokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.variant,
            trust_remote_code=True,
        )
        # Set pad_token = eos_token because eos token will be discarded by attention_mask
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        return tokenizer

    def load_torch_model(self):
        """
        This function load the PyTorch model from HuggingFace.

        Returns:
            model(torch.Module): PyTorch model downloaded from HuggingFace

        """

        if self.torch_model:
            return self.torch_model

        t0 = time.time()
        torch_dir = self.workspace.torch_path
        if torch_dir is None or not os.path.isdir(torch_dir):
            torch_dir = self.workspace.create_pytorch_folder()

        torch_model = None
        hf_config = self.config.hf_config
        assert hf_config.use_cache == self.config.use_cache

        # Use this flag to load torch model if and only if benchmarking seqlen > model n_positions
        # There is a known issue in HuggingFace in ignore_mismatched_sizes that is fixed in 4.31.0
        # https://github.com/huggingface/transformers/issues/22563.

        ignore_mismatched_sizes = self.config.ignore_mismatched_sizes

        def _load_torch_model_from_hf(model_loc):
            if self.config.is_encoder_decoder:
                _torch_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_loc,
                    config=hf_config,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    trust_remote_code=True,
                )
            else:
                _torch_model = AutoModelForCausalLM.from_pretrained(
                    model_loc,
                    config=hf_config,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    trust_remote_code=True,
                )
            return _torch_model

        def _download_and_save_torch_model_from_hf(variant):
            try:
                _torch_model = _load_torch_model_from_hf(variant)
            except Exception as e:
                raise RuntimeError(f"Unable to download {self.config.variant} from HuggingFace Hub. Reason: {str(e)}")

            _torch_model.save_pretrained(torch_dir)
            G_LOGGER.info("Pytorch Model saved to {}".format(torch_dir))
            return _torch_model

        if not os.path.exists(os.path.join(torch_dir, "config.json")):
            torch_model = _download_and_save_torch_model_from_hf(self.config.variant)
        else:
            G_LOGGER.info(
                "Frameworks file already exists, skipping download and loading from file instead."
            )
            try:
                torch_model = _load_torch_model_from_hf(torch_dir)
            except Exception as e:
                G_LOGGER.warning(f"Fails to load model from {torch_dir}. Reason: {str(e)}. Attempt to redownload from HuggingFace.")
                torch_model = _download_and_save_torch_model_from_hf(self.config.variant)

        G_LOGGER.info("PyTorch model loading time is {:.4f}s".format(time.time() - t0))
        return torch_model

    def load_onnx_model(self):
        """
        Load ONNX model.
        First attempt to load models from user-provided arguments;
        If does not exist, attempt to look for ONNX files in workspace;
        If could not find ONNX files, attempt to convert from PyTorch.

        Returns:
            True

        Sets:
            self.onnx_decoder: DecoderONNXFile
            self.onnx_encoder: EncoderONNXFile if encoder_decoder, None otherwise
            self.onnx_cross_attn_cache_generator: CrossAttnCacheGeneratorONNXFile if encoder_decoder and use_cache, None otherwise
        """
        G_LOGGER.info("Attempt to load ONNX models from arguments...")
        if self.check_onnx_inputs_valid():
            self.load_onnx_model_from_workspace()
            G_LOGGER.info("ONNX models found from arguments.")
            return True

        G_LOGGER.info("ONNX models not found from arguments. Attempt to search load ONNX models from existing workspace...")

        self.workspace.create_onnx_folders()
        if self.check_onnx_inputs_valid():
            self.load_onnx_model_from_workspace()
            G_LOGGER.info("ONNX models found from workspace.")
            return True

        G_LOGGER.info("ONNX model not in existing workspace. Attempt to export ONNX models from PyTorch...")
        torch_model = self.load_torch_model()
        # For TRT, we try to always use CPU models to export to onnx as fp32.
        torch_model = torch_model.cpu()
        t0 = time.time()
        torch_decoder = self.config.decoder_classes["torch"](torch_model, network_metadata = self.metadata)
        self.onnx_decoder = torch_decoder.as_onnx_model(
            self.workspace.decoder_onnx_path, force_overwrite=False, config=self.config
        )

        if self.config.is_encoder_decoder:
            torch_encoder = self.config.encoder_classes["torch"](torch_model, network_metadata = self.metadata)
            self.onnx_encoder = torch_encoder.as_onnx_model(
                self.workspace.encoder_onnx_path, force_overwrite=False, config=self.config
            )
        if self.use_generator:
            torch_cross_attn_cache_generator = self.config.cross_attn_cache_generator_classes["torch"](torch_model, network_metadata = self.metadata)
            self.onnx_cross_attn_cache_generator = torch_cross_attn_cache_generator.as_onnx_model(
                self.workspace.cross_attn_generator_onnx_path, force_overwrite=False, config=self.config
            )
        G_LOGGER.info("ONNX models successfully exported from PyTorch. ONNX export and post-processing time: {:.4f}s".format(time.time() - t0))
        return True

    def check_onnx_inputs_valid(self):
        """
        Helper method for load_onnx_model. Checks whether onnx inputs are valid paths to onnx files.
        """
        encoder_onnx_fpath = self.workspace.encoder_onnx_path
        decoder_onnx_fpath = self.workspace.decoder_onnx_path
        cache_generator_onnx_fpath = self.workspace.cross_attn_generator_onnx_path
        is_encoder_valid = encoder_onnx_fpath is not None and os.path.exists(encoder_onnx_fpath)
        is_decoder_valid = decoder_onnx_fpath is not None and  os.path.exists(decoder_onnx_fpath)
        is_generator_valid = cache_generator_onnx_fpath is not None and os.path.exists(cache_generator_onnx_fpath)
        if self.config.is_encoder_decoder:
            if self.config.use_cache:
                return is_encoder_valid and is_decoder_valid and is_generator_valid
            else:
                return is_encoder_valid and is_decoder_valid

        return is_decoder_valid

    def load_onnx_model_from_workspace(self):
        """
        Helper method for load_onnx_model. Loads onnx model from workspace, assuming ONNX model is already there.
        """
        self.onnx_decoder = self.config.decoder_classes["onnx"](self.workspace.decoder_onnx_path, self.metadata)
        if self.config.is_encoder_decoder:
            self.onnx_encoder = self.config.encoder_classes["onnx"](self.workspace.encoder_onnx_path, self.metadata)
        if self.use_generator:
            self.onnx_cross_attn_cache_generator = self.config.cross_attn_cache_generator_classes["onnx"](self.workspace.cross_attn_generator_onnx_path, self.metadata)


    def load_nn_semantic_checkpoint(self) -> object:
        """
        Loads the NNSemanticCheckpoint instance from checkpoint.toml file.
        """
        # Differ import so that interface file can use used without
        # dependency install for our testing.
        from NNDF.checkpoints import NNSemanticCheckpoint
        checkpoint = NNSemanticCheckpoint(
            "checkpoint.toml",
            framework=self.framework_name,
            network_name=self.config.network_name,
            metadata=self.metadata,
            skip_multibatch=(not self.config.use_mask) # Skip multi-batch tests if not using attention_mask,
        )
        return checkpoint

    @use_cuda
    def decoder_inference(
        self,
        input_ids,
        attention_mask = None,
        encoder_outputs = None,
        use_cuda = True
    ):
        G_LOGGER.info(f"Running decoder inference...")

        if isinstance(self.decoder, TRTNativeRunner) and self.config.is_encoder_decoder:
            self.decoder.set_encoder_hidden_states(encoder_outputs.last_hidden_state)

        if self.config.use_mask and attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        decoder_stmt = lambda: self.decoder(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask
        )

        decoder_e2e_time = measure_python_inference_code(decoder_stmt, self.timing_profile)
        decoder_output = decoder_stmt()

        return (decoder_output, decoder_e2e_time)

    @use_cuda
    def encoder_inference(
        self,
        input_ids,
        attention_mask = None,
        use_cuda = True
    ):
        G_LOGGER.info(f"Running encoder inference...")
        if self.config.use_mask and attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        encoder_stmt = lambda: self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        encoder_e2e_time = measure_python_inference_code(encoder_stmt, self.timing_profile)
        encoder_output = encoder_stmt()
        return (encoder_output, encoder_e2e_time)

    @use_cuda
    def full_inference(
        self,
        input_ids,
        attention_mask = None,
        early_stopping = True,
        use_cuda = True
    ):

        G_LOGGER.info(f"Running full inference...")
        if self.config.use_mask and attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        def _e2e():
            with torch.no_grad():
                encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                ) if self.encoder else None

                mask_kwargs = {"attention_mask": attention_mask} if self.config.use_mask else {}
                if self.decoder:
                    decoder_output = self.decoder.generate(
                        input_ids,
                        num_beams=self.config.num_beams,
                        early_stopping=early_stopping,
                        eos_token_id=self.config.eos_token_id,
                        pad_token_id=self.config.pad_token_id,
                        use_cache=self.config.use_cache,
                        encoder_outputs=encoder_outputs,
                        min_length=self.config.min_output_length,
                        max_length=self.config.max_output_length,
                        **mask_kwargs,
                    )

                    return decoder_output

                return encoder_outputs

        measurement_function = _e2e

        full_e2e_time = measure_python_inference_code(measurement_function, self.timing_profile)
        model_outputs = _e2e()

        return (model_outputs, full_e2e_time)

    @use_cuda
    def generate(
        self,
        input_str: str = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        use_cuda: bool = True,
    ):
        # If models are not set, need to setup tokenizer and models to run inference
        if self.models is None:
            self.models = self.setup_tokenizer_and_model()

        if input_str is None and input_ids is None:
            raise RuntimeError("Please provide either input_str or input_ids for generate")

        # If no input_ids, use input_str to tokenize
        if input_ids is None:
            tokenizer_output = self.tokenizer([input_str] * self.config.batch_size, padding=True, return_tensors="pt")
            input_ids = tokenizer_output.input_ids
            if self.config.use_mask:
                attention_mask = tokenizer_output.attention_mask
        elif self.config.use_mask and attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if use_cuda:
            input_ids = input_ids.to("cuda")
            if self.config.use_mask:
                attention_mask = attention_mask.to("cuda")

        encoder_outputs = None
        if self.config.is_encoder_decoder:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)

        mask_kwargs = {"attention_mask": attention_mask} if self.config.use_mask else {}
        decoder_outputs = self.decoder.generate(
            input_ids,
            num_beams=self.config.num_beams,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            use_cache=self.config.use_cache,
            encoder_outputs=encoder_outputs,
            min_length=self.config.min_output_length,
            max_length=self.config.max_output_length,
            **mask_kwargs,
        )

        semantic_outputs = self.tokenizer.decode(
            decoder_outputs[-1, :], skip_special_tokens=True
        )

        return decoder_outputs, semantic_outputs

    @use_cuda
    def execute_inference(
        self,
        inference_input: Union[str, list],
        use_cuda: bool = True
    ) -> Union[NetworkResult, BenchmarkingResult]:

        if self.models is None:
            self.models = self.setup_tokenizer_and_model()

        attention_mask = None
        # Prepare the input tokens and find out output sequence length.
        if not self.benchmarking_mode:
            if isinstance(inference_input, list):
                tokenizer_output = self.tokenizer(inference_input * (self.config.batch_size // self.config.input_case_size), padding=True, return_tensors="pt")
            else:
                # TODO: Ideally should be self.config.batch_size // self.config.input_case_size, but this would violate the shape restriction
                tokenizer_output = self.tokenizer([inference_input] * self.config.batch_size, padding=True, return_tensors="pt")

            input_ids = tokenizer_output.input_ids
            if self.config.use_mask:
                attention_mask = tokenizer_output.attention_mask
        else:
            # opt_input_length = input_seq_len in benchmarking mode
            input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, self.config.opt_input_length))
            if self.config.use_mask:
                attention_mask = torch.ones_like(input_ids)

        decoder_output, full_e2e_runtime = self.full_inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cuda=use_cuda,
        )

        if self.config.is_encoder_decoder:
            encoder_outputs, encoder_e2e_time = self.encoder_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cuda=use_cuda,
            )

        # Run decoder once at a reasonable seq length
        # Note: For kv cache, because we do not have kv cache information, we are always running in context mode.
        if not self.config.is_encoder_decoder:
            decoder_input_ids = input_ids
        else:
            decoder_input_ids = input_ids[:,-1:]
            if self.config.use_mask:
                attention_mask = attention_mask[:,-1:]

        # Expand decoder_inputs if using beam search
        decoder_input_ids, model_kwargs = GenerationMixin._expand_inputs_for_generation(
            expand_size=self.config.num_beams,
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            is_encoder_decoder=self.config.is_encoder_decoder,
            encoder_outputs=encoder_outputs if self.config.is_encoder_decoder else None,
        )

        expanded_encoder_outputs = model_kwargs["encoder_outputs"]
        expanded_attention_mask = model_kwargs["attention_mask"]

        _, decoder_e2e_time = self.decoder_inference(
            input_ids=decoder_input_ids,
            attention_mask=expanded_attention_mask,
            encoder_outputs=expanded_encoder_outputs,
            use_cuda=use_cuda,
        )

        # Prepare runtime results.
        runtime = [
            NetworkRuntime(
                name=self.config_class.NETWORK_FULL_NAME,
                runtime=full_e2e_runtime,
            ),
            NetworkRuntime(
                name=self.config_class.NETWORK_DECODER_SEGMENT_NAME,
                runtime=decoder_e2e_time,
            ),
        ]

        if self.config.is_encoder_decoder:
            runtime.append(
                NetworkRuntime(
                    name=self.config_class.NETWORK_ENCODER_SEGMENT_NAME,
                    runtime=encoder_e2e_time,
                )
            )

        # Skip result checking in benchmarking mode since the input data is random.
        if self.benchmarking_mode:
            return BenchmarkingResult(median_runtime=runtime, models=self.models)

        if isinstance(inference_input, list):
            semantic_outputs = list()
            for batch_index in range(decoder_output.shape[0]):
                semantic_outputs.append(self.tokenizer.decode(
                    decoder_output[batch_index, :], skip_special_tokens=True
        ))
        else:
            # Check that each expanded batch returns the same result
            for batch_index in range(1, decoder_output.shape[0]):
                if not torch.equal(decoder_output[0, :], decoder_output[batch_index,:]):
                    # This usually indicate some accuracy issue with multi-batch inference
                    G_LOGGER.warning(f"batches do not equal for identical inputs. \
                                       decoder_output = {decoder_output}. \
                                       input_ids = {input_ids}.")
                    break

            # Remove the padding and end tokens.
            semantic_outputs = self.tokenizer.decode(
                decoder_output[0, :], skip_special_tokens=True
            )

        return NetworkResult(
            input=inference_input,
            output_tensor=decoder_output,
            semantic_output=semantic_outputs,
            median_runtime=runtime,
        )

    @use_cuda
    def calculate_perplexity(
        self,
        input_str: str,
        reference_str: str,
        use_cuda: bool = True,
    ):
        """
        Each child class should have a `calculate_perplexity` that takes in the result str and reference str for perplexity calculation.

        """
        G_LOGGER.warning(
            f"Make sure that a `calculate_perplexity` function is correctly implemented in {self.__class__.__module__} to"
            f" enable accuracy check for {self.__class__}. Default=None"
        )

        return None


    def run(self) -> Union[List[NetworkResult], BenchmarkingResult]:
        """
        Main entry point of our function which compiles and generates our model data for command-line mode.
        The general process for the commands are all the same:
        (1) Download the model
        (2) Run either checkpoint or benchmark
        (3) Returns the result
        """
        t0 = time.time()
        self.models = self.setup_tokenizer_and_model()
        t1 = time.time()
        G_LOGGER.info("setup_tokenizer_and_model() takes {:.4f}s in total.".format(t1 - t0))

        if self.checkpoint is None:
            perplexity_reference = None
        else:
            network_input = list(self.checkpoint.inputs())

            perplexity_reference = list(self.checkpoint.labels())

        inference_results = []
        ppl_results = []

        try:
            if not self.benchmarking_mode:
                for ninput in network_input:
                    inference_results.append(
                        self.execute_inference(ninput, use_cuda=self.use_cuda)
                    )
                if perplexity_reference is not None:
                    assert len(network_input) == len(perplexity_reference), "Encoder and decoder inputs must pair up"
                    for ei, di in zip(network_input, perplexity_reference):
                        if ei == "":
                            raise ValueError("Perplexity reference encoder input is empty")
                        if di == "":
                            raise ValueError("Perplexity reference decoder input is empty")
                        ppl_results.append(
                            self.calculate_perplexity(ei, di, use_cuda=self.use_cuda)
                        )
            else:
                inference_results = self.execute_inference(inference_input = None)

        finally:
            self.cleanup()

        t2 = time.time()
        G_LOGGER.info("Inference session is {:.4f}s in total.".format(t2 - t1))

        return inference_results, ppl_results

    def __call__(self):
        t0 = time.time()
        self.add_args()
        self._args = self._parser.parse_args()

        self.setup_environment(
            **vars(self._args),
            benchmarking_mode=(self._args.action == "benchmark")
        )
        t1 = time.time()
        G_LOGGER.info("Set up environment takes {:.4f}s.".format(t1 - t0))

        if self.benchmarking_mode:
            network_results = self.run()
            return network_results
        else:
            network_results, ppl_results = self.run()
            return NetworkCheckpointResult(
                network_results=network_results,
                accuracy=self.checkpoint.accuracy(network_results),
                perplexity=(sum(ppl_results) / len(ppl_results) if not (None in ppl_results) else None),
                models=self.models
            )

    def setup_chat(self):
        self.add_args()
        t0 = time.time()
        self._args = self._parser.parse_args()
        self.setup_environment(
            **vars(self._args),
            skip_checkpoint_load=True,
        )
        self.models = self.setup_tokenizer_and_model()
        G_LOGGER.info("Total time to setup is: {:.4f}s".format(time.time() - t0))


class FrameworkCommand(NetworkCommand):
    """Base class that is associated with Frameworks related scripts."""

    def __init__(self, network_config: NNConfig, description: str, model_classes, **kwargs):
        super().__init__(network_config, description, model_classes, **kwargs)
        self.framework_name = FRAMEWORK_NATIVE

    def __call__(self):
        return super().__call__()

    def add_args(self):
        super().add_args()
        device_group = self._parser.add_argument_group("device")
        device_group.add_argument(
            "--cpu",
            help="Run inference using CPU for frameworks.",
            action="store_true",
        )

    def cleanup(self) -> None:
        """
        Cleans up the working directory and leaves models if available.
        Should not assume any functions from the framework class has been called.
        Return:
            None
        """

        if not self.keep_torch_model:
            # Using rmtree can be dangerous, have user confirm before deleting.
            confirm_folder_delete(
                self.workspace.torch_path,
                prompt="Confirm you want to delete downloaded pytorch model folder?",
            )

            self.workspace.cleanup(force_remove=False)

class TRTInferenceCommand(NetworkCommand):
    """Base class that is associated with Polygraphy related scripts."""

    def __init__(
        self,
        network_config: NNConfig,
        description: str,
        model_classes,
        **kwargs,
    ):
        super().__init__(network_config, description, model_classes, **kwargs)
        self.framework_name = FRAMEWORK_TENSORRT
        # TRT always use GPU
        self.use_cuda = True

    def __call__(self):
        return super().__call__()

    def add_args(self):
        super().add_args()
        trt_group = self._parser.add_argument_group("trt")
        trt_group.add_argument(
            "--disable-preview-dynamic-shapes",
            help="Disable the FASTER_DYNAMIC_SHAPES_0805 preview feature when building the TensorRT engine",
            action="store_true",
            default=False,
        )

        trt_group.add_argument(
            "--dynamic-batch",
            help="Build TensorRT engines with dynamic batch sizes.",
            default=False,
            action="store_true",
        )

        trt_group.add_argument(
            "--min-dynamic-batch",
            default=None,
            help="Minimum batch size for engines built with dynamic batch size.",
        )

        trt_group.add_argument(
            "--max-dynamic-batch",
            default=None,
            help="Maximum batch size for engines built with dynamic batch size.",
        )

        engine_group = self._parser.add_argument_group("trt engine")
        engine_group.add_argument(
            "--encoder-engine",
            default=None,
            help="Path to encoder TRT engine. Only use for encoder-decoder models. If None is supplied, scripts will generate from TensorRT.",
        )

        engine_group.add_argument(
            "--decoder-engine",
            default=None,
            help="Path to decoder TRT engine. If None is supplied, scripts will generate from TensorRT.",
        )

        engine_group.add_argument(
            "--cache-generator-engine",
            default=None,
            help="Path to cross-attention cache generator TRT engine. Only use with use-cache mode. If None is supplied, scripts will generate from TensorRT.",
        )

        engine_group.add_argument(
            "--use-timing-cache",
            default=False,
            help="Use Timing Cache could speed up engine building",
            action="store_true",
        )

        engine_group.add_argument(
            "--nvtx-verbose",
            default=False,
            help="nvtx verbosity in inference stage.",
            action="store_true",
        )

        engine_group.add_argument(
            "--engine-postfix",
            default="",
            help="Postfix in engine file name to encode customized engine info, e.g. GPU, Platform, cuda, etc.",
        )

class OnnxRTCommand(NetworkCommand):
    """ONNX Runtime command."""

    def __init__(
        self,
        network_config: NNConfig,
        description: str,
        model_classes,
        **kwargs
    ):
        super().__init__(network_config, description, model_classes, **kwargs)
        self.framework_name = FRAMEWORK_ONNXRT
        self.use_cuda = False

    def __call__(self):
        return super().__call__()
