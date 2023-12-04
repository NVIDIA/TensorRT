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

from abc import abstractmethod
from typing import List, Union
# For time purpose
import time

# NNDF
from NNDF.networks import (
    TopNAccuracy,
    AccuracyResult,
    AccuracyMetadata,
    BenchmarkingResult,
    NetworkResult,
    Precision,
    NetworkMetadata,
    NetworkCheckpointResult,
    NNConfig,
    NetworkModels,
    TimingProfile,
    NetworkRuntime,
    InferenceResult,
)
from NNDF.logger import G_LOGGER
from NNDF.general_utils import NNFolderWorkspace, confirm_folder_delete, measure_python_inference_code
from NNDF.torch_utils import use_cuda
from NNDF.tensorrt_utils import TRTNativeRunner, setup_benchmark_arg

# From HuggingFace
from transformers import (
    AutoModelForCausalLM, # For decoder-only models
    AutoModelForSeq2SeqLM, # For encoder_decoder models
    AutoModelForVision2Seq, # For Vison2Seq models
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    GenerationMixin,
)

import os, gc
import torch

MAX_LOG_PPL = 12.0

# Program-wide constants for passing in valid frameworks.
FRAMEWORK_NATIVE = "native"
FRAMEWORK_TENSORRT = "trt"
FRAMEWORK_ONNXRT = "onnxrt"
VALID_FRAMEWORKS = [
    FRAMEWORK_NATIVE,
    FRAMEWORK_ONNXRT,
    FRAMEWORK_TENSORRT
]

class NetworkCommand:
    """Base class that each network script's command module should inherit."""

    description = "NetworkCommand"

    DEFAULT_ITERATIONS = 10
    DEFAULT_NUMBER = 1
    DEFAULT_WARMUP = 3
    DEFAULT_DURATION = 0.0
    DEFAULT_PERCENTILE = 50

    DEFAULT_NUM_SAMPLES = 20
    DEFAULT_TOPN = 5
    DEFAULT_TOKENS_TO_GENERATE = 2

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
        action: str = "run",
        working_dir: str = "temp",
        # Model args
        batch_size: int = 1,
        num_beams: int = 1,
        use_cache: bool = True,
        enable_kv_cache: bool = False,
        fp16: bool = True,
        use_mask: bool = False,
        # Log level
        verbose: bool = False,
        info: bool = False,
        # Timing Profile args
        iterations: int = DEFAULT_ITERATIONS,
        number: int = DEFAULT_NUMBER,
        warmup: int = DEFAULT_WARMUP,
        duration: int = DEFAULT_DURATION,
        percentile: int = DEFAULT_PERCENTILE,
        # Benchmarking args
        input_seq_len: int = None,
        output_seq_len: int = None,
        input_profile_max_len: int = None,
        output_profile_max_len: int = None,
        # Workspace management args
        cleanup: bool = False,
        torch_dir: str = None,
        encoder_onnx: str = None,
        decoder_onnx: str = None,
        cache_generator_onnx: str = None,
        engine_postfix: str = "",
        # Accuracy args
        num_samples: int = DEFAULT_NUM_SAMPLES,
        topN: int = DEFAULT_TOPN,
        tokens_to_generate: int = DEFAULT_TOKENS_TO_GENERATE,
        **kwargs,
    ) -> None:
        """
        The main entrance point for configuring models.
        Uses arguments from command line or user specified to setup config for the model.
        """

        if verbose:
            G_LOGGER.setLevel(level=G_LOGGER.DEBUG)
        elif info:
            G_LOGGER.setLevel(level=G_LOGGER.INFO)

        if variant is None:
            raise RuntimeError("You should specify --variant to run HuggingFace demo")

        if enable_kv_cache:
            G_LOGGER.warning("--enable-kv-cache has been deprecated to --use-cache to fit HuggingFace config.")
            use_cache = True

        if self._args is not None:
            G_LOGGER.info("Setting up environment with arguments: {}".format(self._args))
            skip_checkpoint_load = False
        else:
            # Skip checkpoint load for notebook/API users.
            skip_checkpoint_load = True

        self.action = action

        self.metadata = NetworkMetadata(
            variant=variant,
            precision=Precision(fp16=fp16),
            use_cache=use_cache,
            num_beams=num_beams,
            batch_size=batch_size,
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

            if action == "benchmark":

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
        # Not able to set it inside config class. Therefore needs to be set up here.
        self.config.precision = torch.float16 if self.config.fp16 else torch.float32
        if self.config.consume_image_embeds:
            self.config.text_config.precision = torch.float16 if self.config.text_config.fp16 else torch.float32

        # Not able to specify model classes in config and therefore needs to set up here.
        self.config.set_model_classes(self.model_classes)

        if use_mask:
            self.config.use_mask = True
            if self.config.consume_image_embeds: # For Vision2Seq
                self.config.text_config.use_mask = True

        if action == "benchmark":
            self.checkpoint = None
            # Overwrite some fields for generation
            self.seq_tag = input_profile_max_len is None and output_profile_max_len is None
            self.process_benchmarking_args(
                input_seq_len=input_seq_len,
                output_seq_len=output_seq_len,
                input_profile_max_len=input_profile_max_len,
                output_profile_max_len=output_profile_max_len,
            )

        elif action == "accuracy":
            self.accuracy_metadata = AccuracyMetadata(dataset="LAMBADA", num_samples=num_samples, tokens_to_generate=tokens_to_generate)
            # Default to Top1 +TopN + Top10 accuracy
            self.topN = {1, topN, 10}
            self.checkpoint = self.load_lambada_dataset(
                num_samples=num_samples,
                tokens_to_generate=tokens_to_generate,
            )

        elif not skip_checkpoint_load:
            self.checkpoint = self.load_nn_semantic_checkpoint()

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
        n_positions = self.config.text_config.n_positions if self.config.consume_image_embeds else self.config.n_positions
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
        if hasattr(tokenizer, "eos_token"):
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
        if not self.config.consume_image_embeds:
            # In BLIP/BLIPv2/pix2struct hf_config.text_config refers to the config for text decoder, but define use_cache in the config by the model itself. Thus we skip the config use_cache checking for Vison2Seq
            # Note: vision-encoder-decoder has different names (decoder instead of text_config)
            assert hf_config.use_cache == self.config.use_cache

        # Use this flag to load torch model if and only if benchmarking seqlen > model n_positions
        # There is a known issue in HuggingFace in ignore_mismatched_sizes that is fixed in 4.31.0
        # https://github.com/huggingface/transformers/issues/22563.

        ignore_mismatched_sizes = self.config.ignore_mismatched_sizes

        def _load_torch_model_from_hf(model_loc):
            if self.config.consume_image_embeds: # For Vision2Seq
                _torch_model = AutoModelForVision2Seq.from_pretrained(
                    model_loc,
                    config=hf_config,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    trust_remote_code=True,
                )
            elif self.config.is_encoder_decoder:
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

        # torch_model is no longer used. We need to remove it from CPU otherwise it takes 1xfp32 model size and thus affect TRT memory comsumption
        torch_model.cpu()
        del torch_model
        gc.collect()

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
        if self.config.consume_image_embeds:
            return is_encoder_valid and is_decoder_valid
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

    def load_lambada_dataset(
        self,
        num_samples,
        tokens_to_generate,
    ):
        """
        Loads LAMBADA dataset from online for accuracy checks.

        Args:
            num_samples: number of samples for the dataset.
            tokens_to_generate: number of tokens masked out for accuracy testing.

        Returns:
            NNLambadaCheckpoint
        """
        from NNDF.checkpoints import NNLambadaCheckpoint
        # LAMBADA dataset will be shared for all models, so put in a shared folder.
        demo_dir = os.path.dirname(os.getcwd())
        checkpoint = NNLambadaCheckpoint(
            base_dir=demo_dir,
            tokens_to_generate=tokens_to_generate,
            num_samples=num_samples,
            batch_size=self.config.batch_size,
            use_mask=self.config.use_mask,
        )

        return checkpoint

    @use_cuda
    def decoder_inference(
        self,
        input_ids,
        attention_mask = None,
        encoder_outputs = None,
        image_embeds = None,
        use_cuda = True
    ):
        G_LOGGER.info(f"Running decoder inference...")

        if isinstance(self.decoder, TRTNativeRunner) and self.config.is_encoder_decoder:
            if self.config.consume_image_embeds:
                self.decoder.set_image_embeds(image_embeds)
            else:
                self.decoder.set_encoder_hidden_states(encoder_outputs.last_hidden_state)

        if self.config.use_mask and attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        encoder_outputs_kwargs = {"image_embeds": image_embeds} if self.config.consume_image_embeds else {"encoder_outputs": encoder_outputs}

        decoder_stmt = lambda: self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **encoder_outputs_kwargs,
        )

        decoder_e2e_time = measure_python_inference_code(decoder_stmt, self.timing_profile)
        decoder_output = decoder_stmt()

        return (decoder_output, decoder_e2e_time)

    @use_cuda
    def encoder_inference(
        self,
        input_ids = None,
        attention_mask = None,
        pixel_values: torch.Tensor = None, # For Vision encoder in Vision2Seq models
        use_cuda = True
    ):
        G_LOGGER.info(f"Running encoder inference...")
        if self.config.use_mask and attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.config.consume_image_embeds:
            if pixel_values is None:
                raise RuntimeError("Please provide pixel_values for vision encoder in Vision2Seq models")
            else:
                encoder_stmt = lambda: self.encoder(
                    pixel_values=pixel_values,
                )
        else:
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
        pixel_values: torch.Tensor = None, # For Vision encoder in Vision2Seq models
        early_stopping = True,
        use_cuda = True
    ) -> InferenceResult:

        G_LOGGER.info(f"Running full inference...")
        if self.config.use_mask and attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.config.consume_image_embeds and pixel_values is None:
            raise RuntimeError("Please provide pixel_values for vision encoder in Vision2Seq models")

        decoder_input_ids=input_ids
        if self.config.consume_image_embeds: # For Vision2Seq (Specifically for BLIP)
            input_ids[:, 0] = self.config.text_config.bos_token_id
            decoder_input_ids=input_ids[:, :-1]
            if self.config.use_mask and attention_mask is not None:
                attention_mask=attention_mask[:, :-1]
            if self.config.fp16 and not isinstance(self.decoder, TRTNativeRunner):
                pixel_values = pixel_values.half()

        def _e2e():
            with torch.no_grad():
                if self.config.consume_image_embeds: # For Vision2Seq
                    encoder_outputs = self.encoder(
                        pixel_values=pixel_values,
                    )
                else:
                    encoder_outputs = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    ) if self.encoder else None

                mask_kwargs = {"attention_mask": attention_mask} if self.config.use_mask else {}
                encoder_outputs_kwargs = {"image_embeds": encoder_outputs[0]} if self.config.consume_image_embeds else {"encoder_outputs": encoder_outputs}

                if self.decoder:
                    decoder_output = self.decoder.generate(
                        input_ids=decoder_input_ids,
                        num_beams=self.config.num_beams,
                        early_stopping=early_stopping,
                        eos_token_id=self.config.eos_token_id,
                        pad_token_id=self.config.pad_token_id,
                        use_cache=self.config.use_cache,
                        min_length=self.config.min_output_length,
                        max_length=self.config.max_output_length,
                        **mask_kwargs,
                        **encoder_outputs_kwargs,
                    )

                    return decoder_output

                return encoder_outputs

        measurement_function = _e2e

        full_e2e_time = measure_python_inference_code(measurement_function, self.timing_profile)

        model_outputs = _e2e()

        return InferenceResult(
            output=model_outputs,
            runtime=full_e2e_time,
        )

    @use_cuda
    def generate(
        self,
        input = None,
        pixel_values: torch.Tensor = None, # For Vision encoder in Vision2Seq models
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        min_length: int = None,
        max_length: int = None,
        use_cuda: bool = True,
    ):
        # If models are not set, need to setup tokenizer and models to run inference
        if self.models is None:
            self.models = self.setup_tokenizer_and_model()

        if input is None and input_ids is None:
            raise RuntimeError("Please provide either input(list/str) or input_ids(torch.Tensor) for generate")

        if self.config.consume_image_embeds and pixel_values is None:
            raise RuntimeError("Please provide pixel_values for vision encoder in Vision2Seq models")

        # If no input_ids, use input_str to tokenize
        if input_ids is None:
            if not isinstance(input, list):
                input = [input] * self.config.batch_size

            tokenizer_output = self.tokenizer(input, padding=True, return_tensors="pt")
            input_ids = tokenizer_output.input_ids

            if self.config.consume_image_embeds: # For Vision2Seq (Specifically for BLIP)
                input_ids = input_ids[:, :-1]
                input_ids[:, 0] = self.config.text_config.bos_token_id

            if self.config.use_mask:
                attention_mask = tokenizer_output.attention_mask
                if self.config.consume_image_embeds: # For Vision2Seq (Specifically for BLIP)
                    attention_mask = attention_mask[:, :-1]

        elif self.config.use_mask and attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if self.config.use_mask:
            assert input_ids.shape[0] == attention_mask.shape[0], "batch_size of input_ids and attention_mask does not match."

        bs = input_ids.shape[0]
        if bs != self.config.batch_size:
            G_LOGGER.warning(f"Input bs {bs} != desired bs {self.config.batch_size}."
                " Please make sure you are running with framework or with --dynamic-batch with batch_size=max_dynamic_batch."
                " This behavior may lead to mismatched size or tensor OOM error."
                " Attempt to cast batch size."
            )

            self.decoder.config.batch_size = bs
            self.decoder.config.expand_size = bs * self.decoder.config.num_beams
            self.decoder.expand_size = bs * self.decoder.config.num_beams

        if use_cuda:
            input_ids = input_ids.to("cuda")
            if self.config.use_mask:
                attention_mask = attention_mask.to("cuda")
            if self.config.consume_image_embeds:
                pixel_values = pixel_values.to("cuda")

        encoder_outputs = None
        image_embeds = None
        if self.config.is_encoder_decoder:
            if self.config.consume_image_embeds: # Vision encoder
                if self.config.fp16 and not isinstance(self.decoder, TRTNativeRunner):
                    pixel_values = pixel_values.half()
                vision_outputs = self.encoder(pixel_values=pixel_values)
                image_embeds = vision_outputs[0]
            else:
                encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)

        mask_kwargs = {"attention_mask": attention_mask} if self.config.use_mask else {}
        vision2seq_kwargs = {}
        if self.config.consume_image_embeds:
            vision2seq_kwargs["image_embeds"] = image_embeds
            vision2seq_kwargs["early_stopping"] = self.config.text_config.early_stopping

        min_length = self.config.min_output_length if min_length is None else min_length
        max_length = self.config.max_output_length if max_length is None else max_length
        decoder_outputs = self.decoder.generate(
            input_ids=input_ids,
            num_beams=self.config.num_beams,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
            use_cache=self.config.use_cache,
            encoder_outputs=encoder_outputs,
            min_length=min_length,
            max_length=max_length,
            **mask_kwargs,
            **vision2seq_kwargs,
        )

        semantic_outputs = self.tokenizer.batch_decode(
            decoder_outputs, skip_special_tokens=True
        )

        return decoder_outputs, semantic_outputs


    @use_cuda
    def execute_inference(
        self,
        input: List[str] = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        use_cuda: bool = True
    ) -> InferenceResult:

        '''
        Measure encoder/decoder/e2e performance for a single example.

        Args:
            input: List[str]: example.
            input_ids: torch.Tensor
            attention_mask: torch.Tensor

        '''

        if input is None and input_ids is None:
            raise RuntimeError("Please provide either input(list of str) or input_ids(torch.Tensor) for execute_inference function.")

        if input:
            tokenizer_output = self.tokenizer(input, padding=True, return_tensors="pt")
            input_ids = tokenizer_output.input_ids
            if self.config.use_mask:
                attention_mask = tokenizer_output.attention_mask
        else:
            if self.config.use_mask and attention_mask is None:
                G_LOGGER.warn("attention_mask is None, but use_mask = True. Using trivial attention_mask.")
                attention_mask = torch.ones_like(input_ids)

        if self.config.consume_image_embeds:
            if pixel_values is None:
                G_LOGGER.warn("pixel_values is None but running Vision2Seq model. Using random image.")
                pixel_values = torch.rand(self.config.batch_size, 3, self.config.image_size, self.config.image_size)
            if self.config.fp16 and not isinstance(self.decoder, TRTNativeRunner):
                pixel_values = pixel_values.half()

        if use_cuda:
            input_ids = input_ids.to("cuda")
            if self.config.use_mask:
                attention_mask = attention_mask.to("cuda")
            if self.config.consume_image_embeds:
                pixel_values = pixel_values.to("cuda")

        decoder_output, full_e2e_runtime = self.full_inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cuda=use_cuda,
        )

        if self.config.is_encoder_decoder:
            encoder_outputs, encoder_e2e_time = self.encoder_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
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

        expanded_encoder_outputs = model_kwargs["encoder_outputs"] if not self.config.consume_image_embeds else None
        expanded_attention_mask = model_kwargs["attention_mask"]

        _, decoder_e2e_time = self.decoder_inference(
            input_ids=decoder_input_ids,
            attention_mask=expanded_attention_mask,
            encoder_outputs=expanded_encoder_outputs,
            image_embeds=encoder_outputs[0] if self.config.consume_image_embeds else None,
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

        return InferenceResult(
            output=decoder_output,
            runtime=runtime,
        )

    @use_cuda
    def execute_accuracy(
        self,
        labels: List[List[str]],
        use_cuda = True,
    ) -> AccuracyResult:
        """
        Runs accuracy checks using labels from LAMBADA datasets.
        It uses the first `label_length - tokens_to_generate` tokens to predict the last
        `tokens_to_generate` tokens and calculates perplxity and TopN accuracy.

        Args:
            labels: List[List[str]] of shape (num_samples, batch_size)

        Returns:
            accuracy: AccuracyResult(topN, token_perplexity, seq_perplexity)

        """

        if self.config.consume_image_embeds:
            raise RuntimeError("Vision2Seq models currently do not support LAMBADA accuracy check!")

        results = []
        max_length = self.config.max_length
        tokens_to_generate = self.accuracy_metadata.tokens_to_generate
        num_beams = self.config.num_beams

        if num_beams > 1:
            G_LOGGER.warn("num_beams > 1. Skip perplexity calculation and default to -1")
        if self.config.network_name == "T5":
            G_LOGGER.warn(f"T5 may have very bad perplexity because perplexity tends to be very high when logits and tokens do not match. Please refer to TopN.")

        for label in labels:

            self.tokenizer.truncation_side = "right"
            tokenizer_output = self.tokenizer(
                label,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )

            target_ids = tokenizer_output.input_ids
            if self.config.is_encoder_decoder:
                # Need to insert decoder_start_token_id and truncate eos token
                target_ids = torch.cat(
                    (torch.ones((target_ids.shape[0], 1)).long().to(target_ids.device) * self.config.decoder_start_token_id, target_ids),
                    dim=1
                )

            target_ids = target_ids[:,:max_length]

            if use_cuda:
                target_ids = target_ids.cuda()

            G_LOGGER.info("-------------------------------")
            G_LOGGER.debug(f"target={self.tokenizer.batch_decode(target_ids,skip_special_tokens=True)};target_ids={target_ids}")

            num_tokens = target_ids.shape[1]

            # Always generate tokens_to_generate number of tokens
            context_len = target_ids.shape[1] - tokens_to_generate
            if self.config.is_encoder_decoder:
                # Because the last word may be eos token, we need to truncate that for encoder/decoder models
                context_len -= 1
            input_ids = target_ids[:,:context_len]

            if self.config.use_mask:
                attention_mask = tokenizer_output.attention_mask[:,:context_len]
                if use_cuda:
                    attention_mask = attention_mask.cuda()
            else:
                attention_mask = None

            if num_beams > 1:
                target_ids = target_ids.repeat_interleave(num_beams, dim=0)

            self.decoder.accuracy_mode(target_ids)

            _,_ = self.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # Force generating num_tokens for accuracy check
                min_length=num_tokens,
                max_length=num_tokens,
                use_cuda=use_cuda,
            )

            full_logits = self.decoder.full_logits.float()

            if num_beams == 1:
                shifted_logits = full_logits
                shifted_ids = target_ids[:,1:]
                if self.config.is_encoder_decoder:
                    # Truncate the last prediction
                    shifted_logits = shifted_logits[:,:-1,:]
                    shifted_ids = shifted_ids[:,:-1]
                seq_ppl = min(torch.nn.CrossEntropyLoss()(shifted_logits.permute((0, 2, 1)), shifted_ids), MAX_LOG_PPL)
            else:
                seq_ppl = -1

            if not self.config.is_encoder_decoder:
                target_ids = target_ids[:,-tokens_to_generate:]
                full_logits = full_logits[:, -tokens_to_generate:,:]
            else:
                # Truncate the last prediction
                target_ids = target_ids[:,-tokens_to_generate-1:-1]
                full_logits = full_logits[:, -tokens_to_generate-1:-1,:]

            assert target_ids.shape[1] == full_logits.shape[1], "Reference and full logits should have the same sequence length."

            # Calculate full sequence perplexity
            if num_beams == 1:
                token_ppl = min(torch.nn.CrossEntropyLoss()(full_logits.permute((0, 2, 1)), target_ids).item(), MAX_LOG_PPL)
            else:
                token_ppl = -1

            # Each batch and each token should have its own TopN Accuracy and perplexity
            topn_accuracy = []
            for token_id in range(tokens_to_generate):
                for bs in range(input_ids.shape[0]):
                    token = target_ids[bs:bs+1, token_id:token_id+1]
                    token_logits = full_logits[bs*num_beams:(bs+1)*num_beams,token_id:token_id+1, :]
                    G_LOGGER.info(f"Token:{token}, logits:{token_logits}, top10:{token_logits.topk(10).indices}")
                    for _topn in self.topN:
                        accuracy=1.0 if (token[0][0] in token_logits.topk(_topn, dim=-1).indices) else 0.0
                        topn_accuracy.append(TopNAccuracy(n=_topn, accuracy=accuracy))

            G_LOGGER.info(f"TopN accuracy={topn_accuracy};ppl(last {tokens_to_generate} tokens)={token_ppl}; ppl(sequence)={seq_ppl}")

            results.append(AccuracyResult(
                topN=topn_accuracy,
                token_perplexity=token_ppl,
                seq_perplexity=seq_ppl,
            ))

        return self.checkpoint.summary(results)

    @use_cuda
    def execute_benchmark(
        self,
        use_cuda: bool = True
    ) -> BenchmarkingResult:

        """
        Runs performance benchmark with random inputs and controled generation length.

        Args:
            None
        Returns:
            BenchmarkingResult: container runtime for each section and full inferecne

        """

        if self.models is None:
            self.models = self.setup_tokenizer_and_model()

        # opt_input_length = input_seq_len in benchmarking mode
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, self.config.opt_input_length))
        if self.config.consume_image_embeds:
            pixel_values = torch.rand(self.config.batch_size, 3, self.config.image_size, self.config.image_size)
            if self.config.fp16 and not isinstance(self.decoder, TRTNativeRunner):
                pixel_values = pixel_values.half()
        else:
            pixel_values = None

        attention_mask = None
        if self.config.use_mask:
            attention_mask = torch.ones_like(input_ids)

        inference_result = self.execute_inference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            use_cuda=use_cuda,
        )

        return BenchmarkingResult(median_runtime=inference_result.runtime, models=self.models)

    @use_cuda
    def execute_run(
        self,
        inputs: List[List[str]],
        labels: List[List[str]],
        use_cuda: bool = True
    ) -> NetworkCheckpointResult:

        """
        (Deprecated) Runs checkpoint.toml checks with absolute accuracy and perplexity calculation.

        Args:
            inputs: List[List[str]]: inputs from checkpoint. Shape: (num_samples, batch_size)
            labels: List[List[str]]: Golden outputs from checkpoint.toml. Shape: (num_samples, batch_size)

        Returns:
            NetworkCheckpointResult:
                network_results: input, output, accuracy, log_perplexity, runtime
                accuracy: average accuracy over all samples
                perplexity: average perplexity over all samples
        """

        if self.config.consume_image_embeds:
            raise RuntimeError("Vision2Seq models currently do not support checkpoint.toml accuracy check!")

        if self.models is None:
            self.models = self.setup_tokenizer_and_model()

        results = []
        for (input, label) in zip(inputs, labels):

            tokenizer_output = self.tokenizer(input, padding=True, return_tensors="pt")
            input_ids = tokenizer_output.input_ids
            attention_mask = None

            if self.config.use_mask:
                attention_mask = tokenizer_output.attention_mask

            decoder_output, runtime = self.execute_inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cuda=use_cuda,
            )

            max_length = self.config.max_length
            # Calculating perplexity
            if self.config.num_beams == 1:
                target_ids = self.tokenizer(
                    label,
                    padding=True,
                    return_tensors="pt",
                ).input_ids

                if self.config.is_encoder_decoder:
                    # Need to insert decoder start token and truncate eos token
                    target_ids = torch.cat(
                        (torch.ones((target_ids.shape[0], 1)).long().to(target_ids.device) * self.config.decoder_start_token_id, target_ids),
                        dim=1
                    )

                target_ids = target_ids[:,:max_length]
                if use_cuda:
                    target_ids = target_ids.cuda()

                self.decoder.accuracy_mode(target_ids)
                num_tokens = target_ids.shape[1]
                _,_ = self.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    # Force generating num_tokens for accuracy check
                    min_length=num_tokens,
                    max_length=num_tokens,
                    use_cuda=use_cuda,
                )
                full_logits = self.decoder.full_logits.float()
                shifted_logits = full_logits
                shifted_ids = target_ids[:,1:]
                if self.config.is_encoder_decoder:
                    # Truncate the last prediction for BART and T5 as it may be eos token.
                    shifted_logits = shifted_logits[:,:-1,:]
                    shifted_ids = shifted_ids[:,:-1]

                G_LOGGER.debug(f"target_ids:{shifted_ids};logits:{shifted_logits};top10:{shifted_logits.topk(10).indices}")
                ppl = min(torch.nn.CrossEntropyLoss()(shifted_logits.permute((0, 2, 1)), shifted_ids).item(), MAX_LOG_PPL)
                self.decoder.disable_accuracy_mode()

            else:
                G_LOGGER.warn("num_beams > 1. Skip perplexity calculation and default to -1.")
                ppl = -1

            semantic_outputs = self.tokenizer.batch_decode(
                decoder_output, skip_special_tokens=True
            )

            # Compare native accuracy for each batch
            def _process_text(text):
                return text.replace('\\n','').replace('\n','').replace('\\\\"','\"').replace('\\"','\"')

            accuracy = 0.0
            for (_result, _label) in zip(semantic_outputs, label):
                if _process_text(_result) == _process_text(_label):
                    accuracy += 1.0

            accuracy = accuracy / len(semantic_outputs)

            results.append(NetworkResult(
                input=input,
                output_tensor=decoder_output,
                semantic_output=semantic_outputs,
                median_runtime=runtime,
                perplexity=ppl,
                accuracy=accuracy,
            ))

        return self.checkpoint.summary(results)

    def run(self) -> Union[NetworkCheckpointResult, BenchmarkingResult, AccuracyResult]:
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

        try:
            if self.action == "benchmark":
                return self.execute_benchmark()
            elif self.action == "accuracy":
                labels = list(self.checkpoint.labels())
                return self.execute_accuracy(labels, use_cuda=True)
            elif self.action == "run" or self.action == "compare":
                inputs = list(self.checkpoint.inputs())
                labels = list(self.checkpoint.labels())
                return self.execute_run(inputs, labels, use_cuda=True)
            elif self.action == "chat":
                # Return self for chat mode
                return self
            else:
                raise RuntimeError(f"Invalid action {self.action} unimplemented for `run` function.")
        finally:
            if self.action != "chat":
                self.cleanup()

    def __call__(self):
        t0 = time.time()
        self.add_args()
        self._args = self._parser.parse_args()

        self.setup_environment(
            **vars(self._args),
        )
        t1 = time.time()
        G_LOGGER.info("Set up environment takes {:.4f}s.".format(t1 - t0))

        return self.run()

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
            "--use-cuda-graph",
            default=False,
            help="Use CUDA graph to inference. Only available if --use-cache is also added",
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
