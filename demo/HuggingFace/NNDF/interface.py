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
Interface classes required for each registered network script.
"""

import argparse

from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

# NNDF
from NNDF.networks import (
    BenchmarkingResult,
    NetworkResult,
    NetworkMetadata,
    NetworkCheckpointResult,
    NNConfig,
    NetworkModel,
    TimingProfile,
)
from NNDF.logger import G_LOGGER

# externals
# None, there should be no external dependencies for testing purposes.

# Program-wide constants for passing in valid frameworks.
FRAMEWORK_NATIVE = "native"
FRAMEWORK_TENSORRT = "trt"
FRAMEWORK_ONNXRT = "onnxrt"
VALID_FRAMEWORKS = [
    FRAMEWORK_NATIVE,
    FRAMEWORK_ONNXRT,
    FRAMEWORK_TENSORRT
]

class MetadataArgparseInteropMixin:
    """Add argparse support where the class can add new arguments to an argparse object."""

    @staticmethod
    @abstractmethod
    def add_args(parser):
        pass

    @staticmethod
    @abstractmethod
    def from_args(args):
        pass

    @staticmethod
    @abstractmethod
    def add_inference_args(parser):
        pass

    @staticmethod
    @abstractmethod
    def from_inference_args(args):
        pass

    @staticmethod
    @abstractmethod
    def add_benchmarking_args(parser):
        """
        Add args needed for perf benchmarking mode.
        """
        pass

class NetworkCommand(metaclass=ABCMeta):
    """Base class that each network script's command module should inherit."""

    description = "NetworkCommand"

    DEFAULT_ITERATIONS = 10
    DEFAULT_NUMBER = 1
    DEFAULT_WARMUP = 3
    DEFAULT_DURATION = 0.0
    DEFAULT_PERCENTILE = 50

    def __init__(self, network_config: NNConfig, description: str):
        self.config = network_config()
        self.description = description
        self.framework_name = None
        self._parser = argparse.ArgumentParser(description=description, conflict_handler="resolve")

    def __call__(self):
        self.add_args(self._parser)
        self.config.MetadataClass.add_args(self._parser)
        self._args = self._parser.parse_args()

        if self._args.verbose:
            G_LOGGER.setLevel(level=G_LOGGER.DEBUG)
        elif self._args.info:
            G_LOGGER.setLevel(level=G_LOGGER.INFO)

        self.metadata = self.args_to_network_metadata(self._args)
        self.check_network_metadata_is_supported(self.metadata)

    @abstractmethod
    def run_benchmark(self):
        """
        Run inference in performance benchmarking mode for apples-to-apples perf comparisons across platforms.
        Differences with normal run mode include (but are not limited to):

        - Use random input data and disable accuracy checking.
        - Use fixed input/output sequence lengths and disable early stopping.
        - Provide better controls on the number of warm-ups and the number/duration of inference iterations.

        The derived class should override this method for the benchmarking implementation for the specific framework.
        """
        pass

    def add_args(self, parser) -> None:
        general_group = parser.add_argument_group("general")
        general_group.add_argument(
            "--verbose", help="Display verbose logs.", action="store_true"
        )
        general_group.add_argument(
            "--info", help="Display info logs.", action="store_true"
        )
        general_group.add_argument(
            "--cleanup",
            help="Cleans up user-specified workspace. Can not be cleaned if external files exist in workspace.",
            action="store_false",
        )
        general_group.add_argument(
            "--working-dir",
            help="Location of where to save the model and other downloaded files.",
            required=True,
        )
        general_group.add_argument(
            "--batch-size", "-b",
            help="Chosen batch size for given network",
            required=False,
            type=int,
            default=1
        )

        timing_group = parser.add_argument_group("inference measurement")
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

    def check_network_metadata_is_supported(self, metadata: NetworkMetadata) -> None:
        """
        Checks if current command supports the given metadata as defined by the NNConfig.
        Args:
            metadata (NetworkMetadata): NetworkMetadata to check if input is supported.

        Throws:
            NotImplementedError: If the given metadata is not a valid configuration for this network.

        Returns:
            None
        """
        if metadata not in self.config.variants:
            raise NotImplementedError(
                "The following network config is not yet supported by our scripts: {}".format(
                    metadata
                )
            )

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        return self.config.MetadataClass.from_args(args)

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
        )
        return checkpoint

    def get_timing_profile(self) -> TimingProfile:
        """
        Get TimingProfile settings given current args.
        """
        return TimingProfile(
                iterations=int(self._args.iterations),
                number=int(self._args.number),
                warmup=int(self._args.warmup),
                duration=int(self._args.duration),
                percentile=int(self._args.percentile),
            )


class FrameworkCommand(NetworkCommand):
    """Base class that is associated with Frameworks related scripts."""

    def __init__(self, network_config: NNConfig, description: str):
        super().__init__(network_config, description)
        self.framework_name = FRAMEWORK_NATIVE

    @abstractmethod
    def run_framework(
        self,
        metadata: NetworkMetadata,
        network_input: List[str],
        working_directory: str,
        keep_onnx_model: bool,
        keep_pytorch_model: bool,
        timing_profile: TimingProfile,
        batch_size: int,
        args: object = None,
        benchmarking_mode: bool = False,
        perplexity_reference: List[str] = None,
    ) -> Union[List[NetworkResult], BenchmarkingResult]:
        pass

    def __call__(self):
        super().__call__()

        checkpoint = self.load_nn_semantic_checkpoint()

        network_results, ppl_results = self.run_framework(
            metadata=self.metadata,
            network_input=list(checkpoint.inputs()),
            working_directory=self._args.working_dir,
            keep_onnx_model=self._args.cleanup,
            keep_pytorch_model=self._args.cleanup,
            timing_profile=self.get_timing_profile(),
            use_cpu=self._args.cpu,
            batch_size=self._args.batch_size,
            args=self._args,
            benchmarking_mode=False,
            perplexity_reference=list(checkpoint.labels()),
        )

        return NetworkCheckpointResult(
            network_results=network_results,
            accuracy=checkpoint.accuracy(network_results),
            perplexity=(sum(ppl_results) / len(ppl_results) if ppl_results else None),
        )

    def run_benchmark(self):
        self.config.MetadataClass.add_benchmarking_args(self._parser)
        super().__call__()

        network_results = self.run_framework(
            metadata=self.metadata,
            network_input=None,
            working_directory=self._args.working_dir,
            keep_onnx_model=self._args.cleanup,
            keep_pytorch_model=self._args.cleanup,
            timing_profile=self.get_timing_profile(),
            use_cpu=self._args.cpu,
            batch_size=self._args.batch_size,
            args=self._args,
            benchmarking_mode=True,
        )

        return network_results

    def add_args(self, parser) -> argparse.ArgumentParser:
        super().add_args(parser)
        device_group = parser.add_argument_group("device")
        device_group.add_argument(
            "--cpu",
            help="Run inference using CPU for frameworks.",
            action="store_true",
        )

class TRTInferenceCommand(NetworkCommand):
    """Base class that is associated with Polygraphy related scripts."""

    def __init__(
        self,
        network_config: NNConfig,
        description: str,
        frameworks_cmd: FrameworkCommand,
    ):
        super().__init__(network_config, description)
        self.framework_name = FRAMEWORK_TENSORRT
        # Should be set by
        self.frameworks_cmd = frameworks_cmd()

    @abstractmethod
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
        perplexity_reference: List[str] = None,
    ) -> Union[List[NetworkResult], BenchmarkingResult]:
        pass

    def __call__(self):
        self.config.MetadataClass.add_inference_args(self._parser)
        super().__call__()
        onnx_fpaths = self.args_to_network_models(self._args)

        checkpoint = self.load_nn_semantic_checkpoint()

        network_results, ppl_results = self.run_trt(
            metadata=self.metadata,
            onnx_fpaths=onnx_fpaths,
            network_input=list(checkpoint.inputs()),
            working_directory=self._args.working_dir,
            keep_trt_engine=self._args.cleanup,
            keep_onnx_model=self._args.cleanup,
            keep_torch_model=self._args.cleanup,
            timing_profile=self.get_timing_profile(),
            batch_size=self._args.batch_size,
            args=self._args,
            benchmarking_mode=False,
            preview_dynamic_shapes=self._args.preview_dynamic_shapes,
            perplexity_reference=list(checkpoint.labels()),
        )

        return NetworkCheckpointResult(
            network_results=network_results,
            accuracy=checkpoint.accuracy(network_results),
            perplexity=(sum(ppl_results) / len(ppl_results) if ppl_results else None),
        )

    def run_benchmark(self):
        self.config.MetadataClass.add_inference_args(self._parser)
        self.config.MetadataClass.add_benchmarking_args(self._parser)
        super().__call__()
        onnx_fpaths = self.args_to_network_models(self._args)

        network_results = self.run_trt(
            metadata=self.metadata,
            onnx_fpaths=onnx_fpaths,
            network_input=None,
            working_directory=self._args.working_dir,
            keep_trt_engine=self._args.cleanup,
            keep_onnx_model=self._args.cleanup,
            keep_torch_model=self._args.cleanup,
            timing_profile=self.get_timing_profile(),
            batch_size=self._args.batch_size,
            args=self._args,
            benchmarking_mode=True,
            preview_dynamic_shapes=self._args.preview_dynamic_shapes
        )

        return network_results

    def add_args(self, parser) -> argparse.ArgumentParser:
        super().add_args(parser)
        trt_group = parser.add_argument_group("trt")
        trt_group.add_argument(
            "--preview-dynamic-shapes",
            help="Use the FASTER_DYNAMIC_SHAPES_0805 preview feature when building the TensorRT engine",
            action="store_true",
        )

        trt_benchmarking_group = parser.add_argument_group("trt benchmarking group")
        trt_benchmarking_group.add_argument(
            "--input-profile-max-len",
            type=int,
            help="Specify max input sequence length in TRT engine profile. (default: max supported sequence length)",
        )
        trt_benchmarking_group.add_argument(
            "--output-profile-max-len",
            type=int,
            help="Specify max output sequence length in TRT engine profile. (default: max supported sequence length)",
        )

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        return self.config.MetadataClass.from_inference_args(args)

    @abstractmethod
    def args_to_network_models(self, args) -> Tuple[NetworkModel]:
        """
        Converts argparse arguments into a list of valid NetworkModel fpaths. Specifically for ONNX.
        Invokes conversion scripts if not.
        Return:
            List[NetworkModel]: List of network model names.
        """

class OnnxRTCommand(NetworkCommand):
    """ONNX Runtime command."""

    def __init__(
        self,
        network_config: NNConfig,
        description: str,
        frameworks_cmd: FrameworkCommand,
    ):
        super().__init__(network_config, description)
        self.framework_name = FRAMEWORK_ONNXRT
        # Should be set by
        self.frameworks_cmd = frameworks_cmd()

    @abstractmethod
    def run_onnxrt(
        self,
        metadata: NetworkMetadata,
        onnx_fpaths: Tuple[NetworkModel],
        network_input: List[str],
        working_directory: str,
        keep_onnx_model: bool,
        keep_torch_model: bool,
        timing_profile: TimingProfile,
        args: object = None,
        benchmarking_mode: bool = False,
    ) -> Union[List[NetworkResult], BenchmarkingResult]:
        pass

    def __call__(self):
        self.config.MetadataClass.add_inference_args(self._parser)
        super().__call__()
        onnx_fpaths = self.args_to_network_models(self._args)

        checkpoint = self.load_nn_semantic_checkpoint()

        network_results = self.run_onnxrt(
            metadata=self.metadata,
            onnx_fpaths=onnx_fpaths,
            network_input=list(checkpoint.inputs()),
            working_directory=self._args.working_dir,
            keep_onnx_model=self._args.cleanup,
            keep_torch_model=self._args.cleanup,
            timing_profile=self.get_timing_profile(),
            batch_size=self._args.batch_size,
            args=self._args,
            benchmarking_mode=False,
        )

        return NetworkCheckpointResult(
            network_results=network_results,
            accuracy=checkpoint.accuracy(network_results),
            perplexity=None,
        )

    def run_benchmark(self):
        self.config.MetadataClass.add_inference_args(self._parser)
        self.config.MetadataClass.add_benchmarking_args(self._parser)
        super().__call__()
        onnx_fpaths = self.args_to_network_models(self._args)

        network_results = self.run_onnxrt(
            metadata=self.metadata,
            onnx_fpaths=onnx_fpaths,
            network_input=None,
            working_directory=self._args.working_dir,
            keep_onnx_model=self._args.cleanup,
            keep_torch_model=self._args.cleanup,
            timing_profile=self.get_timing_profile(),
            batch_size=self._args.batch_size,
            args=self._args,
            benchmarking_mode=True,
        )

        return network_results

    def args_to_network_metadata(self, args) -> NetworkMetadata:
        return self.config.MetadataClass.from_inference_args(args)

    @abstractmethod
    def args_to_network_models(self, args) -> Tuple[NetworkModel]:
        """
        Converts argparse arguments into a list of valid NetworkModel fpaths. Specifically for ONNX.
        Invokes conversion scripts if not.
        Return:
            List[NetworkModel]: List of network model names.
        """
