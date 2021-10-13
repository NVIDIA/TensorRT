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

"""
Interface classes required for each registered network script.
"""

import argparse

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

# NNDF
from NNDF.networks import (
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

class NetworkCommand(metaclass=ABCMeta):
    """Base class that each network script's command module should inherit."""

    description = "NetworkCommand"

    DEFAULT_ITERATIONS = 10
    DEFAULT_NUMBER = 1
    DEFAULT_WARMUP = 3

    def __init__(self, network_config: NNConfig, description: str):
        self.config = network_config()
        self.description = description
        self._parser = argparse.ArgumentParser(description=description, conflict_handler="resolve")

    def __call__(self):
        self.add_args(self._parser)
        self.config.MetadataClass.add_args(self._parser)
        self._args = self._parser.parse_args()

        if self._args.verbose:
            G_LOGGER.setLevel(level=G_LOGGER.DEBUG)

        self.metadata = self.args_to_network_metadata(self._args)
        self.check_network_metadata_is_supported(self.metadata)

    def add_args(self, parser) -> None:
        general_group = parser.add_argument_group("general")
        general_group.add_argument(
            "--verbose", help="Display verbose logs.", action="store_true"
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

        timing_group = parser.add_argument_group("inference measurement")
        timing_group.add_argument(
            "--iterations", help="Number of iterations to measure.", default=self.DEFAULT_ITERATIONS
        )
        timing_group.add_argument(
            "--number",
            help="Number of actual inference cycles per iterations.",
            default=self.DEFAULT_NUMBER,
        )
        timing_group.add_argument(
            "--warmup",
            help="Number of warmup iterations before actual measurement occurs.",
            default=self.DEFAULT_WARMUP,
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


class FrameworkCommand(NetworkCommand):
    """Base class that is associated with Frameworks related scripts."""

    @abstractmethod
    def run_framework(
        self,
        metadata: NetworkMetadata,
        network_input: List[str],
        working_directory: str,
        keep_onnx_model: bool,
        keep_pytorch_model: bool,
        timing_profile: TimingProfile,
    ) -> List[NetworkResult]:
        pass

    def __call__(self):
        super().__call__()

        # Differ import so that interface file can use used without
        # dependency install for our testing.
        from NNDF.checkpoints import NNSemanticCheckpoint
        checkpoint = NNSemanticCheckpoint(
            "checkpoint.toml",
            framework="native",
            network_name=self.config.network_name,
            metadata=self.metadata,
        )
        network_results = self.run_framework(
            metadata=self.metadata,
            network_input=list(checkpoint.inputs()),
            working_directory=self._args.working_dir,
            keep_onnx_model=self._args.cleanup,
            keep_pytorch_model=self._args.cleanup,
            timing_profile=TimingProfile(
                iterations=int(self._args.iterations),
                number=int(self._args.number),
                warmup=int(self._args.warmup),
            ),
        )

        return NetworkCheckpointResult(
            network_results=network_results,
            accuracy=checkpoint.accuracy(network_results),
        )

    def add_args(self, parser) -> argparse.ArgumentParser:
        super().add_args(parser)


class TRTInferenceCommand(NetworkCommand):
    """Base class that is associated with Polygraphy related scripts."""

    def __init__(
        self,
        network_config: NNConfig,
        description: str,
        frameworks_cmd: FrameworkCommand,
    ):
        super().__init__(network_config, description)
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
    ) -> List[NetworkResult]:
        pass

    def __call__(self):
        self.config.MetadataClass.add_inference_args(self._parser)
        super().__call__()
        onnx_fpaths = self.args_to_network_models(self._args)

        # Differ import so that interface file can use used without
        # dependency install for our testing.
        from NNDF.checkpoints import NNSemanticCheckpoint
        checkpoint = NNSemanticCheckpoint(
            "checkpoint.toml",
            framework="native",
            network_name=self.config.network_name,
            metadata=self.metadata,
        )
        network_results = self.run_trt(
            metadata=self.metadata,
            onnx_fpaths=onnx_fpaths,
            network_input=list(checkpoint.inputs()),
            working_directory=self._args.working_dir,
            keep_trt_engine=self._args.cleanup,
            keep_onnx_model=self._args.cleanup,
            keep_torch_model=self._args.cleanup,
            timing_profile=TimingProfile(
                iterations=int(self._args.iterations),
                number=int(self._args.number),
                warmup=int(self._args.warmup),
            ),
        )

        return NetworkCheckpointResult(
            network_results=network_results,
            accuracy=checkpoint.accuracy(network_results),
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
    ) -> List[NetworkResult]:
        pass

    def __call__(self):
        self.config.MetadataClass.add_inference_args(self._parser)
        super().__call__()
        onnx_fpaths = self.args_to_network_models(self._args)

        # Differ import so that interface file can use used without
        # dependency install for our testing.
        from NNDF.checkpoints import NNSemanticCheckpoint
        checkpoint = NNSemanticCheckpoint(
            "checkpoint.toml",
            framework="native",
            network_name=self.config.network_name,
            metadata=self.metadata,
        )
        network_results = self.run_onnxrt(
            metadata=self.metadata,
            onnx_fpaths=onnx_fpaths,
            network_input=list(checkpoint.inputs()),
            working_directory=self._args.working_dir,
            keep_onnx_model=self._args.cleanup,
            keep_torch_model=self._args.cleanup,
            timing_profile=TimingProfile(
                iterations=int(self._args.iterations),
                number=int(self._args.number),
                warmup=int(self._args.warmup),
            ),
        )

        return NetworkCheckpointResult(
            network_results=network_results,
            accuracy=checkpoint.accuracy(network_results),
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
