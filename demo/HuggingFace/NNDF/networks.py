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
Helpers for abstracting high-level network concepts. Different from 'models.py' which deals
with IO abstraction.
"""

import string

from typing import Dict, Union, Tuple
from collections import namedtuple, OrderedDict

# externals
# None. Should not have any external dependencies.

FILENAME_VALID_CHARS = "-~_.() {}{}".format(string.ascii_letters, string.digits)

"""NetworkResult(input: str, output_tensor: np.array, semantic_output: np.array, median_runtime: NetworkRuntime)"""
NetworkResult = namedtuple(
    "NetworkResult",
    ["input", "output_tensor", "semantic_output", "accuracy", "perplexity", "median_runtime"],
)

"""BenchmarkingResult(median_runtime: NetworkRuntime, models: [str])"""
BenchmarkingResult = namedtuple(
    "BenchmarkingResult",
    ["median_runtime", "models"],
)

"""CheckpointResult(network_results: List[NetworkResult], accuracy: float, perplexity: float)"""
NetworkCheckpointResult = namedtuple(
    "NetworkCheckpointResult", ["network_results", "accuracy", "perplexity"]
)

"""TopNAccuracy(n: int, accuracy: float)"""
TopNAccuracy = namedtuple(
    "TopNAccuracy", ["n", "accuracy"]
)

"""AccuracyMetadata(dataset: str, num_samples: int, tokens_to_generate: int)"""
AccuracyMetadata = namedtuple(
    "AccuracyMetadata", ["dataset", "num_samples", "tokens_to_generate"]
)

"""AccuracyResult(topN: List[TopNAccuracy], token_perplexity: float, seq_perplexity: float)"""
AccuracyResult = namedtuple(
    "AccuracyResult", ["topN", "token_perplexity", "seq_perplexity"]
)

InputResult = namedtuple(
    "InputResult", ["output"]
)

# Tracks TRT Precision Config
"""Precision(fp16: Bool)"""
Precision = namedtuple("Precision", ["fp16"])

"""NetworkMetadata(variant: str, precision: Precision, use_cache: bool, num_beams: int, batch_size: int)"""
NetworkMetadata = namedtuple("NetworkMetadata", ["variant", "precision", "use_cache", "num_beams", "batch_size"])

"""TimingProfile(iterations: int, number: int, warmup: int, duration: int, percentile: int or [int])"""
TimingProfile = namedtuple("TimingProfile", ["iterations", "number", "warmup", "duration", "percentile"])


"""NetworkModel(name: str, fpath: str)"""
NetworkModel = namedtuple("NetworkModel", ["name", "fpath"])

"""
String encodings to genereted network models.
    NetworkModels(torch: Tuple[NetworkModel], onnx: Tuple[NetworkModel])
"""
NetworkModels = namedtuple("NetworkModels", ["torch", "onnx", "trt"])

"""
Args:
    name: Name of the network / parts of the network timed.
    runtime: Runtime of the time.

NetworkRuntime(name: str, runtime: float)
"""
NetworkRuntime = namedtuple("NetworkRuntime", ["name", "runtime"])

"""
Args:
    output: Output tokens for an inference session
    runtime: Runtime for an inference session
"""

InferenceResult = namedtuple("InferenceResult", ["output", "runtime"])

class Dims:
    """Helper class for interfacing dimension constructs with Polygraphy and PyTorch."""

    BATCH = "batch"
    SEQUENCE = "sequence"

    def __init__(self, encoding: OrderedDict):
        self.encoding = encoding

    def create_new_sequence_dim(dim_type: str) -> str:
        """
        Returns a new sequence dimension.

        Return:
            str: Returns a sequence dimension which Dims.SEQUENCE appended by dim_type.
        """
        return Dims.SEQUENCE + "_" + dim_type

    def get_dims(self):
        """
        Returns the encoding dimensions.

        Return:
            OrderedDict[str, Union[int, str]]: Returns dimensional encoding. Example: {'input_ids': (1, SEQUENCE_DIM)}
        """
        return self.encoding

    def get_names(self) -> Tuple[str]:
        return tuple(self.encoding.keys())

    def get_lengths(self) -> Tuple[Union[int, str]]:
        return tuple(self.encoding.values())

    def get_torch_dynamic_axis_encoding(self) -> dict:
        """
        Returns a Pytorch "dynamic_axes" encoding for onnx.export.

        Returns:
            dict: Returns a 'dynamic' index with corresponding names according to:
                https://pytorch.org/docs/stable/onnx.html
        """

        dynamic_axes = {}
        for k, v in self.encoding.items():
            encodings = []
            for idx, e in enumerate(v):
                if isinstance(e, str) and (self.BATCH in e or self.SEQUENCE in e):
                    encodings.append((idx, e))
            dynamic_axes[k] = {idx: e for idx, e in encodings}

        return dynamic_axes

# Config Class
class NNConfig:
    """Contains info for a given network that we support."""

    NETWORK_SEGMENTS = ["full"]

    def __init__(self, network_name, variants=None):
        assert self._is_valid_filename(
            network_name
        ), "Network name: {} is not filename friendly.".format(network_name)

        self.network_name = network_name
        self.variants = variants

    def from_hf_config(self, hf_config):
        """
        Set up config class from a HuggingFace config.
        """
        pass

    def get_network_segments(self):
        """
        Returns exportable segments for the given network.
        Used in the case where a single network needs to
        be exported into multiple parts.
        """
        return self.NETWORK_SEGMENTS

    def get_output_dims(self) -> Dict:
        """
        Returns the output dimensions of the current network.
        Since some networks can have multiple parts, should be a dictionary encoding.

        Returns:
            (Dict): {"network_section": Dims}
        """
        raise NotImplementedError("Output dims not yet defined.")

    def get_input_dims(self) -> Dict:
        """
        Returns the input dimensions of the current network.
        Since some networks can have multiple parts, should be a dictionary encoding.

        Returns:
            (Dict): {"network_section": Dims} example:
                {"encoder": Dims(...), "decoder": Dims(...)}
        """
        raise NotImplementedError("Input dims not yet defined.")

    def _is_valid_filename(self, filename: str) -> bool:
        """
        Checks if a given filename is valid, helpful for cross platform dependencies.
        """
        return all(c in FILENAME_VALID_CHARS for c in filename)

    def get_python_requirements():
        return []

    def get_metadata_string(self, metadata: NetworkMetadata) -> str:
        """
        Serializes a Metadata object into string.
        String will be checked if friendly to filenames across Windows and Linux operating systems.

        returns:
            string: <network>-<variant-name>-<precision>-<others>
        """

        precision_str = "-".join(
            [k for k, v in metadata.precision._asdict().items() if v]
        )

        result = [self.network_name, metadata.variant]
        if precision_str:
            result.append(precision_str)

        if metadata.use_cache:
            result.append("kv_cache")

        final_str = "-".join(result)
        assert self._is_valid_filename(
            final_str
        ), "Metadata for current network {} is not filename friendly: {}.".format(
            self.network_name, final_str
        )

        return final_str
