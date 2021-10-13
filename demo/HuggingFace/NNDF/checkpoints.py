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
Helper file for generating common checkpoints.
"""

from typing import List

# TRT-HuggingFace
from NNDF.networks import NetworkMetadata, NetworkResult

# externals
import toml


class NNTomlCheckpoint:
    """Loads a toml checkpoint file for comparing labels and inputs."""

    def __init__(self, fpath: str, framework: str, network_name: str, metadata: NetworkMetadata):
        """Loads the toml file for processing."""
        data = {}
        with open(fpath) as f:
            data = toml.load(f)

        # Select the current input data
        # try to get the base data
        network = data.get(network_name, {})
        self.baseline = network.get("all", {}).get("default", {})
        specific_general_data = network.get("all", {}).get(metadata.variant, {})
        # Defaults are also used as baselines for the network in case there are deviations known in variants.

        # then apply specific data
        addendum = network.get(framework, {})
        addendum_default = addendum.get("default", {})
        addendum_specific = addendum.get(metadata.variant, {})
        self.data = {
            k: {**self.baseline[k],
                **specific_general_data.get(k, {}),
                **addendum_default.get(k, {}),
                **addendum_specific.get(k, {})} for k in self.baseline.keys()
        }

        # Used when accuracy() is called
        self._lookup_cache = None

    def _iterate_data(self, slice: List[str], skip_keyword: str = "skip"):
        """
        Helper for child classes to iterate through a slice of data.

        Return:
            (Union[Dict[str, str], List[str]]): Returns a list of all value keys given in 'slice' or if more than one value is given for 'slice' then a dictionary instead.
        """
        returns_dict = len(slice) > 1
        for value in self.data.values():
            if "skip" in value:
                continue

            if returns_dict:
                yield {s: value[s] for s in slice}
            else:
                yield value[slice[0]]


class NNSemanticCheckpoint(NNTomlCheckpoint):
    """Requires the following data structure:

    [<network>.<framework>.<variant>]
        [input_a]
        label = "sample_label"
        input = "sample_input"

        [input_b]
        label = "sample_label"
        input = "sample_input"

    Following are reserved keywords:
    <framework> = "all" indicates rules apply to all frameworks
    <variant> = "default" indicates rules apply to all networks.
    """

    def __iter__(self):
        return self._iterate_data(["label", "input"])

    def labels(self):
        return self._iterate_data(["label"])

    def inputs(self):
        return self._iterate_data(["input"])

    def accuracy(self, results: List[NetworkResult]) -> float:
        # Hash checkpoints by their input
        if self._lookup_cache is None:
            self._lookup_cache = {}
            for k, v in self.data.items():
                self._lookup_cache[v["input"]] = k

        correct_count = 0
        for r in results:
            # Find the data the corresponds to input
            key = self._lookup_cache[r.input]
            # remove new line characters
            r_new = r.semantic_output[0] if isinstance(r.semantic_output, list) else r.semantic_output
            correct_count += int(self.data[key]["label"].replace('\\n','').replace('\n','') == r_new.replace('\\n','').replace('\n',''))

        return correct_count / len(results)
