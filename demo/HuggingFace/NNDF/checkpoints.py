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
Helper file for generating common checkpoints.
"""

import itertools
from typing import List

# TRT-HuggingFace
from NNDF.networks import NetworkMetadata, NetworkResult
from NNDF.interface import VALID_FRAMEWORKS

# externals
import toml
class NNTomlCheckpoint:
    """
    Loads a toml checkpoint file for comparing labels and inputs.
    The following nested key structure is required:

    [Network.Framework.Variant.Precision]

    For each category, you can assign a default behviour using a special key
    defined by CHECKPOINT_STRUCTURE_FLAT.

    CHECKPOINT_STRUCTURE_FLAT cannot be valid in terms of the result that is being added inwards.
    """

    # The checkpoint structure and their default keys
    CHECKPOINT_STRUCTURE_FLAT = {
        "framework": "all",
        "variant": "default",
        "precision": "all"
    }

    def __init__(self, fpath: str, framework: str, network_name: str, metadata: NetworkMetadata):
        """Loads the toml file for processing."""
        data = {}
        with open(fpath) as f:
            data = toml.load(f)

        assert framework in VALID_FRAMEWORKS
        # These keys are reserved to indicate the default state.
        assert self.CHECKPOINT_STRUCTURE_FLAT["framework"] not in VALID_FRAMEWORKS

        # Select the current input data
        # try to get the base data
        network_data = data.get(network_name, {})

        cur_keys = {
            "framework": framework,
            "variant": metadata.variant,
            "precision": "fp16" if metadata.precision.fp16 else "fp32"
        }

        combined_keys =[[self.CHECKPOINT_STRUCTURE_FLAT[k], cur_keys[k]] for k in self.CHECKPOINT_STRUCTURE_FLAT.keys()]
        # A helper function for flattening the getters.
        def flat_getter(d=network_data, *args):
            for k in args:
                if k not in d:
                    return {}
                d = d[k]
            return d

        # self.data stores several keys:
        # {"checkpoint_name": {"label": xxx, "input": xxx}}
        # The loop below attempts to merge several code snippets together.
        self.data = network_data["all"]["default"]["all"]
        for keys in itertools.product(*combined_keys):
            values = flat_getter(network_data, *keys)
            if len(values) == 0:
                continue
            for data_k, data_v in self.data.items():
                if data_k in values:
                    self.data[data_k] = {**data_v, **values[data_k]}

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

    [<network>.<framework>.<variant>.<precision>]
        [input_a]
        label = "sample_label"
        input = "sample_input"

        [input_b]
        label = "sample_label"
        input = "sample_input"

    Following are reserved keywords:
    <framework> = "all" indicates rules apply to all frameworks
    <variant> = "default" indicates rules apply to all networks.
    <precision> = "all" indicates rules apply to all precisions.
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
