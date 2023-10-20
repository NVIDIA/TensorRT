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
Helper file for generating common checkpoints.
"""

import os
import itertools
from typing import List
import math

# TRT-HuggingFace
from NNDF.networks import (
    NetworkMetadata,
    NetworkResult,
    NetworkCheckpointResult,
    AccuracyResult,
    TopNAccuracy
)
from NNDF.interface import VALID_FRAMEWORKS

# externals
import toml
import torch
from datasets import load_dataset, load_from_disk

RANDOM_SEED = 42

def process_text(text):
    '''
    Process text such that all the quotes are the same
    '''
    # print(text)
    return text.replace("“", '"').replace("”", '"').\
        replace("’", "'").replace("‘", "'").\
        replace("`", "'").replace('\\n','\n').\
        replace('\\\\"','\"').replace('\\"','\"')

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

        self.metadata = metadata

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
            if skip_keyword in value:
                continue

            try:
                if returns_dict:
                    yield {s: value[s] for s in slice}
                else:
                    yield value[slice[0]]
            except KeyError as e:
                raise KeyError(f"Your checkpoint is missing fields for this model: {slice}") from e



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
    def __init__(self, *args, skip_multibatch = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_multibatch = skip_multibatch
        self._labels = None
        self._inputs = None

    def __iter__(self):
        return self._iterate_data(["label", "input"])

    def labels(self):
        if self._labels:
            return self._labels
        self._labels = [n for n in self._iterate_data(["label"])]
        bs = self.metadata.batch_size
        if self.skip_multibatch:
            self._labels = [n for n in self._labels if not isinstance(n, list)]
        # Use round-robin for multi-batch cases
        self._labels = [[n] * bs if not isinstance(n, list) else n * (bs // len(n)) + n[:(bs % len(n))] for n in self._labels]
        self._labels = [[process_text(n) for n in b] for b in self._labels]
        return self._labels

    def inputs(self):
        if self._inputs:
            return self._inputs
        self._inputs = [n for n in self._iterate_data(["input"])]
        if self.skip_multibatch:
            self._inputs = [n for n in self._inputs if not isinstance(n, list)]
        bs = self.metadata.batch_size
        self._inputs = [[n] * bs if not isinstance(n, list) else n * (bs // len(n)) + n[:(bs % len(n))] for n in self._inputs]
        self._inputs = [[process_text(n) for n in b] for b in self._inputs]
        return self._inputs

    def summary(self, results: List[NetworkResult]) -> NetworkCheckpointResult:
        assert len(results) > 0, "No valid results are summarized"
        accuracy = sum([n.accuracy for n in results]) / len(results)
        avg_log_ppl = sum([n.perplexity for n in results]) / len(results)
        # To avoid math.inf not supported by inf
        perplexity = -1 if avg_log_ppl == -1 else min(math.exp(avg_log_ppl), 100000.0)
        return NetworkCheckpointResult(
            network_results=results,
            accuracy=accuracy,
            perplexity=perplexity,
        )

class NNLambadaCheckpoint:
    """
    Loads an LAMBADA dataset from datasets and saves to disk. It only containes `labels` which contains texts batched automatically.
    """

    def __init__(
        self,
        base_dir,
        tokens_to_generate = 1,
        num_samples = 20,
        batch_size = 1,
        use_mask = False,
    ):
        assert tokens_to_generate >= 1
        self.base_dir = base_dir
        self.data_dir = os.path.join(self.base_dir, "lambada")
        self.tokens_to_generate = tokens_to_generate
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.use_mask = use_mask
        self.dataset = None
        self._labels = []
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.data_dir):
            try:
                self.dataset = load_from_disk(self.data_dir)
            except:
                self._download_from_remote()
        else:
            self._download_from_remote()

    def _download_from_remote(self):
        # Domain is not relevant for the dataset
        self.dataset = load_dataset("lambada", split="validation").remove_columns("domain")
        self.dataset.save_to_disk(self.data_dir)

    def _filter_dataset(self):
        index = torch.randint(0, self.dataset.num_rows, (self.num_samples,), generator=torch.manual_seed(RANDOM_SEED))
        return self.dataset.select(index.tolist()).map(lambda x: {"text":[process_text(y) for y in x['text']]}, batched=True)

    def _generate_test_data(self):
        filtered_dataset = self._filter_dataset()
        self._labels = []
        # Only batch inputs together if using attention_mask
        number_skip = self.batch_size if self.use_mask else 1
        for i in range(0, len(filtered_dataset) - number_skip + 1, number_skip):
            label = filtered_dataset['text'][i:i+number_skip]
            # If not use-mask but multi-batch, attempt to expand the same input
            if self.batch_size > 1 and number_skip == 1:
                label = label * self.batch_size

            self._labels.append(label)

    def labels(self):
        if not self._labels:
            self._generate_test_data()
        return self._labels

    def summary(self, results: List[AccuracyResult]) -> AccuracyResult:
        assert len(results) > 0, "No valid results are summarized"

        avg_seq_log_ppl = sum([n.seq_perplexity for n in results]) / len(results)
        # To avoid math.inf not supported by pickle.
        seq_perplexity = -1 if avg_seq_log_ppl == -1 else min(math.exp(avg_seq_log_ppl), 100000.0)
        avg_token_log_ppl = sum([n.token_perplexity for n in results]) / len(results)
        token_perplexity = -1 if avg_token_log_ppl == -1 else min(math.exp(avg_token_log_ppl), 100000.0)

        sum_topN_accuracy = {}
        for n in results:
            for topN in n.topN:
                if topN.n not in sum_topN_accuracy:
                    sum_topN_accuracy[topN.n] = 0.0
                sum_topN_accuracy[topN.n] += topN.accuracy

        topN_accuracy = []
        for n in sum_topN_accuracy:
            topN_accuracy += [TopNAccuracy(n=n, accuracy=sum_topN_accuracy[n]/len(results)/self.tokens_to_generate/self.batch_size)]

        return AccuracyResult(
            topN=topN_accuracy,
            token_perplexity=token_perplexity,
            seq_perplexity=seq_perplexity,
        )
