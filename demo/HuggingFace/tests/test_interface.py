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
Tests and verifies our interface objects
"""

# std
import os
import sys

# pytest
import pytest

# Add library path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(TEST_DIR, os.pardir))


@pytest.fixture(scope="session")
def inetwork():
    import NNDF.networks as mod
    return mod


def test_network_result(inetwork):
    # Test the API by explicit flags
    inetwork.NetworkResult(
        input="example",
        output_tensor=[],
        semantic_output="hello",
        accuracy=0.0,
        perplexity=0.0,
        median_runtime=9001,
    )


def test_network_checkpoint_result(inetwork):
    inetwork.NetworkCheckpointResult(
        network_results=[],
        accuracy=9001.0,
        perplexity=5.0
    )


def test_precision(inetwork):
    inetwork.Precision(fp16=True)


def test_network_metadata(inetwork):
    inetwork.NetworkMetadata(
        variant="gpt2",
        precision=inetwork.Precision(fp16=True),
        use_cache=True,
        num_beams=1,
        batch_size=1,
    )


def test_topN_accuracy(inetwork):
    inetwork.TopNAccuracy(
        n=1,
        accuracy=1.0,
    )


def test_accuracy_metadata(inetwork):
    inetwork.AccuracyMetadata(
        dataset="LAMBADA",
        num_samples=100,
        tokens_to_generate=1,
    )


def test_accuract_result(inetwork):
    inetwork.AccuracyResult(
        topN=[
            inetwork.TopNAccuracy(
                n=1,
                accuracy=1.0,
            )
        ],
        token_perplexity=0.0,
        seq_perplexity=0.0,
    )