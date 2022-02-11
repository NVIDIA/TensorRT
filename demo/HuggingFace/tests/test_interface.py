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
        median_runtime=9001,
        models=[],
    )


def test_network_checkpoint_result(inetwork):
    inetwork.NetworkCheckpointResult(network_results=[], accuracy=9001.0)


def test_precision(inetwork):
    inetwork.Precision(fp16=True)


def test_network_metadata(inetwork):
    inetwork.NetworkMetadata(
        variant="gpt2", precision=inetwork.Precision(fp16=True), other=None
    )
