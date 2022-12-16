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

from textwrap import dedent

import pytest
import tensorrt as trt
from polygraphy import mod
from polygraphy.backend.trt import CreateConfig, Profile, create_network
from polygraphy.backend.trt import util as trt_util


@pytest.fixture(scope="session")
def dummy_network():
    builder, network = create_network()
    network.add_input("X", dtype=trt.float32, shape=[-1])
    with builder, network:
        yield builder, network


@pytest.fixture(scope="session")
def layer_class_mapping():
    return trt_util.get_layer_class_mapping()


@pytest.mark.parametrize("layer_type", trt.LayerType.__members__.values())
def test_all_layer_types_mapped(layer_class_mapping, layer_type):
    if layer_type == trt.LayerType.PLUGIN:
        pytest.skip("PLUGIN has no corresponding ILayer")
    assert layer_type in layer_class_mapping


# Can't use pytest.skip because we can't construct the test unless trt.MemoryPoolType exists.
if mod.version(trt.__version__) >= mod.version("8.4"):

    def add_default_preview_features_after_8_6(expected):
        if mod.version(trt.__version__) >= mod.version("8.6"):
            expected = expected.replace("MiB]", "MiB, TACTIC_DRAM: 24267.00 MiB]")

            if "Preview Features" not in expected:
                expected = (
                    dedent(expected).strip()
                    + "\nPreview Features       | [DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]"
                )

        return expected

    @pytest.mark.parametrize(
        "create_config, expected",
        # NOTE: We set workspace sizes here so we can have predictable output
        [
            (
                CreateConfig(memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 16 << 20}),
                add_default_preview_features_after_8_6(
                    """
                    Flags                  | []
                    Engine Capability      | EngineCapability.DEFAULT
                    Memory Pools           | [WORKSPACE: 16.00 MiB]
                    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                    Profiling Verbosity    | ProfilingVerbosity.DETAILED
                    """
                ),
            ),
            (
                CreateConfig(memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 16 << 20}, tactic_sources=[]),
                add_default_preview_features_after_8_6(
                    """
                    Flags                  | []
                    Engine Capability      | EngineCapability.DEFAULT
                    Memory Pools           | [WORKSPACE: 16.00 MiB]
                    Tactic Sources         | []
                    Profiling Verbosity    | ProfilingVerbosity.DETAILED
                    """
                ),
            ),
            (
                CreateConfig(memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 4 << 20}),
                add_default_preview_features_after_8_6(
                    """
                    Flags                  | []
                    Engine Capability      | EngineCapability.DEFAULT
                    Memory Pools           | [WORKSPACE: 4.00 MiB]
                    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                    Profiling Verbosity    | ProfilingVerbosity.DETAILED
                    """
                ),
            ),
            (
                CreateConfig(
                    memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 16 << 20},
                    fp16=True,
                    int8=True,
                    fp8=True,
                    tf32=True,
                    refittable=True,
                    precision_constraints="obey",
                ),
                add_default_preview_features_after_8_6(
                    """
                    Flags                  | [FP16, INT8, REFIT, TF32, OBEY_PRECISION_CONSTRAINTS, FP8]
                    Engine Capability      | EngineCapability.DEFAULT
                    Memory Pools           | [WORKSPACE: 16.00 MiB]
                    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                    Profiling Verbosity    | ProfilingVerbosity.DETAILED
                    """
                ),
            ),
            (
                CreateConfig(
                    memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 16 << 20},
                    profiles=[Profile().add("X", [1], [1], [1]), Profile().add("X", [2], [2], [2])],
                ),
                add_default_preview_features_after_8_6(
                    """
                    Flags                  | []
                    Engine Capability      | EngineCapability.DEFAULT
                    Memory Pools           | [WORKSPACE: 16.00 MiB]
                    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                    Profiling Verbosity    | ProfilingVerbosity.DETAILED
                    Optimization Profiles  | 2 profile(s)
                    """
                ),
            ),
            (
                CreateConfig(memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 16 << 20}, use_dla=True),
                add_default_preview_features_after_8_6(
                    """
                    Flags                  | []
                    Engine Capability      | EngineCapability.DEFAULT
                    Memory Pools           | [WORKSPACE: 16.00 MiB, DLA_MANAGED_SRAM: 0.00 MiB, DLA_LOCAL_DRAM: 1024.00 MiB, DLA_GLOBAL_DRAM: 512.00 MiB]
                    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                    DLA                    | Default Device Type: DeviceType.DLA, Core: -1
                    Profiling Verbosity    | ProfilingVerbosity.DETAILED
                    """
                ),
            ),
        ]
        + [
            (
                CreateConfig(
                    memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 16 << 20},
                    preview_features=[trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805],
                ),
                add_default_preview_features_after_8_6(
                    """
                    Flags                  | []
                    Engine Capability      | EngineCapability.DEFAULT
                    Memory Pools           | [WORKSPACE: 16.00 MiB]
                    Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                    Profiling Verbosity    | ProfilingVerbosity.DETAILED
                    Preview Features       | [FASTER_DYNAMIC_SHAPES_0805]
                    """
                ),
            ),
        ]
        if mod.version(trt.__version__) >= mod.version("8.5")
        else [],
        ids=["default", "tactic-sources", "memory-pool-limits", "builder-flags", "profiles", "dla"]
        + ["preview-features"]
        if mod.version(trt.__version__) >= mod.version("8.5")
        else [],
    )
    def test_str_from_config(create_config, expected, dummy_network):
        config = create_config(*dummy_network)
        assert trt_util.str_from_config(config) == dedent(expected).strip()


def test_get_all_tensors_layer_with_null_inputs():
    builder, network = create_network()
    with builder, network:
        inp = network.add_input("input", shape=(1, 3, 224, 224), dtype=trt.float32)
        slice_layer = network.add_slice(inp, (0, 0, 0, 0), (1, 3, 224, 224), (1, 1, 1, 1))

        # Set a tensor for `stride` to increment `num_inputs` so we have some inputs
        # which are `None` in between.
        slice_layer.set_input(3, inp)
        assert slice_layer.num_inputs == 4

        slice = slice_layer.get_output(0)
        slice.name = "Slice"
        network.mark_output(slice)

        assert trt_util.get_all_tensors(network) == {"input": inp, "Slice": slice}
