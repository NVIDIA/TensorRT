#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
from textwrap import dedent

import pytest

from polygraphy import config, mod
from polygraphy.backend.trt import Profile, create_network
from polygraphy.backend.trt import util as trt_util

# Import CreateConfigRTX conditionally for TensorRT-RTX builds
if config.USE_TENSORRT_RTX:
    import tensorrt_rtx as trt
    from polygraphy.backend.tensorrt_rtx import CreateConfigRTX as CreateConfig
else:
    import tensorrt as trt
    from polygraphy.backend.trt import CreateConfig


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
    waived_layers = [trt.LayerType.PLUGIN]
    with contextlib.suppress(AttributeError):
        waived_layers.append(trt.LayerType.PLUGIN_V3)
    if layer_type in waived_layers:
        pytest.skip("PLUGIN has no corresponding ILayer")
    assert layer_type in layer_class_mapping


# Can't use pytest.skip because we can't construct the test unless trt.MemoryPoolType exists.


def adjust_memory_pool_limits_after_8_6(limits):
    # Adjust tactic DRAM so we can match the output text reliably in update_expected_output.
    if mod.version(trt.__version__) >= mod.version("8.6") or config.USE_TENSORRT_RTX:
        limits[trt.MemoryPoolType.TACTIC_DRAM] = 1 << 30
    return limits


def update_expected_output(expected):
    is_trt_10_plus = (
        mod.version(trt.__version__) >= mod.version("10.0") or 
        config.USE_TENSORRT_RTX
    )
    is_trt_8_6_plus = (
        mod.version(trt.__version__) >= mod.version("8.6") or 
        config.USE_TENSORRT_RTX
    )
    is_trt_8_7_plus = (
        mod.version(trt.__version__) >= mod.version("8.7") or 
        config.USE_TENSORRT_RTX
    )
    
    if is_trt_8_6_plus:
        if is_trt_10_plus:
            expected = expected.replace(
                "MiB]",
                "MiB, TACTIC_DRAM: 1024.00 MiB, TACTIC_SHARED_MEMORY: 1024.00 MiB]",
            )
        else:
            expected = expected.replace("MiB]", "MiB, TACTIC_DRAM: 1024.00 MiB]")

        if "Preview Features" not in expected:
            if not is_trt_10_plus:
                expected = (
                    dedent(expected).strip()
                    + "\nPreview Features       | [FASTER_DYNAMIC_SHAPES_0805, DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]"
                )
            else:
                preview_features = "[PROFILE_SHARING_0806"
                if config.USE_TENSORRT_RTX:
                    preview_features += ", RUNTIME_ACTIVATION_RESIZE_10_10"
                preview_features += "]"
                expected = (
                    dedent(expected).strip()
                    + f"\nPreview Features       | {preview_features}"
                )

    if is_trt_8_7_plus:
        # CUBLAS_LT is not longer enabled by default
        expected = expected.replace("CUBLAS_LT, ", "")

    if is_trt_10_plus:
        expected = expected.replace(
            "EngineCapability.DEFAULT", "EngineCapability.STANDARD"
        )
        expected = expected.replace("CUBLAS, ", "")
        expected = expected.replace("CUDNN, ", "")

    return expected


@pytest.mark.parametrize(
    "create_config, expected",
    # NOTE: We set workspace sizes here so we can have predictable output
    [
        (
            CreateConfig(
                memory_pool_limits=adjust_memory_pool_limits_after_8_6(
                    {trt.MemoryPoolType.WORKSPACE: 16 << 20}
                )
            ),
            update_expected_output(
                """
                Flags                  | [{}]
                Engine Capability      | EngineCapability.DEFAULT
                Memory Pools           | [WORKSPACE: 16.00 MiB]
                Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                Profiling Verbosity    | ProfilingVerbosity.DETAILED
                """.format("TF32" if config.USE_TENSORRT_RTX else "")
            ),
        ),
        (
            CreateConfig(
                memory_pool_limits=adjust_memory_pool_limits_after_8_6(
                    {trt.MemoryPoolType.WORKSPACE: 16 << 20}
                ),
                tactic_sources=[],
            ),
            update_expected_output(
                """
                Flags                  | [{}]
                Engine Capability      | EngineCapability.DEFAULT
                Memory Pools           | [WORKSPACE: 16.00 MiB]
                Tactic Sources         | []
                Profiling Verbosity    | ProfilingVerbosity.DETAILED
                """.format("TF32" if config.USE_TENSORRT_RTX else "")
            ),
        ),
        (
            CreateConfig(
                memory_pool_limits=adjust_memory_pool_limits_after_8_6(
                    {trt.MemoryPoolType.WORKSPACE: 4 << 20}
                )
            ),
            update_expected_output(
                """
                Flags                  | [{}]
                Engine Capability      | EngineCapability.DEFAULT
                Memory Pools           | [WORKSPACE: 4.00 MiB]
                Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                Profiling Verbosity    | ProfilingVerbosity.DETAILED
                """.format("TF32" if config.USE_TENSORRT_RTX else "")
            ),
        ),
        (
            CreateConfig(
                memory_pool_limits=adjust_memory_pool_limits_after_8_6(
                    {trt.MemoryPoolType.WORKSPACE: 16 << 20}
                ),
                **({} if config.USE_TENSORRT_RTX else {
                    "fp16": True,
                    "int8": True,
                    "tf32": True,
                }),
                refittable=True,
                precision_constraints="obey",
            ),
            update_expected_output(
                """
                Flags                  | [{}REFIT, TF32, OBEY_PRECISION_CONSTRAINTS]
                Engine Capability      | EngineCapability.DEFAULT
                Memory Pools           | [WORKSPACE: 16.00 MiB]
                Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                Profiling Verbosity    | ProfilingVerbosity.DETAILED
                """.format(
                    "" if config.USE_TENSORRT_RTX else "FP16, INT8, ",
                )
            ),
        ),
        (
            CreateConfig(
                memory_pool_limits=adjust_memory_pool_limits_after_8_6(
                    {trt.MemoryPoolType.WORKSPACE: 16 << 20}
                ),
                profiles=[
                    Profile().add("X", [1], [1], [1]),
                    Profile().add("X", [2], [2], [2]),
                ],
            ),
            update_expected_output(
                """
                Flags                  | [{}]
                Engine Capability      | EngineCapability.DEFAULT
                Memory Pools           | [WORKSPACE: 16.00 MiB]
                Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                Profiling Verbosity    | ProfilingVerbosity.DETAILED
                Optimization Profiles  | 2 profile(s)
                """.format("TF32" if config.USE_TENSORRT_RTX else "")
            ),
        ),
    ] + ([] if config.USE_TENSORRT_RTX else [
        (
            CreateConfig(
                memory_pool_limits=adjust_memory_pool_limits_after_8_6(
                    {trt.MemoryPoolType.WORKSPACE: 16 << 20}
                ),
                use_dla=True,
            ),
            update_expected_output(
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
    ]) + [
        (
            (
                CreateConfig(
                    memory_pool_limits=adjust_memory_pool_limits_after_8_6(
                        {trt.MemoryPoolType.WORKSPACE: 16 << 20}
                    ),
                    preview_features=[trt.PreviewFeature.PROFILE_SHARING_0806],
                ),
                update_expected_output(
                    """
                Flags                  | [{}]
                Engine Capability      | EngineCapability.DEFAULT
                Memory Pools           | [WORKSPACE: 16.00 MiB]
                Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                Profiling Verbosity    | ProfilingVerbosity.DETAILED
                Preview Features       | [PROFILE_SHARING_0806]
                """.format("TF32" if config.USE_TENSORRT_RTX else "")
                ),
            )
            if mod.version(trt.__version__) >= mod.version("10.0") or config.USE_TENSORRT_RTX
            else (
                CreateConfig(
                    memory_pool_limits=adjust_memory_pool_limits_after_8_6(
                        {trt.MemoryPoolType.WORKSPACE: 16 << 20}
                    ),
                    preview_features=(
                        [trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03] 
                        if config.USE_TENSORRT_RTX 
                        else [trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805]
                    ),
                ),
                update_expected_output(
                    """
                Flags                  | [{}]
                Engine Capability      | EngineCapability.DEFAULT
                Memory Pools           | [WORKSPACE: 16.00 MiB]
                Tactic Sources         | [CUBLAS, CUBLAS_LT, CUDNN, EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
                Profiling Verbosity    | ProfilingVerbosity.DETAILED
                Preview Features       | [{}]
                """.format(
                    "TF32" if config.USE_TENSORRT_RTX else "",
                    "ALIASED_PLUGIN_IO_10_03" if config.USE_TENSORRT_RTX else "FASTER_DYNAMIC_SHAPES_0805"
                )
                ),
            )
        ),
    ],
    ids=[
        "default",
        "tactic-sources",
        "memory-pool-limits",
        "builder-flags" + ("-rtx" if config.USE_TENSORRT_RTX else ""),
        "profiles",
    ] + ([] if config.USE_TENSORRT_RTX else ["dla"]) + [
        "preview-features",
    ],
)
def test_str_from_config(create_config, expected, dummy_network):
    config = create_config(*dummy_network)
    actual = trt_util.str_from_config(config, dummy_network)
    expected = dedent(expected).strip()
    assert actual == expected


def test_get_all_tensors_layer_with_null_inputs():
    builder, network = create_network()
    with builder, network:
        inp = network.add_input("input", shape=(1, 3, 224, 224), dtype=trt.float32)
        slice_layer = network.add_slice(
            inp, (0, 0, 0, 0), (1, 3, 224, 224), (1, 1, 1, 1)
        )

        # Set a tensor for `stride` to increment `num_inputs` so we have some inputs
        # which are `None` in between.
        slice_layer.set_input(3, inp)
        assert slice_layer.num_inputs == 4

        slice = slice_layer.get_output(0)
        slice.name = "Slice"
        network.mark_output(slice)

        assert trt_util.get_all_tensors(network) == {"input": inp, "Slice": slice}
