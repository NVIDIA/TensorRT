#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from ctypes import c_char_p, c_void_p, py_object, pythonapi

import atexit
import numpy as np
import pytest
import onnx
from polygraphy.tools.multi_device import ShardHints
from polygraphy import util
from tests.models.meta import ONNX_MODELS

from cuda.bindings import runtime as cudart

# ---------------------------------------------------------------------------
# Shared graph-structure helper (used by TestShard and TestCPShardingAccuracy)
# ---------------------------------------------------------------------------


def _check_dist_count(path, expected_scatter_count, expected_gather_count):
    model = onnx.load(path)
    dist_nodes = [n for n in model.graph.node if n.op_type == "DistCollective"]

    def _collective_op(node):
        for attr in node.attribute:
            if attr.name == "collective_operation":
                return attr.s.decode()
        return None

    gather_nodes = [n for n in dist_nodes if _collective_op(n) == "all_gather"]
    scatter_nodes = [n for n in dist_nodes if _collective_op(n) == "reduce_scatter"]
    assert len(scatter_nodes) == expected_scatter_count
    assert len(gather_nodes) == expected_gather_count


# ---------------------------------------------------------------------------
# Multi-GPU skip helpers
# ---------------------------------------------------------------------------


def _gpu_count():
    res, count = cudart.cudaGetDeviceCount()
    _assertCudaSuccess(res)
    return count


def _trt_supports_dist_collective():
    """Return True only if TRT >= 10.16 (DistCollective op available)."""
    try:
        import tensorrt as trt

        parts = [int(x) for x in trt.__version__.split(".")[:2]]
        return tuple(parts) >= (10, 16)
    except Exception:
        return False


def _multigpu_skip(min_gpus: int = 2):
    return pytest.mark.skipif(
        _gpu_count() < min_gpus or not _trt_supports_dist_collective(),
        reason=f"requires at least {min_gpus} CUDA GPUs and TRT >= 10.16 (DistCollective support)",
    )


# ---------------------------------------------------------------------------
# TRT inference helpers used by TestCPShardingAccuracy
# ---------------------------------------------------------------------------


def _isCudaSuccess(err):
    return err == cudart.cudaError_t.cudaSuccess


def _assertCudaSuccess(err):
    assert _isCudaSuccess(err)


def _accuracy_metrics(output_single, output_multi):
    assert (
        output_single.shape == output_multi.shape
    ), f"Shape mismatch: single={output_single.shape}, multi={output_multi.shape}"

    max_abs_diff = float(np.max(np.abs(output_single - output_multi)))
    mean_abs_diff = float(np.mean(np.abs(output_single - output_multi)))

    norm_a = float(np.linalg.norm(output_single))
    norm_b = float(np.linalg.norm(output_multi))
    cosine_sim = float(
        np.dot(output_single.ravel(), output_multi.ravel()) / (norm_a * norm_b + 1e-12)
    )
    rel_l2_err = float(np.linalg.norm(output_single - output_multi) / (norm_a + 1e-12))

    assert max_abs_diff < 1e-3, f"Max absolute diff {max_abs_diff:.2e} exceeds 1e-3"
    print(f"Max absolute diff: {max_abs_diff:.2e}")
    assert mean_abs_diff < 1e-4, f"Mean absolute diff {mean_abs_diff:.2e} exceeds 1e-4"
    print(f"Mean absolute diff: {mean_abs_diff:.2e}")
    assert cosine_sim > 0.9999, f"Cosine similarity {cosine_sim:.6f} below 0.9999"
    print(f"Cosine similarity {cosine_sim:.6f}")
    assert rel_l2_err < 1e-2, f"Relative L2 error {rel_l2_err:.2e} exceeds 1e-2"
    print(f"Relative L2 error: {rel_l2_err:.2e}")


def _trt_run(model_path, input_data, rank=0, root=0, mpi_comm=None, nccl_comm=None):
    """Run *model_path* on GPU-0 with TRT and return the output as a numpy array."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        flags=1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Set multi-device flag if comm provided
    if nccl_comm:
        config.set_preview_feature(trt.PreviewFeature.MULTIDEVICE_RUNTIME_10_16, True)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(model_path):
        raise RuntimeError("TRT ONNX parse failed")

    # Build engine on root only
    if rank == root:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build failed")
        engine_bytes = bytes(serialized)
    else:
        engine_bytes = None

    if mpi_comm:
        engine_bytes = mpi_comm.bcast(engine_bytes, root=root)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # If multi-device, convert the NCCL comm to C pointers so it can be passed
    # to execution context
    if nccl_comm:
        PyCapsule_New = pythonapi.PyCapsule_New
        PyCapsule_New.restype = py_object
        PyCapsule_New.argtypes = [c_void_p, c_char_p, c_void_p]
        capsule = PyCapsule_New(c_void_p(nccl_comm.ptr), b"ncclComm_t", None)
        assert capsule, "Failed to create NCCL capsule"
        context.set_communicator(capsule)

    d_input = None
    d_output = None
    stream = None

    try:
        # Setup Input address
        input_name = "hidden_states"
        input_size = trt.volume(input_data.shape) * 4
        res, d_input = cudart.cudaMalloc(input_size)
        _assertCudaSuccess(res)
        (res,) = cudart.cudaMemcpy(
            d_input,
            input_data,
            input_size,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        _assertCudaSuccess(res)
        context.set_tensor_address(input_name, d_input)

        # Setup output address
        output_tensor = network.get_output(0)
        output_shape = tuple(output_tensor.shape)
        output_name = output_tensor.name  # Either 'output' or 'shard_n'
        output_size = trt.volume(output_shape) * 4
        res, d_output = cudart.cudaMalloc(output_size)
        _assertCudaSuccess(res)
        context.set_tensor_address(output_name, d_output)

        # Run inference
        res, stream = cudart.cudaStreamCreate()
        _assertCudaSuccess(res)
        context.execute_async_v3(stream)
        cudart.cudaStreamSynchronize(stream)

        # Get output array
        output = np.zeros(output_shape, dtype=np.float32)
        (res,) = cudart.cudaMemcpy(
            output, d_output, output_size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
        _assertCudaSuccess(res)

    finally:
        if stream is not None:
            cudart.cudaStreamDestroy(stream)
        if d_input is not None:
            cudart.cudaFree(d_input)
        if d_output is not None:
            cudart.cudaFree(d_output)

    return output


class TestShard:

    def check_node_type_in_graph(self, model, op, count=0):
        node_idx_op = [(i, node.op_type) for i, node in enumerate(model.graph.node)]
        node_idxs = [i for i, op_type in node_idx_op if op_type == op]

        assert len(node_idxs) >= count

        return node_idxs

    def test_multi_attention_head_shard(
        self, poly_multi_device_shard, poly_template_shard
    ):
        # Test for a network with multiple attention heads, shard only affects specified ones
        with util.NamedTemporaryFile(
            suffix=".onnx"
        ) as outmodel, util.NamedTemporaryFile(mode="w+", suffix=".json") as hints:
            poly_template_shard([ONNX_MODELS["multi_attention"].path, "-o", hints.name])

            # Remove the second attention
            json = ShardHints.load(hints.name)
            json.inputs = json.inputs[: len(json.inputs) // 2]
            json.outputs = json.outputs[: len(json.outputs) // 2]
            json.attention_layers = json.attention_layers[
                : len(json.attention_layers) // 2
            ]
            json.save(hints.name)

            poly_multi_device_shard(
                [
                    ONNX_MODELS["multi_attention"].path,
                    "-o",
                    outmodel.name,
                    "-s",
                    hints.name,
                ]
            )

            # Should only gather specified kv
            _check_dist_count(outmodel.name, 3, 3)

    @pytest.mark.parametrize("use_one_shot", [False, True])
    def test_shard_fused(
        self, poly_multi_device_shard, poly_template_shard, use_one_shot
    ):
        # Test shard can replace an entire attention head with Fused attention
        def check_fused_attention(path):
            model = onnx.load(path)

            _check_dist_count(path, 3, 1)
            attention_idx = self.check_node_type_in_graph(model, "Attention", 1)[0]

            attention_node = model.graph.node[attention_idx]
            to_check = ["scale", "TRT_decomposable", "nb_rank", "is_causal"]
            readable_attrs = [attr.name for attr in attention_node.attribute]

            for attr in to_check:
                assert attr in readable_attrs

        with util.NamedTemporaryFile(
            suffix=".onnx"
        ) as outmodel, util.NamedTemporaryFile(mode="w+", suffix=".json") as hints:
            if use_one_shot:
                poly_multi_device_shard(
                    [
                        ONNX_MODELS["attention"].path,
                        "-o",
                        outmodel.name,
                        "--one-shot",
                        "--cp-type",
                        "fused",
                    ]
                )
            else:
                poly_template_shard(
                    [
                        ONNX_MODELS["attention"].path,
                        "--cp-type",
                        "fused",
                        "-o",
                        hints.name,
                    ]
                )
                poly_multi_device_shard(
                    [
                        ONNX_MODELS["attention"].path,
                        "-o",
                        outmodel.name,
                        "-s",
                        hints.name,
                    ]
                )

            _check_dist_count(outmodel.name, 3, 1)
            check_fused_attention(outmodel.name)

    @pytest.mark.parametrize("use_one_shot", [False, True])
    def test_shard_same_qkv(
        self, poly_multi_device_shard, poly_template_shard, use_one_shot
    ):
        # Test shard doesn't insert multiple all_gathers if any inputs are the same
        with util.NamedTemporaryFile(
            suffix=".onnx"
        ) as outmodel, util.NamedTemporaryFile(mode="w+", suffix=".json") as hints:
            if use_one_shot:
                poly_multi_device_shard(
                    [
                        ONNX_MODELS["attention_same_qkv"].path,
                        "-o",
                        outmodel.name,
                        "--one-shot",
                    ]
                )
            else:
                poly_template_shard(
                    [
                        ONNX_MODELS["attention_same_qkv"].path,
                        "-o",
                        hints.name,
                    ]
                )
                poly_multi_device_shard(
                    [
                        ONNX_MODELS["attention_same_qkv"].path,
                        "-o",
                        outmodel.name,
                        "-s",
                        hints.name,
                    ]
                )

            # One scatter for single input, one gather for qkv (skip gather at end because q was gathered)
            _check_dist_count(outmodel.name, 1, 1)

    @pytest.mark.parametrize("use_one_shot", [False, True])
    def test_shard(self, poly_multi_device_shard, poly_template_shard, use_one_shot):
        # Test normal sharding of kv for attention
        with util.NamedTemporaryFile(
            suffix=".onnx"
        ) as outmodel, util.NamedTemporaryFile(mode="w+", suffix=".json") as hints:
            if use_one_shot:
                poly_multi_device_shard(
                    [
                        ONNX_MODELS["attention"].path,
                        "-o",
                        outmodel.name,
                        "--one-shot",
                    ]
                )
            else:
                poly_template_shard(
                    [
                        ONNX_MODELS["attention"].path,
                        "-o",
                        hints.name,
                    ]
                )
                poly_multi_device_shard(
                    [
                        ONNX_MODELS["attention"].path,
                        "-o",
                        outmodel.name,
                        "-s",
                        hints.name,
                    ]
                )

            # 3 scatters for each input, 2 gathers for kv, one gather at end
            _check_dist_count(outmodel.name, 3, 3)


# ---------------------------------------------------------------------------
# End-to-end CP sharding accuracy test (requires >= 2 GPUs)
# ---------------------------------------------------------------------------


@pytest.mark.multigpu
@_multigpu_skip(min_gpus=2)
class TestCPShardingAccuracy:
    """
    Validates that CP sharding preserves numerical correctness:
    the sharded (multi-device) model produces the same output as the
    original single-device model when run on real hardware.

    These tests are skipped automatically when fewer than 2 CUDA GPUs are
    present. They require TRT >= 10.16 with DistCollective (NCCL) support.
    """

    def test_sd_attention_cp_accuracy(
        self, poly_multi_device_shard, poly_template_shard
    ):
        """
        Shard a synthetic SD 1.5 self-attention ONNX across 2 ranks (CP mode,
        native / non-fused, seq_len_idx=1) and assert that rank-0's output
        numerically matches the single-device TRT baseline.

        Model dimensions: batch=1, seq_len=256, d_model=320, n_heads=8.
        Input : hidden_states [1, 256, 320]   (seq_len at dim-1)
        Output: [1, 256, 320]

        Expected DistCollective nodes after sharding:
          - 1 reduce_scatter  (hidden_states input, i_idx=1)
          - 3 all_gathers     (k_T with k_idx=2, v with v_idx=1, output with o_idx=1)
        """
        from mpi4py import MPI
        import nccl.core as nccl

        atexit.register(MPI.Finalize)

        # Setup MPI (root = 0)
        ROOT = 0
        mpi_comm = MPI.COMM_WORLD
        n_ranks = mpi_comm.Get_size()
        assert n_ranks >= 2, "Test Requires >= 2 ranks"
        rank = mpi_comm.Get_rank()
        cudart.cudaSetDevice(rank % _gpu_count())

        # Setup NCCL
        unique_id = nccl.get_unique_id() if rank == ROOT else None
        unique_id = mpi_comm.bcast(unique_id, root=ROOT)
        nccl_comm = nccl.Communicator.init(
            nranks=n_ranks, rank=rank, unique_id=unique_id
        )

        np.random.seed(0)

        if rank == ROOT:
            input_data = np.random.randn(1, 256, 320).astype(np.float32) * 0.1
        else:
            input_data = np.zeros((1, 256, 320)).astype(np.float32)

        with (
            util.NamedTemporaryFile(suffix=".onnx") as sharded_model,
            util.NamedTemporaryFile(mode="w+", suffix=".json") as hints,
        ):
            # --- Step 1: generate sharding hints ---
            poly_template_shard(
                [
                    ONNX_MODELS["sd_attention"].path,
                    "--i-idx",
                    "1",
                    "--o-idx",
                    "1",
                    "--k-idx",
                    "2",
                    "--v-idx",
                    "1",
                    "--nb-rank",
                    "2",
                    "--reduce-op",
                    "sum",
                    "-o",
                    hints.name,
                ]
            )

            # --- Step 2: produce sharded model ---
            poly_multi_device_shard(
                [
                    ONNX_MODELS["sd_attention"].path,
                    "-o",
                    sharded_model.name,
                    "-s",
                    hints.name,
                ]
            )

            # --- Step 3: sanity-check graph structure ---
            # 1 scatter for hidden_states (i_idx=1)
            # 3 gathers: k_T (k_idx=2), v (v_idx=1), output (o_idx=1)
            _check_dist_count(
                sharded_model.name, expected_scatter_count=1, expected_gather_count=3
            )

            # --- Step 4: single-device TRT baseline ---
            if rank == ROOT:
                output_single = _trt_run(ONNX_MODELS["sd_attention"].path, input_data)

            # --- Step 5: multi-device TRT with NCCL ---
            mpi_comm.Barrier()
            output_multi = _trt_run(
                sharded_model.name, input_data, rank, ROOT, mpi_comm, nccl_comm
            )
            mpi_comm.Barrier()
            nccl_comm.destroy()

            # --- Step 6: accuracy metrics ---

            # Verify MD is symmetric
            for i in range(1, n_ranks):
                if rank == ROOT:
                    other_multi = mpi_comm.recv(source=i)
                    print(f"Rank {rank} compared to Rank {i}...")
                    _accuracy_metrics(output_multi, other_multi)
                else:
                    mpi_comm.send(output_multi, dest=ROOT)

            # Verify SD == MD
            if rank == ROOT:
                print(f"Rank {rank} compared to SD...")
                _accuracy_metrics(output_single, output_multi)
