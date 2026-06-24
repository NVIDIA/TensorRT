#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from cuda.bindings.driver import CUstream

from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    TrtRunner,
    create_network,
    engine_from_network,
)

THREADS_PER_BLOCK = 256


def volume(d):
    return int(np.prod(d))


class UnownedMemory:
    def __init__(self, ptr, shape, dtype):
        mem = cp.cuda.UnownedMemory(ptr, volume(shape) * cp.dtype(dtype).itemsize, self)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        self.d = cp.ndarray(shape, dtype=dtype, memptr=memptr)


@cute.kernel
def rms_norm_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    threads_per_block: cutlass.Constexpr,
    hidden_dim: cutlass.Constexpr,
    epsilon: cutlass.Constexpr,
):
    # One block per token. Threads cooperatively sum-of-squares across the
    # hidden dim into shared memory, reduce to a single RMS value, then scale.
    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(
        cutlass.Float32,
        layout=cute.make_layout(threads_per_block),
        byte_alignment=16,
    )
    rms = smem.allocate_tensor(cutlass.Float32, layout=cute.make_layout(1))

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Per-thread partial sum-of-squares. Accumulate in fp32 regardless of
    # input dtype to avoid catastrophic cancellation on large hidden dims.
    local_sum = cutlass.Float32(0.0)
    for i in cutlass.range(tidx, hidden_dim, threads_per_block):
        x = cutlass.Float32(mX[bidx, i])
        local_sum += x * x
    sdata[tidx] = local_sum
    cute.arch.sync_threads()

    # Tree reduction in shared memory down to one warp, then warp shuffle.
    if tidx < 128:
        sdata[tidx] += sdata[tidx + 128]
    cute.arch.sync_threads()

    if tidx < 64:
        sdata[tidx] += sdata[tidx + 64]
    cute.arch.sync_threads()

    if tidx < 32:
        v = sdata[tidx] + sdata[tidx + 32]
        v = cute.arch.warp_reduction_sum(v, threads_in_group=32)
        if tidx == 0:
            rms[0] = cute.math.rsqrt(v / hidden_dim + epsilon, fastmath=True)
    cute.arch.sync_threads()

    scale = rms[0]
    for i in cutlass.range(tidx, hidden_dim, threads_per_block):
        y = cutlass.Float32(mX[bidx, i]) * cutlass.Float32(mW[i]) * scale
        mY[bidx, i] = y.to(mY.element_type)


@cute.jit
def rms_norm_launch(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mY: cute.Tensor,
    num_tokens: cutlass.Int32,
    hidden_dim: cutlass.Constexpr,
    epsilon: cutlass.Constexpr,
    stream: CUstream,
):
    # `num_tokens` is a runtime value (grid dim); only `hidden_dim` is baked
    # in at compile time. That way one compiled kernel handles every sequence
    # length the optimization profile allows.
    # The `stream` parameter is consumed by the CuteDSL runtime and used as
    # the launch stream; it does not need to be passed to .launch() directly.
    rms_norm_kernel(mX, mW, mY, THREADS_PER_BLOCK, hidden_dim, epsilon).launch(
        grid=(num_tokens, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
    )


class RmsNormPlugin(
    trt.IPluginV3,
    trt.IPluginV3OneCore,
    trt.IPluginV3OneBuild,
    trt.IPluginV3OneRuntime,
):
    def __init__(self, fc=None):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3OneCore.__init__(self)
        trt.IPluginV3OneBuild.__init__(self)
        trt.IPluginV3OneRuntime.__init__(self)

        self.plugin_namespace = ""
        self.plugin_name = "RmsNormPlugin"
        self.plugin_version = "1"
        self.num_outputs = 1
        self.timing_cache_id = ""

        # JIT-compiled kernel cache keyed by (hidden_dim, epsilon). Both are
        # baked into the kernel as `Constexpr`, so a change in either yields
        # a different binary.
        self._compiled = {}

        # `fc is None` is the path clone() takes; the cloned plugin then
        # picks up epsilon via __dict__.update(). For the build path TRT
        # always hands us a populated fc, and we require epsilon in it.
        if fc is not None:
            fields = {f.name: f for f in fc}
            assert "epsilon" in fields, "RmsNormPlugin requires an 'epsilon' field"
            self.epsilon = float(fields["epsilon"].data[0])

    def get_capability_interface(self, type):
        """Return the object that implements the requested capability.

        TRT calls this with `type` set to CORE, BUILD, or RUNTIME. Because
        this class inherits all three capability mixins, the same `self` can
        serve any of them.
        """
        return self

    def get_output_data_types(self, input_types):
        """Output dtype is FP16 (matches X)."""
        return [input_types[0]]

    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        """Y has the same shape as X (N, H)."""
        return [trt.DimsExprs(inputs[0])]

    def get_fields_to_serialize(self):
        """Tell TRT which plugin attributes to save into the engine.

        TRT calls this at build time and hands the same fields back to the
        creator's `create_plugin()` at engine load time. We only need to
        persist `epsilon`; everything else is either fixed or recomputed.
        """
        return trt.PluginFieldCollection(
            [
                trt.PluginField(
                    "epsilon",
                    np.array([self.epsilon], dtype=np.float32),
                    trt.PluginFieldType.FLOAT32,
                )
            ]
        )

    def configure_plugin(self, inp, out):
        """No-op. We precompute nothing from the I/O descriptors at build time."""
        pass

    def on_shape_change(self, inp, out):
        """No-op. The only dimension that can change between calls is
        `num_tokens` (`hidden_dim` is static). The kernel cache is keyed on
        `(hidden_dim, epsilon)`, so a change in `num_tokens` doesn't
        invalidate anything.
        """
        pass

    def supports_format_combination(self, pos, in_out, num_inputs):
        """Accept LINEAR layout for all three positions, with `X` and `Y` FP16
        and `weight` FP32.
        """
        assert num_inputs == 2
        assert pos < len(in_out)

        desc = in_out[pos].desc
        if desc.format != trt.TensorFormat.LINEAR:
            return False
        if pos == 0 or pos == 2:
            return desc.type == trt.DataType.HALF
        if pos == 1:
            return desc.type == trt.DataType.FLOAT
        raise AssertionError(f"unexpected pos={pos}")

    def enqueue(self, input_desc, output_desc, inputs, outputs, workspace, stream):
        """Launch the CuteDSL kernel on TRT's stream.

        Read shape and dtype from the descriptors, wrap the raw device
        pointers as CuteDSL tensors zero-copy via cupy + dlpack, JIT-compile
        once per `(hidden_dim, epsilon)` (cached), then launch.
        """
        x_dims = input_desc[0].dims
        w_dims = input_desc[1].dims
        assert len(x_dims) == 2, "X must be rank-2 (num_tokens, hidden_dim)"
        assert len(w_dims) == 1, "weight must be rank-1 (hidden_dim,)"
        num_tokens, hidden_dim = int(x_dims[0]), int(x_dims[1])
        assert int(w_dims[0]) == hidden_dim

        x_np = trt.nptype(input_desc[0].type)
        w_np = trt.nptype(input_desc[1].type)
        y_np = trt.nptype(output_desc[0].type)

        # TRT hands us raw int pointers into device memory it already owns.
        # CuteDSL wants `cute.Tensor` objects. We bridge the two without
        # copying via two protocol conversions:
        #   raw int ptr
        #     -> cupy.UnownedMemory: the only public Python API that wraps a
        #        foreign device pointer (we do NOT own it; cupy must not free
        #        it) into a typed, shaped ndarray.
        #     -> torch.as_tensor: reads cupy's __cuda_array_interface__ and
        #        gives us a torch view over the same bytes.
        #     -> cute.runtime.from_dlpack: reads torch's __dlpack__ capsule
        #        and produces the cute.Tensor the kernel actually sees.
        # Each hop is zero-copy; we're just re-typing the same GPU bytes.
        x_t = torch.as_tensor(UnownedMemory(inputs[0], (num_tokens, hidden_dim), x_np).d, device="cuda")
        w_t = torch.as_tensor(UnownedMemory(inputs[1], (hidden_dim,), w_np).d, device="cuda")
        y_t = torch.as_tensor(UnownedMemory(outputs[0], (num_tokens, hidden_dim), y_np).d, device="cuda")

        mX = from_dlpack(x_t, assumed_align=16)
        mW = from_dlpack(w_t, assumed_align=16)
        mY = from_dlpack(y_t, assumed_align=16)

        # Launch the kernel on the CUDA stream TRT gave us, not the default
        # stream. CuteDSL picks the stream from a `CUstream`-typed argument
        # of the @cute.jit launcher; at compile time we hand it a placeholder.
        key = (hidden_dim, self.epsilon)
        if key not in self._compiled:
            self._compiled[key] = cute.compile(
                rms_norm_launch, mX, mW, mY, num_tokens, hidden_dim, self.epsilon, make_fake_stream()
            )
        self._compiled[key](mX, mW, mY, num_tokens, CUstream(stream))

    def attach_to_context(self, context):
        """Give each execution context its own plugin copy so the JIT cache
        is not shared across concurrent inferences.
        """
        return self.clone()

    def set_tactic(self, tactic):
        """No-op. This plugin advertises a single tactic, so there is nothing
        to select.
        """
        pass

    def clone(self):
        """Return a copy of this plugin with an empty JIT cache.

        TRT clones the plugin per execution context. State is copied via
        `__dict__.update`; the compiled-kernel dict is intentionally reset
        so each context owns its own cubins.
        """
        cloned = RmsNormPlugin()
        cloned.__dict__.update(self.__dict__)
        cloned._compiled = {}
        return cloned

    def destroy(self):
        """Release the JIT cache when TRT is done with the plugin."""
        self._compiled.clear()


class RmsNormPluginCreator(trt.IPluginCreatorV3One):
    def __init__(self):
        trt.IPluginCreatorV3One.__init__(self)
        self.name = "RmsNormPlugin"
        self.plugin_namespace = ""
        self.plugin_version = "1"
        self.field_names = trt.PluginFieldCollection(
            [trt.PluginField("epsilon", np.array([], dtype=np.float32), trt.PluginFieldType.FLOAT32)]
        )

    def create_plugin(self, name, fc, phase):
        return RmsNormPlugin(fc)


def build_engine(hidden_dim, min_tokens, opt_tokens, max_tokens, epsilon):
    """Build an engine where num_tokens is dynamic in [min, max]."""
    # Fails fast on `hidden_dim < THREADS_PER_BLOCK` since the kernel's tree
    # reduction uses hard-coded offsets (128, 64, 32) over shared memory,
    # so anything smaller would read uninitialized slots.
    if hidden_dim < THREADS_PER_BLOCK:
        raise ValueError(f"RmsNormPlugin requires hidden_dim >= {THREADS_PER_BLOCK}, got {hidden_dim}")

    plg_registry = trt.get_plugin_registry()
    if plg_registry.get_creator("RmsNormPlugin", "1", "") is None:
        plg_registry.register_creator(RmsNormPluginCreator(), "")

    creator = plg_registry.get_creator("RmsNormPlugin", "1", "")
    pfc = trt.PluginFieldCollection(
        [trt.PluginField("epsilon", np.array([epsilon], dtype=np.float32), trt.PluginFieldType.FLOAT32)]
    )
    plugin = creator.create_plugin("RmsNormPlugin", pfc, trt.TensorRTPhase.BUILD)

    builder, network = create_network(strongly_typed=True)
    # X has dynamic num_tokens (axis 0 = -1) and static hidden_dim.
    X = network.add_input(name="X", dtype=trt.float16, shape=(-1, hidden_dim))
    W = network.add_input(name="weight", dtype=trt.float32, shape=(hidden_dim,))
    out = network.add_plugin_v3([X, W], [], plugin)
    out.get_output(0).name = "Y"
    network.mark_output(out.get_output(0))

    profile = Profile().add(
        "X",
        min=(min_tokens, hidden_dim),
        opt=(opt_tokens, hidden_dim),
        max=(max_tokens, hidden_dim),
    )
    return engine_from_network((builder, network), CreateConfig(profiles=[profile]))


def main():
    cc_major, cc_minor = torch.cuda.get_device_capability()
    assert cc_major >= 8, f"CuteDSL requires SM80+; this GPU is SM{cc_major}{cc_minor}."

    epsilon = 1e-5
    hidden_dim = 1024
    num_tokens = 128

    # Build with num_tokens dynamic in [1, 512]. The kernel is compiled once
    # for hidden_dim=1024 and serves every num_tokens within that range.
    engine = build_engine(hidden_dim, min_tokens=1, opt_tokens=128, max_tokens=512, epsilon=epsilon)

    # fp16 cast sets the tolerance floor.
    rtol, atol = 1e-2, 1e-2
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float16)
    w = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    y_ref = F.rms_norm(x.float(), (hidden_dim,), w, eps=epsilon).to(torch.float16)

    with TrtRunner(engine, "trt_runner") as runner:
        outputs = runner.infer({"X": x.cpu().numpy().astype(np.float16), "weight": w.cpu().numpy()})
        y = torch.as_tensor(outputs["Y"]).to(torch.float16)

    max_diff = (y - y_ref.cpu()).abs().max().item()
    label = f"N={num_tokens}, H={hidden_dim}"
    assert torch.allclose(
        y, y_ref.cpu(), rtol=rtol, atol=atol
    ), f"[{label}] Inference result incorrect! (max abs diff={max_diff:.3e})"
    print(f"[{label}] Inference result correct! (max abs diff={max_diff:.3e})")


if __name__ == "__main__":
    main()
