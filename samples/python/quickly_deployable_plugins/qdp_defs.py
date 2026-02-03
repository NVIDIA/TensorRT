#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tensorrt as trt
import torch
import numpy as np

from typing import Tuple, List, Union

import tensorrt.plugin as trtp
import numpy.typing as npt

import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("QuicklyDeployablePlugins").setLevel(logging.INFO)

########## Elemwise-add plugin definition ##########


@trtp.register("sample::elemwise_add_plugin")
def add_plugin_desc(inp0: trtp.TensorDesc, block_size: int) -> trtp.TensorDesc:
    return inp0.like()


# Helper to simulate defining/omitting an autotune definition for the plugin
def register_autotune():
    # Type annotations can be omitted for autotune and impl definitions, but will be checked for consistency if added
    @trtp.autotune("sample::elemwise_add_plugin")
    def add_plugin_autotune(
        inp0: trtp.TensorDesc, outputs: Tuple[trtp.TensorDesc]
    ) -> List[trtp.AutoTuneCombination]:
        return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16")]


@trtp.impl("sample::elemwise_add_plugin")
def add_plugin_impl(
    inp0: trtp.Tensor, block_size: int, outputs: Tuple[trtp.Tensor], stream: int
) -> None:

    log = logging.getLogger("QuicklyDeployablePlugins")
    log.debug(
        f"Executing for inp0: dtype={inp0.dtype},format={inp0.format} and output[0]: dtype={outputs[0].dtype},format={outputs[0].format}"
    )

    n = inp0.numel()

    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        inp0_t = torch.as_tensor(inp0, device="cuda")
        out_t = torch.as_tensor(outputs[0], device="cuda")

        import triton
        from oait_kernels import add_kernel

        add_kernel[(triton.cdiv(n, block_size),)](inp0_t, out_t, n, BLOCK_SIZE=block_size)


########## In-place elemwise-add plugin definition ##########


@trtp.register("sample::elemwise_add_plugin_")
def add_plugin_desc_(inp0: trtp.TensorDesc, delta: int) -> trtp.TensorDesc:
    return inp0.aliased()


@trtp.autotune("sample::elemwise_add_plugin_")
def add_plugin_autotune_(inp0, outputs) -> List[trtp.AutoTuneCombination]:
    return [
        trtp.AutoTuneCombination("FP32, FP32", "LINEAR*HWC"),
        trtp.AutoTuneCombination("FP32|FP16, FP32|FP16", "LINEAR"),
    ]


@trtp.impl("sample::elemwise_add_plugin_")
def add_plugin_impl_(inp0, delta: int, outputs, stream) -> None:

    log = logging.getLogger("QuicklyDeployablePlugins")
    log.debug(
        f"Executing for inp0: dtype={inp0.dtype},format={inp0.format} and output[0]: dtype={outputs[0].dtype},format={outputs[0].format}"
    )

    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        inp0_t = torch.as_tensor(inp0, device="cuda")
        inp0_t.add_(delta)


########## Non-zero plugin (DDS) ##########


@trtp.register("sample::non_zero_plugin")
def non_zero_plugin_reg(
    inp0: trtp.TensorDesc,
) -> Tuple[trtp.TensorDesc, trtp.TensorDesc]:
    upper_bound = inp0.shape_expr[0] * inp0.shape_expr[1]
    st = trtp.size_tensor(upper_bound // 2, upper_bound)
    st.dtype = trt.int64
    return trtp.from_shape_expr((st.expr(), 2), dtype=trt.int32), st


@trtp.autotune("sample::non_zero_plugin")
def non_zero_plugin_autotune(inp0, outputs) -> List[trtp.AutoTuneCombination]:
    return [trtp.AutoTuneCombination("FP32|FP16, INT32, INT64")]


@trtp.impl("sample::non_zero_plugin")
def non_zero_plugin_impl(inp0, outputs, stream) -> None:

    log = logging.getLogger("QuicklyDeployablePlugins")
    log.debug(
        f"Executing for inp0: dtype={inp0.dtype},format={inp0.format} and output[0]: dtype={outputs[0].dtype},format={outputs[0].format}"
    )

    with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
        inp0_t = torch.as_tensor(inp0, device="cuda")
        out_1 = torch.as_tensor(outputs[1], device="cuda").reshape((-1,))

        out = torch.nonzero(inp0_t)

        out0 = torch.as_tensor(outputs[0].aliased(out.shape), device="cuda")
        out0.copy_(out)
        out_1.copy_(torch.Tensor([out.shape[0]]))


########## Circular padding plugin ########


@trtp.register("sample::circ_pad_plugin")
def circ_pad_plugin_desc(
    inp0: trtp.TensorDesc, pads: npt.NDArray[np.int32]
) -> trtp.TensorDesc:
    ndim = inp0.ndim
    out_desc = inp0.like()

    for i in range(np.size(pads) // 2):
        out_desc.shape_expr[ndim - i - 1] += int(pads[i * 2] + pads[i * 2 + 1])

    return out_desc


# Helper to define a multi-tactic implementation of the plugin
def enable_multi_tactic_circ_pad():

    from enum import IntEnum

    class Tactic(IntEnum):
        TORCH = 1
        TRITON = 2

    @trtp.autotune("sample::circ_pad_plugin")
    def circ_pad_plugin_autotune(
        inp0: trtp.TensorDesc,
        outputs: Tuple[trtp.TensorDesc],
    ) -> List[trtp.AutoTuneCombination]:
        c = trtp.AutoTuneCombination()
        c.pos([0, 1], "FP32|FP16")
        c.tactics([int(Tactic.TORCH), int(Tactic.TRITON)])
        return [c]

    @trtp.impl("sample::circ_pad_plugin")
    def circ_pad_plugin_impl(
        inp0: trtp.Tensor,
        pads: npt.NDArray[np.int32],
        outputs: Tuple[trtp.Tensor],
        stream: int,
        tactic: int,
    ) -> None:

        log = logging.getLogger("QuicklyDeployablePlugins")
        log.debug(
            f"Executing for inp0: dtype={inp0.dtype},format={inp0.format} and output[0]: dtype={outputs[0].dtype},format={outputs[0].format}"
        )

        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            inp_t = torch.as_tensor(inp0, device="cuda")
            out_t = torch.as_tensor(outputs[0], device="cuda")

            if tactic == Tactic.TORCH:
                out = torch.nn.functional.pad(inp_t, pads.tolist(), mode="circular")
                out_t.copy_(out)
            elif tactic == Tactic.TRITON:
                N = inp0.ndim
                all_pads = np.zeros((N * 2,), dtype=np.int32)
                out_dims = trtp.Shape(tuple(inp0.shape))

                for i in range(np.size(pads) // 2):
                    out_dims[N - i - 1] += pads[i * 2] + pads[i * 2 + 1]
                    all_pads[N * 2 - 2 * i - 2] = pads[i * 2]
                    all_pads[N * 2 - 2 * i - 1] = pads[i * 2 + 1]

                all_pads = all_pads.tolist()

                block_size = 256
                num_blocks = tuple(
                    [int((np.prod(out_dims) + block_size - 1) // block_size)]
                )

                from oait_kernels import circ_pad

                circ_pad[num_blocks](
                    inp_t,
                    all_pads[0],
                    all_pads[2],
                    all_pads[4],
                    all_pads[6],
                    inp0.shape[0],
                    inp0.shape[1],
                    inp0.shape[2],
                    inp0.shape[3],
                    int(out_dims[1]),
                    int(out_dims[2]),
                    int(out_dims[3]),
                    inp0.numel(),
                    out_dims.numel(),
                    out_t,
                    BLOCK_SIZE=block_size,
                )


# Helper to define a single tactic implementation of the plugin
def enable_single_tactic_circ_pad():
    @trtp.autotune("sample::circ_pad_plugin")
    def circ_pad_plugin_autotune(
        inp0: trtp.TensorDesc,
        outputs: Tuple[trtp.TensorDesc],
    ) -> List[trtp.AutoTuneCombination]:

        return [trtp.AutoTuneCombination("FP32|FP16, FP32|FP16")]

    @trtp.impl("sample::circ_pad_plugin")
    def circ_pad_plugin_impl(
        inp0: trtp.Tensor,
        pads: npt.NDArray[np.int32],
        outputs: Tuple[trtp.Tensor],
        stream: int,
    ) -> None:
        with torch.cuda.stream(torch.cuda.ExternalStream(stream)):
            inp_t = torch.as_tensor(inp0, device="cuda")
            out_t = torch.as_tensor(outputs[0], device="cuda")

            out = torch.nn.functional.pad(inp_t, pads.tolist(), mode="circular")
            out_t.copy_(out)

    @trtp.aot_impl("sample::circ_pad_plugin")
    def circ_pad_plugin_aot_impl(
        inp0: trtp.TensorDesc, pads: npt.NDArray[np.int32], outputs: Tuple[trtp.TensorDesc], tactic: int
    ) -> Tuple[Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs]:

        block_size = 256

        N = inp0.ndim
        all_pads = np.zeros((N * 2,), dtype=np.int32)
        inp_dims = inp0.shape_expr
        out_dims = outputs[0].shape_expr

        for i in range(np.size(pads) // 2):
            all_pads[N * 2 - 2 * i - 2] = pads[i * 2]
            all_pads[N * 2 - 2 * i - 1] = pads[i * 2 + 1]

        all_pads = all_pads.tolist()

        # Representing all int32-scalar-kernel-inputs as symbolic expressions.
        # These inputs are either constants or derivatives of input/output shapes (that may be dynamic).
        # The symbolic expressions are resolved after the full shape context becomes available at runtime.
        extra_args = trtp.SymIntExprs.from_tuple(
            [
                trtp.SymInt32(e)
                for e in [
                    all_pads[0],
                    all_pads[2],
                    all_pads[4],
                    all_pads[6],
                    inp_dims[0],
                    inp_dims[1],
                    inp_dims[2],
                    inp_dims[3],
                    out_dims[1],
                    out_dims[2],
                    out_dims[3],
                    inp_dims.numel(),
                    out_dims.numel(),
                ]
            ]
        )


        type_str = "fp32" if inp0.dtype == trt.float32 else "fp16"

        from oait_kernels import circ_pad_kernel
        import triton

        src = triton.compiler.ASTSource(
            fn=circ_pad_kernel,
            signature=f"*{type_str},{','.join(['i32']*13)},*{type_str}",
            constants={
                "BLOCK_SIZE": block_size,
            },
        )

        compiled_kernel = triton.compile(src)
        launch_params = trtp.KernelLaunchParams()

        # grid dims
        launch_params.grid_x = trtp.cdiv(out_dims.numel(), block_size)
        # block dims
        launch_params.block_x = compiled_kernel.metadata.num_warps * 32
        # shared memory
        launch_params.shared_mem = compiled_kernel.metadata.shared

        return compiled_kernel.metadata.name.encode(), compiled_kernel.asm["ptx"].encode(), launch_params, extra_args
