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

import re
import torch
import numpy as np
import onnx
import onnx_graphsurgeon as gs

from modelopt.torch.quantization import utils as quant_utils
from modelopt.torch.quantization.calib.max import MaxCalibrator
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear

USE_PEFT = True
try:
    from peft.tuners.lora.layer import Conv2d as PEFTLoRAConv2d
    from peft.tuners.lora.layer import Linear as PEFTLoRALinear
except ModuleNotFoundError:
    USE_PEFT = False

class PercentileCalibrator(MaxCalibrator):
    def __init__(self, num_bits=8, axis=None, unsigned=False, track_amax=False, **kwargs):
        super().__init__(num_bits, axis, unsigned, track_amax)
        self.percentile = kwargs["percentile"]
        self.total_step = kwargs["total_step"]
        self.collect_method = kwargs["collect_method"]
        self.data = {}
        self.i = 0

    def collect(self, x):
        """Tracks the absolute max of all tensors.

        Args:
            x: A tensor

        Raises:
            RuntimeError: If amax shape changes
        """
        # Swap axis to reduce.
        axis = self._axis if isinstance(self._axis, (list, tuple)) else [self._axis]
        # Handle negative axis.
        axis = [x.dim() + i if isinstance(i, int) and i < 0 else i for i in axis]
        reduce_axis = []
        for i in range(x.dim()):
            if i not in axis:
                reduce_axis.append(i)
        local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()
        _cur_step = self.i % self.total_step
        if _cur_step not in self.data.keys():
            self.data[_cur_step] = local_amax
        else:
            if self.collect_method == "global_min":
                self.data[_cur_step] = torch.min(self.data[_cur_step], local_amax)
            elif self.collect_method == "min-max" or self.collect_method == "mean-max":
                self.data[_cur_step] = torch.max(self.data[_cur_step], local_amax)
            else:
                self.data[_cur_step] += local_amax
        if self._track_amax:
            raise NotImplementedError
        self.i += 1

    def compute_amax(self):
        """Return the absolute max of all tensors collected."""
        up_lim = int(self.total_step * self.percentile)
        if self.collect_method == "min-mean":
            amaxs_values = [self.data[i] / self.total_step for i in range(0, up_lim)]
        else:
            amaxs_values = [self.data[i] for i in range(0, up_lim)]
        if self.collect_method == "mean-max":
            act_amax = torch.vstack(amaxs_values).mean(axis=0)[0]
        else:
            act_amax = torch.vstack(amaxs_values).min(axis=0)[0]
        self._calib_amax = act_amax
        return self._calib_amax

    def __str__(self):
        s = "PercentileCalibrator"
        return s.format(**self.__dict__)

    def __repr__(self):
        s = "PercentileCalibrator("
        s += super(MaxCalibrator, self).__repr__()
        s += " calib_amax={_calib_amax}"
        if self._track_amax:
            s += " amaxs={_amaxs}"
        s += ")"
        return s.format(**self.__dict__)

def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|proj_out).*"
    )
    return pattern.match(name) is not None

def filter_func_no_proj_out(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out).*"
    )
    return pattern.match(name) is not None

def quantize_lvl(unet, quant_level=2.5, linear_only=False, enable_conv_3d=True):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            if linear_only:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
            else:
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
        elif isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level >= 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()
        elif isinstance(module, torch.nn.Conv3d) and not enable_conv_3d:
            """
                Error: Torch bug, ONNX export failed due to unknown kernel shape in QuantConv3d.
                TRT_FP8QuantizeLinear and TRT_FP8DequantizeLinear operations in UNetSpatioTemporalConditionModel for svd
                cause issues. Inputs on different devices (CUDA vs CPU) may contribute to the problem.
            """
            module.input_quantizer.disable()
            module.weight_quantizer.disable()
        elif isinstance(module, Attention):
            # TRT only supports FP8 MHA with head_size % 16 == 0.
            head_size = int(module.inner_dim / module.heads)
            if quant_level >= 4 and head_size % 16 == 0:
                module.q_bmm_quantizer.enable()
                module.k_bmm_quantizer.enable()
                module.v_bmm_quantizer.enable()
                module.softmax_quantizer.enable()
            else:
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()

def get_int8_config(
    model,
    quant_level=3,
    alpha=0.8,
    percentile=1.0,
    num_inference_steps=20,
    collect_method="min-mean",
):
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "*output_quantizer": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():
            continue
        if filter_func(name):
            continue
        if isinstance(module, (torch.nn.Linear, LoRACompatibleLinear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}
        elif isinstance(module, (torch.nn.Conv2d, LoRACompatibleConv)):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "collect_method": collect_method,
                    },
                ),
            }
    return quant_config

SD_FP8_FP16_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Half"},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Half",
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

SD_FP8_BF16_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "BFloat16"},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "BFloat16"},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "BFloat16"},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "BFloat16"},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "BFloat16"},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "BFloat16",
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}


SD_FP8_FP32_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*output_quantizer": {"enable": False},
        "*q_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*k_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*v_bmm_quantizer": {"num_bits": (4, 3), "axis": None, "trt_high_precision_dtype": "Float"},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
            "trt_high_precision_dtype": "Float",
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

def set_fmha(unet):
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor())

def check_lora(unet):
    for name, module in unet.named_modules():
        if isinstance(module, (LoRACompatibleConv, LoRACompatibleLinear)):
            assert (
                module.lora_layer is None
            ), f"To quantize {name}, LoRA layer should be fused/merged. Please fuse the LoRA layer before quantization."
        elif USE_PEFT and isinstance(module, (PEFTLoRAConv2d, PEFTLoRALinear)):
            assert (
                module.merged
            ), f"To quantize {name}, LoRA layer should be fused/merged. Please fuse the LoRA layer before quantization."

def generate_fp8_scales(unet):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    for _, module in unet.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)) and (
            hasattr(module.input_quantizer, "_amax") and module.input_quantizer is not None
        ):
            module.input_quantizer._num_bits = 8
            module.weight_quantizer._num_bits = 8
            module.input_quantizer._amax = module.input_quantizer._amax * (127 / 448.0)
            module.weight_quantizer._amax = module.weight_quantizer._amax * (127 / 448.0)
        elif isinstance(module, Attention) and (
            hasattr(module.q_bmm_quantizer, "_amax") and module.q_bmm_quantizer is not None
        ):
            module.q_bmm_quantizer._num_bits = 8
            module.q_bmm_quantizer._amax = module.q_bmm_quantizer._amax * (127 / 448.0)
            module.k_bmm_quantizer._num_bits = 8
            module.k_bmm_quantizer._amax = module.k_bmm_quantizer._amax * (127 / 448.0)
            module.v_bmm_quantizer._num_bits = 8
            module.v_bmm_quantizer._amax = module.v_bmm_quantizer._amax * (127 / 448.0)
            module.softmax_quantizer._num_bits = 8
            module.softmax_quantizer._amax = module.softmax_quantizer._amax * (127 / 448.0)

def get_parent_nodes(node):
    """
    Returns list of input producer nodes for the given node.
    """
    parents = []
    for tensor in node.inputs:
        # If the tensor is not a constant or graph input and has a producer,
        # the producer is a parent of node `node`
        if len(tensor.inputs) == 1:
            parents.append(tensor.inputs[0])
    return parents

def get_child_nodes(node):
    """
    Returns list of output consumer nodes for the given node.
    """
    children = []
    for tensor in node.outputs:
        for consumer in tensor.outputs:  # Traverse all consumer of the tensor
            children.append(consumer)
    return children

def has_path_type(node, graph, path_type, is_forward, wild_card_types, path_nodes):
    """
    Return pattern nodes for the given path_type.
    """
    if not path_type:
        # All types matched
        return True

    # Check if current non-wild node type does not match the expected path type
    node_type = node.op
    is_match = node_type == path_type[0]
    is_wild_match = node_type in wild_card_types
    if not is_match and not is_wild_match:
        return False

    if is_match:
        path_nodes.append(node)
        next_path_type = path_type[1:]
    else:
        next_path_type = path_type[:]

    if is_forward:
        next_level_nodes = get_child_nodes(node)
    else:
        next_level_nodes = get_parent_nodes(node)

    # Check if any child (forward path) or parent (backward path) can match the remaining path types
    for next_node in next_level_nodes:
        sub_path = []
        if has_path_type(next_node, graph, next_path_type, is_forward, wild_card_types, sub_path):
            path_nodes.extend(sub_path)
            return True

    # Path type matches if there is no remaining types to match
    return not next_path_type

def insert_cast(graph, input_tensor, attrs):
    """
    Create a cast layer using tensor as input.
    """
    output_tensor = gs.Variable(name=f"{input_tensor.name}/Cast_output", dtype=attrs["to"])
    next_node_list = input_tensor.outputs.copy()
    graph.layer(
        op="Cast",
        name=f"{input_tensor.name}/Cast",
        inputs=[input_tensor],
        outputs=[output_tensor],
        attrs=attrs,
    )

    # use cast output as input to next node
    for next_node in next_node_list:
        for idx, next_input in enumerate(next_node.inputs):
            if next_input.name == input_tensor.name:
                next_node.inputs[idx] = output_tensor

def convert_zp_fp8(onnx_graph):
    """
    Convert Q/DQ zero datatype from INT8 to FP8.
    """
    # Find all zero constant nodes
    qdq_zero_nodes = set()
    for node in onnx_graph.graph.node:
        if node.op_type == "QuantizeLinear":
            if len(node.input) > 2:
                qdq_zero_nodes.add(node.input[2])

    print(f"Found {len(qdq_zero_nodes)} QDQ pairs")

    # Convert zero point datatype from INT8 to FP8.
    for node in onnx_graph.graph.node:
        if node.output[0] in qdq_zero_nodes:
            node.attribute[0].t.data_type = onnx.TensorProto.FLOAT8E4M3FN

    return onnx_graph

def cast_resize_io(graph):
    """
    After all activations and weights are converted to fp16, we will
    add cast nodes to Resize nodes I/O because Resize need to be run in fp32.
    """
    nodes = graph.nodes
    up_block_resize_regex = r"\/up_blocks.[0-2]\/upsamplers.0\/Resize"
    up_block_resize_nodes = [_n for _n in nodes if re.match(up_block_resize_regex, _n.name)]

    print(f"Found {len(up_block_resize_nodes)} Resize nodes to fix")
    for resize_node in up_block_resize_nodes:
        for input_tensor in resize_node.inputs:
            if input_tensor.name:
                insert_cast(graph, input_tensor=input_tensor, attrs={"to": np.float32})
        for output_tensor in resize_node.outputs:
            if output_tensor.name:
                insert_cast(graph, input_tensor=output_tensor, attrs={"to": np.float16})

def cast_fp8_mha_io(graph):
    r"""
    Insert three cast ops.
    The first cast will be added before the input0 of MatMul to cast fp16 to fp32.
    The second cast will be added before the input1 of MatMul to cast fp16 to fp32.
    The third cast will be added after the output of MatMul to cast fp32 back to fp16.
        Q                  Q
        |                  |
        DQ                 DQ
        |                  |
        Cast               Cast
    (fp16 to fp32)    (fp16 to fp32)
        \                  /
          \              /
            \          /
              MatMul
                |
               Cast (fp32 to fp16)
                |
                Q
                |
                DQ
    The insertion of Cast ops in the FP8 MHA part actually forbids the MHAs to run
    with FP16 accumulation because TensorRT only has FP32 accumulation kernels for FP8 MHAs.
    """
    # Find FP8 MHA pattern.
    # Match FP8 MHA: Q -> DQ -> BMM1 -> (Mul/Div) -> (Add) -> Softmax -> (Cast) -> Q -> DQ -> BMM2 -> Q -> DQ
    softmax_bmm1_chain_type = ["Softmax", "MatMul", "DequantizeLinear", "QuantizeLinear"]
    softmax_bmm2_chain_type = [
        "Softmax",
        "QuantizeLinear",
        "DequantizeLinear",
        "MatMul",
        "QuantizeLinear",
        "DequantizeLinear",
    ]
    wild_card_types = [
        "Div",
        "Mul",
        "ConstMul",
        "Add",
        "BiasAdd",
        "Reshape",
        "Transpose",
        "Flatten",
        "Cast",
    ]

    fp8_mha_partitions = []
    for node in graph.nodes:
        if node.op == "Softmax":
            fp8_mha_partition = []
            if has_path_type(
                node, graph, softmax_bmm1_chain_type, False, wild_card_types, fp8_mha_partition
            ) and has_path_type(
                node, graph, softmax_bmm2_chain_type, True, wild_card_types, fp8_mha_partition
            ):
                if (
                    len(fp8_mha_partition) == 10
                    and fp8_mha_partition[1].op == "MatMul"
                    and fp8_mha_partition[7].op == "MatMul"
                ):
                    fp8_mha_partitions.append(fp8_mha_partition)

    print(f"Found {len(fp8_mha_partitions)} FP8 attentions")

    # Insert Cast nodes for BMM1 and BMM2.
    for fp8_mha_partition in fp8_mha_partitions:
        bmm1_node = fp8_mha_partition[1]
        insert_cast(graph, input_tensor=bmm1_node.inputs[0], attrs={"to": np.float32})
        insert_cast(graph, input_tensor=bmm1_node.inputs[1], attrs={"to": np.float32})
        insert_cast(graph, input_tensor=bmm1_node.outputs[0], attrs={"to": np.float16})

        bmm2_node = fp8_mha_partition[7]
        insert_cast(graph, input_tensor=bmm2_node.inputs[0], attrs={"to": np.float32})
        insert_cast(graph, input_tensor=bmm2_node.inputs[1], attrs={"to": np.float32})
        insert_cast(graph, input_tensor=bmm2_node.outputs[0], attrs={"to": np.float16})

def set_quant_precision(quant_config, precision: str = "Half"):
    for key in quant_config["quant_cfg"]:
        if "trt_high_precision_dtype" in quant_config["quant_cfg"][key]:
            quant_config["quant_cfg"][key]["trt_high_precision_dtype"] = precision

def convert_fp16_io(graph):
    """
    Convert graph I/O to FP16.
    """
    for input_tensor in graph.inputs:
        input_tensor.dtype = onnx.TensorProto.FLOAT16
    for output_tensor in graph.outputs:
        output_tensor.dtype = onnx.TensorProto.FLOAT16
