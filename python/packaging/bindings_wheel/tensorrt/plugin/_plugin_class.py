#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Tuple

import numpy as np
from ._utils import _numpy_to_plugin_field_type, _built_in_to_plugin_field_type
from ._tensor import TensorDesc, Tensor, Shape, ShapeExpr, ShapeExprs
from ._autotune import _TypeFormatCombination


class _TemplatePlugin(
    trt.IPluginV3,
    trt.IPluginV3QuickCore,
    trt.IPluginV3QuickBuild,
    trt.IPluginV3QuickRuntime,
):
    def __init__(self, name, namespace, num_outputs):
        trt.IPluginV3.__init__(self)
        trt.IPluginV3QuickCore.__init__(self)
        trt.IPluginV3QuickBuild.__init__(self)
        trt.IPluginV3QuickRuntime.__init__(self)

        self.plugin_version = "1"
        self.input_types = []
        self.aliased_map = {}  # output index -> input index

        self.plugin_namespace = namespace
        self.plugin_name = name
        self.num_outputs = num_outputs

        self.autotune_combs = []
        self.supported_combs = {}
        self.curr_comb = None
        self.expects_tactic = False

    def init(
        self,
        register_function,
        attrs,
        impl_attr_names,
        impl_function,
        autotune_attr_names,
        autotune_function,
        expects_tactic,
    ):
        self.register_function = register_function
        self.impl_function = impl_function
        self.attrs = attrs
        self.impl_attr_names = impl_attr_names
        self.autotune_attr_names = autotune_attr_names
        self.autotune_function = autotune_function
        self.expects_tactic = expects_tactic

    def get_capability_interface(self, type):
        return self

    def get_num_outputs(self):
        return self.num_outputs

    def get_output_data_types(self, input_types, ranks):
        self.input_types = input_types

        input_descs = [None] * len(input_types)
        input_desc_map = {}
        for i in range(len(input_types)):
            input_descs[i] = TensorDesc()
            input_descs[i].dtype = input_types[i]
            input_descs[i].shape_expr = ShapeExprs(ranks[i], _is_dummy=True)
            input_descs[i]._immutable = True
            input_desc_map[id(input_descs[i])] = i

        output_descs = self.register_function(*input_descs, **self.attrs)
        if not isinstance(output_descs, Tuple):
            output_descs = tuple([output_descs])

        self.output_types = []

        for i in range(len(output_descs)):
            self.output_types.append(output_descs[i].dtype)

            if output_descs[i].get_aliased() is not None:
                self.aliased_map[i] = input_desc_map[id(output_descs[i].get_aliased())]
            else:
                self.aliased_map[i] = -1

        return self.output_types

    def get_fields_to_serialize(self):
        fields = []
        for key, value in self.attrs.items():
            if key in self.impl_attr_names:
                if isinstance(value, np.ndarray):
                    if np.dtype(value.dtype) == np.float16:
                        fields.append(
                            trt.PluginField(
                                key, value.tobytes(), trt.PluginFieldType.UNKNOWN
                            )
                        )
                    else:
                        fields.append(
                            trt.PluginField(
                                key,
                                value,
                                _numpy_to_plugin_field_type[np.dtype(value.dtype)],
                            )
                        )
                elif isinstance(value, str):
                    fields.append(
                        trt.PluginField(key, value.encode(), trt.PluginFieldType.CHAR)
                    )
                elif isinstance(value, bytes):
                    fields.append(
                        trt.PluginField(key, value, trt.PluginFieldType.UNKNOWN)
                    )
                else:
                    fields.append(
                        trt.PluginField(
                            key,
                            np.array([value]),
                            _built_in_to_plugin_field_type[type(value)],
                        )
                    )

        return trt.PluginFieldCollection(fields)

    def get_output_shapes(self, inputs, shape_inputs, exprBuilder):
        assert len(shape_inputs) == 0  # Shape inputs are not yet supported for QDPs
        ShapeExpr._exprBuilder = exprBuilder
        self.input_descs = []
        for i in range(len(inputs)):
            desc = TensorDesc()
            inp = inputs[i]

            desc.dtype = self.input_types[i]
            desc.shape_expr = ShapeExprs(len(inp))
            for j in range(len(inp)):
                desc.shape_expr[j] = ShapeExpr(inp[j])
            desc._immutable = True

            self.input_descs.append(desc)

        self.output_descs = self.register_function(*self.input_descs, **self.attrs)
        if not isinstance(self.output_descs, Tuple):
            self.output_descs = tuple([self.output_descs])

        for idx, desc in enumerate(self.output_descs):
            if desc.is_size_tensor:
                desc._set_index(idx)

        output_exprs = []
        for i in range(len(self.output_descs)):
            exprs = trt.DimsExprs(len(self.output_descs[i].shape_expr))
            for j in range(len(exprs)):
                exprs[j] = self.output_descs[i].shape_expr[j]._expr

            output_exprs.append(exprs)

        return output_exprs

    def configure_plugin(self, inputs, outputs):
        self.curr_comb = _TypeFormatCombination()
        self.curr_comb.types = [inp.desc.type for inp in inputs] + [
            out.desc.type for out in outputs
        ]
        self.curr_comb.layouts = [inp.desc.format for inp in inputs] + [
            out.desc.format for out in outputs
        ]

    def get_supported_format_combinations(self, in_out, num_inputs):
        if self.autotune_function is not None:
            if len(self.autotune_attr_names) > 0:
                val = [self.attrs[k] for k in self.autotune_attr_names]
            else:
                val = ()

            for i, desc in enumerate(in_out):
                if i < num_inputs:
                    self.input_descs[i]._immutable = False
                    self.input_descs[i].shape = Shape(desc)
                    self.input_descs[i].format = desc.desc.format
                    self.input_descs[i].scale = desc.desc.scale
                    self.input_descs[i]._immutable = True
                else:
                    self.output_descs[i - num_inputs]._immutable = False
                    self.output_descs[i - num_inputs].shape = Shape(desc)
                    self.output_descs[i - num_inputs].format = desc.desc.format
                    self.output_descs[i - num_inputs].scale = desc.desc.scale
                    self.output_descs[i - num_inputs]._immutable = True

            self.autotune_combs = self.autotune_function(
                *self.input_descs, *val, self.output_descs
            )

        if len(self.autotune_combs) == 0:
            default_comb = [None] * len(in_out)
            comb = _TypeFormatCombination(len(in_out))
            for j in range(len(in_out)):
                default_comb[j] = trt.PluginTensorDesc()
                default_comb[j].type = (
                    self.input_types[j]
                    if j < num_inputs
                    else self.output_descs[j - num_inputs].dtype
                )
                default_comb[j].format = trt.TensorFormat.LINEAR
                comb.types[j] = default_comb[j].type
                comb.layouts[j] = default_comb[j].format

            self.supported_combs[comb] = set()

            return default_comb

        all_combs = []
        for comb in self.autotune_combs:
            all_combs.extend(comb._get_combinations())

        ret_supported_combs = []
        self.supported_combs = {}

        for i, comb in enumerate(all_combs):
            value = self.supported_combs.get(comb)
            if value is not None:
                value.update(set(comb.tactics) if comb.tactics is not None else set())
            else:
                self.supported_combs[comb] = (
                    set(comb.tactics) if comb.tactics is not None else set()
                )
                for j in range(len(in_out)):
                    curr_comb = trt.PluginTensorDesc()
                    curr_comb.type = comb.types[j]
                    curr_comb.format = comb.layouts[j]
                    ret_supported_combs.append(curr_comb)

        return ret_supported_combs

    def enqueue(
        self,
        input_desc,
        output_desc,
        inputs,
        outputs,
        in_strides,
        out_strides,
        stream,
    ):
        input_tensors = [None] * (len(inputs))
        aliased_input_idxs = list(self.aliased_map.values())

        for i in range(len(inputs)):
            input_tensors[i] = Tensor()
            input_tensors[i].dtype = input_desc[i].type
            input_tensors[i].shape = Shape(input_desc[i])
            input_tensors[i].format = input_desc[i].format
            input_tensors[i].scale = input_desc[i].scale
            input_tensors[i].data_ptr = inputs[i]
            input_tensors[i]._stream = stream
            input_tensors[i]._read_only = i not in aliased_input_idxs
            input_tensors[i].strides = in_strides[i]

        output_tensors = [None] * (len(outputs))
        for i in range(len(outputs)):
            output_tensors[i] = Tensor()
            output_tensors[i].dtype = output_desc[i].type
            output_tensors[i].shape = Shape(output_desc[i])
            output_tensors[i].format = output_desc[i].format
            output_tensors[i].scale = output_desc[i].scale
            output_tensors[i].data_ptr = outputs[i]
            output_tensors[i]._stream = stream
            output_tensors[i]._read_only = False
            output_tensors[i].strides = out_strides[i]

        for i, j in self.aliased_map.items():
            output_tensors[i]._aliased_to = input_tensors[j]
            input_tensors[j]._aliased_to = output_tensors[i]

        for t in input_tensors:
            t._immutable = True

        for t in output_tensors:
            t._immutable = True

        if len(self.impl_attr_names) > 0:
            val = [self.attrs[k] for k in self.impl_attr_names]
        else:
            val = ()

        if self.expects_tactic:
            self.impl_function(
                *input_tensors, *val, output_tensors, stream, self._tactic
            )
        else:
            self.impl_function(*input_tensors, *val, output_tensors, stream=stream)

    def get_aliased_input(self, output_index: int):
        return self.aliased_map[output_index]

    def get_valid_tactics(self):
        tactics = self.supported_combs.get(self.curr_comb)
        assert tactics is not None
        return list(tactics)

    def set_tactic(self, tactic):
        self._tactic = tactic

    def clone(self):
        cloned_plugin = _TemplatePlugin(
            self.plugin_name, self.plugin_namespace, self.num_outputs
        )
        cloned_plugin.__dict__.update(self.__dict__)
        return cloned_plugin
