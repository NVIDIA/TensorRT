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

"""
This file contains layer linting functions.
"""


from collections import OrderedDict
from collections import OrderedDict
from typing import Dict
import pandas as pd
from .activations import create_activations
from .engine_plan import EnginePlan


class ConvLinter():
    """Convolution layer linter."""

    def __init__(self, plan: EnginePlan):
            self.plan = plan
            self.convs = plan.get_layers_by_type('Convolution')

    def tc_lint(self):
        """Search for Convolutions which are not accelerated by TensorCode"""

        def is_small_conv(conv):
            inputs, _ = create_activations(conv)
            if len(inputs[0].shape) != 4:
                return False
            n, c, h, w = inputs[0].shape
            return c < 32

        report = OrderedDict()

        # Look for kernels that are not scheduled for TensorCore acceleration
        tc_candidates = self.convs.query(f"precision != \"FP32\"").copy()

        # Identify acceleration from tactic name
        df = tc_candidates
        df = df[df['tactic'].str.contains("imma|hmma|xmma|i88|884", na=False) == False]
        for _, conv in df.iterrows():
            mitigation = ""
            if is_small_conv(conv):
                mitigation = "This Convolution has a small number " \
                    "of input channels so acceleration may not be possible."

            report[conv.Name] = OrderedDict({
                'name': conv.Name,
                'tactic': conv.tactic,
                'subtype': conv.subtype,
                'hazard': "Convolution is not accelerated.",
                'mitigation': mitigation,
                'help': "TensorCores accelerate large Convolution and GEMM operations."
            })
        return report

    def mixed_precision_lint(self):
        """Search for Convolutions with Int8 inputs and Float outputs"""
        report = OrderedDict()

        df = self.convs
        df = df.loc[df['precision'] == 'INT8'].copy()
        for index, conv in df.iterrows():
            inputs, outputs = create_activations(conv)
            inf = inputs[0].format[:4]
            outf = outputs[0].format[:4]
            found = inf == 'Int8' and outf != 'Int8'
            if found:
                report[conv.Name] = OrderedDict({
                'name': conv.Name,
                'tactic': conv.tactic,
                'subtype': conv.subtype,
                'hazard': "Quantized Convolution has float outputs.",
                'mitigation': "Consider adding quantization after the convolution.",
                'help': "Quantized Convolution with float outputs is ill advised "
                        "for memory-limited convolutions."
            })
        return report

    def alignment_lint(self):
        def alignment_ok(conv):
            inputs, outputs = create_activations(conv)
            prec = conv["precision"]
            if len(inputs[0].shape) != 4 or len(outputs[0].shape) != 4:
                return True
            N, C, H, W = inputs[0].shape
            _, K, P, Q = outputs[0].shape
            if prec == 'INT8':
                ok = (C % 16 == 0) and (K % 16 == 0)
            else:
                ok = (C % 8 == 0) and (K % 8 == 0)
            return ok

        report = OrderedDict()
        for _, conv in self.convs.iterrows():
            if not alignment_ok(conv):
                report[conv.Name] = {
                    'name': conv.Name,
                    'tactic': conv.tactic,
                    'subtype': conv.subtype,
                    'hazard': "Convolution channels are not optimally aligned.",
                    'mitigation': "Consider changing the alignment of the convolution's channels.",
                    'help': "For best performance, the input and outputs channels of a Tensor Core"
                            "accelerated convolution should be aligned to 8 (FP32/FP16) or 16 (INT8)"
                }
        return report

    def lint(self):
            report = self.tc_lint()
            report.update(self.mixed_precision_lint())
            report.update(self.alignment_lint())
            df = pd.DataFrame.from_dict(report, orient='index')
            return df


class ReformatLinter():
    """Reformat layer linter."""

    def __init__(self, plan: EnginePlan):
            self.plan = plan
            self.reformats = plan.get_layers_by_type('Reformat')

    def lint(self):
        """Search for conversions between types.

        Conversions between layouts are assumed to be optimized."""
        report = OrderedDict()

        for index, reformat in self.reformats.iterrows():
            inputs, outputs = create_activations(reformat)
            inf = inputs[0].format[:4]
            outf = outputs[0].format[:4]
            if inf != outf:
                mitigation = ""
                if "INT8" in [inf, outf]:
                    mitigation = "Consider adding quantization around float operations."
                report[reformat.Name] = OrderedDict({
                    'name': reformat.Name,
                    'origin': reformat['attr.origin'],
                    'type conversion': f"{inf} -> {outf}",
                    'shape conversion': f"{inputs[0].shape} -> {outputs[0].shape}",
                    'hazard': "Reformat layer is converting operand data type.",
                    'mitigation': mitigation,
                    'help': "Conversions between float32 and float16 are a red "
                            "flag, as are conversions between float32/16 and INT8."
                })

        df = pd.DataFrame.from_dict(report, orient='index')
        return df


class SliceLinter():
    """Slice layer linter."""

    def __init__(self, plan: EnginePlan):
            self.plan = plan
            self.slices = plan.get_layers_by_type('Slice')

    def lint(self):
        """Search for conversions between types.

        Conversions between layouts are assumed to be optimized."""
        report = OrderedDict()

        for index, slice in self.slices.iterrows():
            inputs, outputs = create_activations(slice)
            inf = inputs[0].format[:4]
            outf = outputs[0].format[:4]
            if inf != outf:
                mitigation = ""
                if "INT8" in [inf, outf]:
                    mitigation = "Consider adding quantization around float operations."
                report[slice.Name] = OrderedDict({
                    'name': slice.Name,
                    'type conversion': f"{inf} -> {outf}",
                    'shape conversion': f"{inputs[0].shape} -> {outputs[0].shape}",
                    'hazard': "Slice layer is converting operand data type.",
                    'mitigation': mitigation,
                    'help': "Conversions between float32 and float16 are a red "
                            "flag, as are conversions between float32/16 <=> INT8."
                })

        df = pd.DataFrame.from_dict(report, orient='index')
        return df


class QDQLinter():
    """Q/DQ layer linter."""

    def __init__(self, plan: EnginePlan):
            self.plan = plan
            self.scales = plan.get_layers_by_type('Scale')

    def lint(self):
        """Search for dangling Q/DQ layers."""
        report = OrderedDict()

        for index, scale in self.scales.iterrows():
            inputs, outputs = create_activations(scale)
            inf = inputs[0].format[:4]
            outf = outputs[0].format[:4]
            is_qdq = ('Int8' in inputs[0].format) ^ ('Int8' in outputs[0].format)
            if is_qdq:
                dq = 'Int8' in inputs[0].format
                role = "Quantize" if not dq else "Dequanitize"
                report[scale.Name] = OrderedDict({
                    'name': scale.Name,
                    'type conversion': f"{inf} -> {outf}",
                    'hazard': f"Unfused {role} layer",
                    'mitigation': f"Check why the {role} layer is not fused",
                    'help': f"Unfused Quantize/Dequantize nodes are wasteful and "
                            "should be avoided. Quantize nodes may be necessary "
                            "for quantizing inputs."
                })

        df = pd.DataFrame.from_dict(report, orient='index')
        return df
