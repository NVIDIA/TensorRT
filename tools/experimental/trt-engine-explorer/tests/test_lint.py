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


from .util import plan
from trex import ConvLinter, ReformatLinter

class TestConvLinter:
    def test_tc_lint(self, plan):
        df = ConvLinter(plan).lint()
        tc_lint_row = df.loc['features.1.conv.0.0.weight + QuantizeLinear_25 + Conv_29 + PWN(Clip_33)']
        assert tc_lint_row['name'] == 'features.1.conv.0.0.weight + QuantizeLinear_25 + Conv_29 + PWN(Clip_33)'
        assert tc_lint_row['hazard'] == 'Convolution is not accelerated.'

    def test_mixed_precision_lint(self, plan):
        df = ConvLinter(plan).lint()
        alignment_lint_row = df.loc['features.4.conv.2.weight + QuantizeLinear_187 + Conv_191']
        assert alignment_lint_row['name'] == 'features.4.conv.2.weight + QuantizeLinear_187 + Conv_191'
        assert alignment_lint_row['hazard'] == 'Quantized Convolution has float outputs.'
        assert alignment_lint_row['mitigation'] == 'Consider adding quantization after the convolution.'

    def test_alignment_lint(self, plan):
        df = ConvLinter(plan).lint()
        alignment_lint_row = df.loc['features.0.0.weight + QuantizeLinear_8 + Conv_12 + PWN(Clip_16)']
        assert alignment_lint_row['name'] == 'features.0.0.weight + QuantizeLinear_8 + Conv_12 + PWN(Clip_16)'
        assert alignment_lint_row['hazard'] == 'Convolution channels are not optimally aligned.'
        assert alignment_lint_row['mitigation'] == "Consider changing the alignment of the convolution's channels."

class TestReformatLinter:
    def test_lint(self, plan):
        df = ReformatLinter(plan).lint()
        lint_row = df.loc['QuantizeLinear_2']
        assert lint_row['name'] == 'QuantizeLinear_2'
        assert lint_row['origin'] == 'QDQ'
        assert lint_row['type conversion'] == 'FP32 -> Int8'
        assert lint_row['shape conversion'] == '[1, 3, 224, 224] -> [1, 3, 224, 224]'
        assert lint_row['hazard'] == 'Reformat layer is converting operand data type.'
