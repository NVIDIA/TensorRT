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
import trex


def test_group_count(plan):
    gc = trex.misc.group_count(plan.df, "type").set_index("type")
    gc_exp = plan.df.groupby(["type"]).size()
    assert (gc.loc['Convolution'] == gc_exp.loc['Convolution']).all()
    assert (gc.loc['Pooling'] == gc_exp.loc['Pooling']).all()
    assert (gc.loc['Reformat'] == gc_exp.loc['Reformat']).all()

def test_group_sum_attr(plan):
    gsa = trex.misc.group_sum_attr(plan.df,"type", "latency.avg_time").set_index("type")
    gsa_exp = plan.df.groupby(["type"])[["latency.avg_time"]].sum()
    assert (gsa.loc['Convolution'] == gsa_exp.loc['Convolution']).all()
    assert (gsa.loc['Pooling'] == gsa_exp.loc['Pooling']).all()
    assert (gsa.loc['Reformat'] == gsa_exp.loc['Reformat']).all()
