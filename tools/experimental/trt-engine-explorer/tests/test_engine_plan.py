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
from trex import summary_dict
import pandas as pd

def test_summary_dict(plan):
    d = summary_dict(plan)
    assert d["Inputs"] == "input1: [1, 3, 224, 224]xFP32 NCHW"
    assert d["Average time"] == "0.470 ms"
    assert d["Layers"] == "72"
    assert d["Weights"] == "3.3 MB"
    assert d["Activations"] == "15.9 MB"

class TestEnginePlan:
    def test_df(self, plan):
        df = plan.df
        assert isinstance(df, (pd.DataFrame))
