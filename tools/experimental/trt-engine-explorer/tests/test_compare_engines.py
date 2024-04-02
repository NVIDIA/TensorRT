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


from .util import plan, plan2
from trex.compare_engines import get_plans_names, match_layers


def test_get_plans_names(plan, plan2):
    assert get_plans_names([plan, plan2]) == [
        'mobilenet.qat.onnx.engine.graph.json',
        'mobilenet_v2_residuals.qat.onnx.engine.graph.json'
        ]

def test_match_layers(plan, plan2):
    expected_pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
    (8, 7), (7, None), (9, 8), (10, 9), (12, 10), (11, None), (13, 11),
    (14, 12), (16, 13), (15, None), (17, 14), (18, 15), (20, 16), (19, None),
    (21, 17), (22, 18), (24, 19), (23, None), (25, 20), (26, 21), (28, 22),
    (27, None), (29, 23), (30, 24), (32, 25), (31, None), (33, 26), (34, 27),
    (36, 28), (35, None), (37, 29), (38, 30), (40, 31), (39, None), (41, 32),
    (42, 33), (44, 34), (43, None), (45, 35), (46, 36), (48, 37), (47, None),
    (49, 38), (50, 39), (52, 40), (51, None), (53, 41), (54, 42), (56, 43),
    (55, None), (57, 44), (58, 45), (60, 46), (59, None), (61, 47), (62, 48),
    (64, 49), (63, None), (65, 50), (66, 51), (67, 52), (68, 53), (69, 54),
    (70, 55), (71, 56)]

    matched_indices_pairs = match_layers(plan, plan2, exact_matching=True)
    assert expected_pairs == matched_indices_pairs
