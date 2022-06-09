#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import json
from trex import *
import pytest

"""
Test the algorithm used to align the layers of two engines.
"""

test_dir_path = os.path.dirname(os.path.realpath(__file__))


def is_match(engine_name_1, engine_name_2, expected_pairs):
    plan1 = EnginePlan(f'{engine_name_1}.graph.json')
    plan2 = EnginePlan(f'{engine_name_2}.graph.json')

    matched_indices_pairs = match_layers(plan1, plan2, exact_matching=True)
    print(matched_indices_pairs)
    print(expected_pairs)
    return expected_pairs == matched_indices_pairs


def test_alignment_algorithm_2():
    expected_pairs = [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (9, 7), (7, None),
        (10, 8), (8, None), (11, 9), (12, None), (13, 10), (14, 11), (15, 12),
        (16, None), (17, 13), (18, 14), (19, 15), (20, None), (21, 16), (22, 17),
        (23, 18), (24, None), (25, 19), (26, 20), (27, 21), (28, None), (30, 22),
        (29, None), (31, 23), (32, 24), (35, 25), (33, None), (36, 26), (34, None),
        (37, 27), (38, None), (40, 28), (39, None), (41, 29), (42, 30), (45, 31),
        (43, None), (46, 32), (44, None), (47, 33), (48, None), (50, 34), (49, None),
        (51, 35), (52, 36), (55, 37), (53, None), (56, 38), (54, None), (57, 39),
        (58, None), (60, 40), (59, None), (61, 41), (62, 42), (65, 43), (63, None),
        (66, 44), (64, None), (67, 45), (68, None), (70, 46), (69, None), (71, 47),
        (72, 48), (75, 49), (73, None), (76, 50), (74, None), (77, 51), (78, 52),
        (79, 53), (80, 54), (81, 55), (82, 56)]
    engine_name_1 = os.path.join(test_dir_path, "inputs", "mobilenet.qat.onnx.engine")
    engine_name_2 = os.path.join(test_dir_path, "inputs", "mobilenet_v2_residuals.qat.onnx.engine")
    is_match(engine_name_1, engine_name_2, expected_pairs)

