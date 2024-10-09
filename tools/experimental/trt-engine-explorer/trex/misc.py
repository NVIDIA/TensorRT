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
This file contains miscellanous utility functions.
"""


import pandas as pd
from typing import Dict, List, Tuple
import functools


def group_count(df, grouping_attr):
    grp = df.groupby([grouping_attr]).size().reset_index()
    grp.rename(columns = {0: 'count'}, inplace = True)
    return grp


def group_sum_attr(df, grouping_attr, reduced_attr):
    grp = df.groupby([grouping_attr])[reduced_attr].sum().reset_index()
    return grp


def shape_to_str(shape):
    return "[" + ",".join(str(dim) for dim in shape) + "]"


def _merge_keys_values(
    keys_lists: List[List],
    values_lists: List[List],
    empty_placeholder: object
) -> Dict:
    # Concatenate the keys lists into a set of all keys
    all_keys = set(functools.reduce(lambda a, b: a+b, keys_lists))

    # Create the stacked output dictionary, and fill missing values.
    dicts = [dict(zip(keys, values)) for keys, values in zip(keys_lists, values_lists)]
    result = {}
    for key in all_keys:
        for d in dicts:
            if key not in d:
                d[key] = empty_placeholder
        result[key] = [d[key] for d in dicts]
    return result


def stack_dicts(dict_list: List[dict], empty_placeholder: object=0):
    """Stack lists of dictionaries as a single dictionary"""
    # A list of names lists.
    keys_lists = [list(d.keys()) for d in dict_list]

    # A list of values lists.
    values_lists = [list(d.values()) for d in dict_list]
    return _merge_keys_values(keys_lists, values_lists, empty_placeholder)


def stack_dataframes(
    df_list: List[pd.DataFrame],
    names_col: str,
    values_col: str,
    empty_placeholder: object=0,
):
    # A list of names lists.
    names = [df[names_col].tolist() for df in df_list]

    # A list of values lists.
    values = [df[values_col].tolist() for df in df_list]

    return _merge_keys_values(names, values, empty_placeholder)