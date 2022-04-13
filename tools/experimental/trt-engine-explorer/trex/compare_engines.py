#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This file contains interaction cells for the compare_engines.ipynb notebook.
"""


import copy
from typing import List
from functools import partial
from matplotlib.pyplot import colormaps
from .engine_plan import EnginePlan
from .misc import group_count, group_sum_attr
from .plotting import *
from .interactive import *


def get_plans_names(plans: List[EnginePlan]):
    """Create unique plans names"""
    engine_names = [plan.name for plan in plans]
    if len(set(engine_names)) != len(plans):
        engine_names = [plan.name + str(i) for i,plan in enumerate(plans)]
    return engine_names


def compare_engines_overview(plans: List[EnginePlan]):
    """A dropdown widget to choose from several diagrams
    that compare 2 or more engine plans.
    """
    engine_names = get_plans_names(plans)

    df_list = [plan.df for plan in plans]
    time_by_type = [plan.df.groupby(["type"]).sum() \
        [["latency.pct_time", "latency.avg_time"]].reset_index() for plan in plans]
    cnt_by_type = [group_count(plan.df, 'type') for plan in plans]

    # Normalize timings by the batch-size
    df_list_bs_normalized = [copy.deepcopy(plan.df) for plan in plans]
    for i, plan in enumerate(plans):
        inputs, outputs = plan.get_bindings()
        bs = inputs[0].shape[0]
        df_list_bs_normalized[i]['latency.avg_time'] /= bs
    time_by_type_bs_normalized = [df.groupby(["type"]).sum() \
        [["latency.pct_time", "latency.avg_time"]].reset_index() for df in df_list_bs_normalized]

    latency_per_type = partial(
        stacked_bars,
        bar_names=engine_names,
        df_list=time_by_type,
        names_col='type',
        values_col='latency.avg_time',
        colormap=layer_colormap)

    latency_per_layer = partial(
        stacked_bars,
        bar_names=engine_names,
        df_list=df_list,
        names_col='Name',
        values_col='latency.avg_time')

    latency_per_type_bs_normalized = partial(
        stacked_bars,
        bar_names=engine_names,
        df_list=time_by_type_bs_normalized,
        names_col='type',
        values_col='latency.avg_time',
        colormap=layer_colormap)

    latency_per_layer_bs_normalized = partial(
        stacked_bars,
        bar_names=engine_names,
        df_list=df_list_bs_normalized,
        names_col='Name',
        values_col='latency.avg_time')

    convs = [plan.get_layers_by_type('Convolution') for plan in plans]
    stacked_convs = partial(
        stacked_bars,
        bar_names=engine_names,
        df_list=convs,
        names_col='Name',
        values_col='latency.avg_time')

    stacked_convs_by_precision = partial(
        stacked_bars,
        bar_names=engine_names,
        df_list=convs,
        names_col='precision',
        values_col='latency.avg_time',
        colormap=precision_colormap)

    d = {engine_name:df for engine_name, df in zip(engine_names, time_by_type)}
    latency_comparison = partial(
        plotly_bar2,
        df=d,
        values_col="latency.avg_time",
        names_col="type",
        orientation='h',
        showlegend=True)

    d = {engine_name:df for engine_name, df in zip(engine_names, cnt_by_type)}
    count_comparison = partial(
        plotly_bar2,
        df=d,
        values_col="count",
        names_col="type",
        orientation='h')

    stacked_layers_by_precision = partial(
        stacked_bars,
        bar_names=engine_names,
        df_list=[plan.df for plan in plans],
        names_col='precision',
        values_col='latency.avg_time',
        colormap=precision_colormap)

    precision_subplots = [(
        group_count(
            plan.df, 'precision'),
            plan.name, 'count', 'precision'
        ) for plan in plans]
    precision_cnts = partial(
        plotly_pie2,
        charts=precision_subplots,
        colormap=precision_colormap)

    precision_subplots = [(
        group_sum_attr(
            plan.df, grouping_attr='precision',
            reduced_attr='latency.pct_time'),
            plan.name, 'latency.pct_time', 'precision'
        ) for plan in plans]
    precision_latency = partial(
        plotly_pie2,
        charts=precision_subplots,
        colormap=precision_colormap)

    dropdown_choices = {
        "Stacked latencies by layer type": latency_per_type,
        "Stacked latencies by layer": latency_per_layer,
        "Stacked latencies by layer type (BS normalized)": latency_per_type_bs_normalized,
        "Stacked latencies by layer (BS normalized)": latency_per_layer_bs_normalized,
        "Stacked convolution latencies": stacked_convs,
        "Stacked convolution latencies (by precision)": stacked_convs_by_precision,
        "Layer count comparison": count_comparison,
        "Layer latency comparison": latency_comparison,
        "Layer latency by precision": stacked_layers_by_precision,
        "Precision counts": precision_cnts,
        "Precision latency": precision_latency,
    }

    InteractiveDiagram_2(dropdown_choices, 'Diagram:')
