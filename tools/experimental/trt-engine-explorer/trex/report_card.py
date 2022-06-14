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

"""
Performance report card.

This file contains interaction cells for the engine_report_card.ipynb notebook.
"""


from functools import partial
from .misc import group_count, group_sum_attr
from .interactive import InteractiveDiagram, InteractiveDiagram_2
from .notebook import display_df
from .df_preprocessing import clean_for_display
from .plotting import *
from .graphing import *
from .parser import read_timing_file


def report_card_perf_overview(plan: EnginePlan):
    """Display performance overview diagrams.

    Display a dropdown widget to choose between diagrams showing various
    characteristics of the plan's convolution layers."""
    layer_types = group_count(plan.df, 'type')
    count_per_layer_type = partial(
        plotly_bar2,
        layer_types,
        values_col='count',
        names_col='type',
        color='type', colormap=layer_colormap,
        orientation='h',
        show_axis_ticks=(True, True))

    time_pct_by_type = plan.df.groupby(['type']).sum()[
        ['latency.pct_time', 'latency.avg_time']].reset_index()
    latency_per_type_ms = partial(
        plotly_bar2,
        time_pct_by_type,
        values_col='latency.avg_time',
        names_col='type',
        color='type', colormap=layer_colormap,
        orientation='h',
        show_axis_ticks=(True, True))

    latency_per_type_pct = partial(
        plotly_bar2,
        time_pct_by_type,
        values_col='latency.pct_time',
        names_col='type',
        color='type', colormap=layer_colormap,
        orientation='h',
        show_axis_ticks=(True, True))

    precision_per_layer = partial(
        plotly_bar2,
        plan.df,
        values_col='latency.avg_time',
        names_col='Name',
        color='precision', colormap=precision_colormap,
        xaxis_title="Layer")

    output_precision_per_layer = partial(
        plotly_bar2,
        plan.df,
        values_col='latency.avg_time',
        names_col='Name',
        color='output_precision', colormap=precision_colormap,
        xaxis_title="Layer")

    output_precision_per_layer = partial(
        plotly_bar2,
        plan.df,
        values_col='latency.avg_time',
        names_col='Name',
        color='output_precision', colormap=precision_colormap,
        xaxis_title="Layer")

    latency_distribution = partial(
        plotly_hist,
        plan.df,
        values_col='latency.pct_time',
        xaxis_title = 'Latency (ms)',
        color='type', colormap=layer_colormap)

    latency_per_layer = partial(
        plotly_bar2,
        plan.df,
        values_col='latency.pct_time',
        names_col='Name',
        color='type', colormap=layer_colormap,
        xaxis_title="Layer")

    latency_per_layer_ms = partial(
        plotly_bar2,
        plan.df,
        values_col='latency.avg_time',
        names_col='Name',
        color='type', colormap=layer_colormap,
        xaxis_title="Layer")

    precision_charts = []
    layer_precisions = group_count(plan.df, 'precision')
    precision_charts.append((
        layer_precisions,
        'Layer Count By Precision',
        'count',
        'precision'))

    layers_time_pct_by_precision = group_sum_attr(
        plan.df,
        grouping_attr='precision',
        reduced_attr='latency.pct_time')

    precision_charts.append((
        layers_time_pct_by_precision,
        '% Latency By Precision',
        'latency.pct_time',
        'precision'))

    precision_statistics = partial(
        plotly_pie2,
        charts=precision_charts,
        colormap=precision_colormap)

    def precision_per_type(title):
        title = f"{title}\n({plan.name})"
        df = plan.df
        precision_sunburst = df.groupby(['type', 'precision']).count().reset_index()
        color = [precision_colormap[p] for p in df['precision']]
        fig = px.sunburst(
            precision_sunburst,
            path=['type', 'precision'],
            values='Name',
            color_discrete_map=precision_colormap,
            color='precision')
        fig.update_layout(title=title, title_x=0.5, font_size=15,)
        fig.show()

    dropdown_choices = {
        "Latency per layer (%)": latency_per_layer,
        "Latency per layer (ms)": latency_per_layer_ms,
        "Layer latency distribution": latency_distribution,
        "Precision per layer": precision_per_layer,
        "Output precision per layer": output_precision_per_layer,
        "Count per layer type": count_per_layer_type,
        "Latency per layer type (ms)": latency_per_type_ms,
        "Latency per layer type (%)": latency_per_type_pct,
        "Precision per layer type": precision_per_type,
        "Precision rollup": precision_statistics,
    }

    return InteractiveDiagram_2(dropdown_choices, 'Diagram:')


def report_card_convolutions_overview(convs: pd.DataFrame):
    """Display convolution layer diagrams.

    Display a dropdown widget to choose between diagrams showing various
    characteristics of the plan's convolution layers"""

    latency_vs_ai_per_conv = partial(
        plotly_bar2,
        convs,
        values_col='latency.pct_time',
        names_col='Name',
        color='attr.arithmetic_intensity', colormap=precision_colormap)
    latency_vs_prec_per_conv = partial(
        plotly_bar2,
        convs,
        values_col='latency.pct_time',
        names_col='Name',
        color='precision', colormap=precision_colormap)
    latency_vs_fmas = partial(
        plotly_bar2,
        convs,
        values_col='latency.pct_time',
        names_col='Name',
        color='attr.macs')
    latency_vs_data = partial(
        plotly_bar2,
        convs,
        values_col='latency.pct_time',
        names_col='Name',
        color='total_footprint_bytes')
    latency_vs_ce_per_conv = partial(
        plotly_bar2,
        convs,
        values_col='latency.pct_time',
        names_col='Name',
        color='attr.compute_efficiency', colormap=precision_colormap)
    latency_vs_group_size = partial(
        plotly_bar2,
        convs,
        values_col='latency.pct_time',
        names_col='Name',
        color='attr.groups')
    latency_vs_kernel_size = partial(
        plotly_bar2,
        convs,
        values_col='latency.pct_time',
        names_col='Name',
        color='attr.kernel')
    footprint_per_conv = partial(
        plotly_bar2,
        convs,
        values_col='total_footprint_bytes',
        names_col='Name',
        color='latency.pct_time')
    fmas_per_conv = partial(
        plotly_bar2,
        convs,
        values_col='attr.macs',
        names_col='Name',
        color='latency.pct_time')
    ai_vs_latency_per_conv = partial(
        plotly_bar2,
        convs,
        values_col='attr.arithmetic_intensity',
        names_col='Name',
        color='latency.pct_time')
    ai_vs_footprint_per_conv = partial(
        plotly_bar2,
        convs,
        values_col='attr.arithmetic_intensity',
        names_col='Name',
        color='total_footprint_bytes')
    ce_vs_latency_per_conv = partial(
        plotly_bar2,
        convs,
        values_col='attr.compute_efficiency',
        names_col='Name',
        color='latency.pct_time')
    me_vs_latency_per_conv = partial(
        plotly_bar2,
        convs,
        values_col='attr.memory_efficiency',
        names_col='Name',
        color='latency.pct_time')

    dropdown_choices = {
        "Latency per convolution (color = precision)":
            latency_vs_prec_per_conv,
        "Latency per convolution (color = FMAs)":
            latency_vs_fmas,
        "Latency per convolution (color = data size)":
            latency_vs_data,
        "Latency per convolution (color = arithmetic intensity)":
            latency_vs_ai_per_conv,
        "Latency per convolution (color = compute efficiency)":
            latency_vs_ce_per_conv,
        "Latency per convolution (color = group size)":
            latency_vs_group_size,
        "Latency per convolution (color = kernel size)":
            latency_vs_kernel_size,
        "Data footprints": footprint_per_conv,
        "Fused Multiply-Accumulate (FMAs)": fmas_per_conv,
        "Arithmetic intensity (color = latency)":
            ai_vs_latency_per_conv,
        "Arithmetic intensity (color = footprint)":
            ai_vs_footprint_per_conv,
        "Compute efficiency (color = latency)":
            ce_vs_latency_per_conv,
        "Memory efficiency (color = latency)":
            me_vs_latency_per_conv,
    }
    InteractiveDiagram_2(dropdown_choices, 'Diagram:')


def report_card_table_view(plan: EnginePlan):
    """Layers tabular views.

    Display a dropdown widget to choose among tabular views of all, or specific,
    layers"""

    def render_diagram(choice, ignore):
        if choice == 'All':
            display_df(clean_for_display(plan.df))
        else:
            df = plan.get_layers_by_type(choice)
            print(f"There are {len(df)} {choice} layers which account for"
                  f"{df['latency.pct_time'].sum(): .2f}% ({df['latency.avg_time'].sum(): .5f} ms) of the overall latency.")
            display_df(clean_for_display(df))

    types = ['All'] + list(set(plan.df['type']))
    dropdown_choices = {t: t for t in types}
    InteractiveDiagram(render_diagram, dropdown_choices, 'Dataframe')


def report_card_memory_footprint(plan: EnginePlan):
    """Memory footprint diagrams"""

    def render_diagram(choice, values_col, colormap):
        if 'distribution' in choice:
            plotly_hist(
                plan.df,
                f"{choice}",
                values_col,
                "Size (bytes)",
                color='type',
                colormap=colormap)
        else:
            plotly_bar2(
                plan.df,
                f"{choice}",
                values_col,
                "Name",
                color='type',
                colormap=colormap,
                show_axis_ticks=(False, True))

    dropdown_choices = {
        "Weights footprint per layer":
            ('weights_size', layer_colormap),
        "Activation footprint per layer":
            ('total_io_size_bytes', layer_colormap),
        "Total footprint per layer":
            ('total_footprint_bytes', layer_colormap),
        "Weights footprint distribution per layer":
            ('weights_size', layer_colormap),
        "Activations footprint distribution per layer":
            ('total_io_size_bytes', layer_colormap),
        "Total footprint distribution per layer":
            ('total_footprint_bytes', layer_colormap),
    }

    InteractiveDiagram(render_diagram, dropdown_choices, 'Bar color')


def report_card_draw_plan_graph(plan: EnginePlan, engine_name: str):
    """Draw the plan graph (export to SVG)"""
    def render_diagram(choice, formatter, display_regions, expand_layer_details):
        graph = to_dot(plan, formatter,
            display_regions=display_regions,
            expand_layer_details=expand_layer_details)
        render_dot(graph, engine_name, 'svg')

    # Color code nodes by precision or layer-type
    dropdown_choices = {
        "Color nodes by type": (layer_type_formatter, False, False),
        "Color nodes by type (detailed)": (layer_type_formatter, True, True),
        "Color nodes by precision": (precision_formatter, False, False),
        "Color nodes by precision (detailed)": (precision_formatter, True, True),
    }

    InteractiveDiagram(render_diagram, dropdown_choices, 'Color formatting:')


def report_card_pointwise_lint(plan: EnginePlan):
    pws = plan.get_layers_by_type('PointWise')

    if len(pws) == 0:
        print("The engine plan does not contain pointwise layers.")
        return

    charts = []
    by_n_operations = group_count(pws, 'attr.n_operations')
    charts.append((by_n_operations,
        "Pointwise layers by number of operations", 'count', 'attr.n_operations'))

    layers_time_pct_by_n_operations = group_sum_attr(
        pws,
        grouping_attr='attr.n_operations',
        reduced_attr='latency.pct_time')

    charts.append((
        layers_time_pct_by_n_operations,
        "Latency by number of operations (%)",
        'latency.pct_time',
        'attr.n_operations'))

    df = layers_time_pct_by_n_operations.merge(by_n_operations, on='attr.n_operations')
    df['per_op_latency'] = df['latency.pct_time'] / (df['count'] * df['attr.n_operations'])
    pws['per_op_latency'] = pws['latency.pct_time'] / pws['attr.n_operations']

    charts.append((
        df,
        "Per op latency by number of operations",
        'per_op_latency',
        'attr.n_operations'))

    def list_pw_operations(pws):
        for _, pw in pws.iterrows():
            if pw['attr.n_operations'] < 2:
                continue
            operations = "\n\t".join([op for op in pw['attr.operations']])
            print(f"{pw.name}\n\t{operations}")

    plotly_pie2("Pointwise Statistics", charts)
    list_pw_operations(pws)
    print(pws['per_op_latency'])


def layer_latency_sunburst(df: pd.DataFrame, title: str):
    precision_sunburst = df.groupby(['type', 'latency.pct_time']).count().reset_index()

    fig = px.sunburst(
        precision_sunburst,
        path=['type', 'latency.pct_time'],
        values='latency.avg_time',
        color_discrete_map=layer_colormap,
        color='type')
    fig.update_layout(title=title, title_x=0.5, font_size=15,)
    fig.show()


def plot_engine_timings(timing_json_file: str):
    """Plot the engine profiling timings"""
    latencies = read_timing_file(timing_json_file)
    samples = range(len(latencies))

    fig = px.scatter(
        title="Engine Timing Samples",
        x=samples, y=latencies)
    trex_base_layout(fig)
    fig.update_layout({
        'yaxis_title': "Latency (ms)",
        'xaxis_title': "Timing Samples",
        'title_x': 0.5})
    fig.show()


def report_card_gemm_MNK(plan: pd.DataFrame):
    def render_scatter3d(choice, x, y, z, color, size):
        convs = plan.get_layers_by_type('Convolution')
        fig = px.scatter_3d(convs, x=x, y=y, z=z, color=color, size=size,
                size_max=18, opacity=0.7)
        trex_base_layout(fig)
        fig.update_layout({
            'title': "Implicit GEMM " + choice,
            'title_x': 0.5})
        fig.show()

    dropdown_choices = {
        "MxNxK color=mean time; size=mean time":
            ('attr.M', 'attr.N', 'attr.K', 'latency.avg_time', 'latency.avg_time',),
        "MxNxK color=arithmetic intensity; size=mean time":
            ('attr.M', 'attr.N', 'attr.K', 'attr.arithmetic_intensity', 'latency.avg_time',),
        "MxNxK color=compute efficiency; size=mean time":
            ('attr.M', 'attr.N', 'attr.K', 'attr.compute_efficiency', 'latency.avg_time',),
        "MxNxK color=memory efficiency; size=mean time":
            ('attr.M', 'attr.N', 'attr.K', 'attr.memory_efficiency', 'latency.avg_time',),
    }

    InteractiveDiagram(render_scatter3d, dropdown_choices, 'Diagram')


def report_card_gemm_MNK_scatter(plan: pd.DataFrame):
    def render_scatter(choice, x, y, color, size):
        convs = plan.get_layers_by_type('Convolution')
        fig = px.scatter(convs, x=x, y=y, color=color, size=size, size_max=18, opacity=0.7)
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=50),
            title=choice, title_x=0.5)
        fig.show()

    dropdown_choices = {
        "M vs. Latency (color = foorprint)": (
            'attr.M', 'latency.avg_time', 'total_footprint_bytes', None),
        "N vs. Latency (color = foorprint)": (
            'attr.N', 'latency.avg_time', 'total_footprint_bytes', None),
        "K vs. Latency (color = foorprint)": (
            'attr.K', 'latency.avg_time', 'total_footprint_bytes', None),

    }

    InteractiveDiagram(render_scatter, dropdown_choices, 'Diagram')


def report_card_efficiency_vs_latency_3d(plan: pd.DataFrame):
    convs = plan.get_layers_by_type('Convolution')
    fig = px.scatter_3d(
        convs,
        x='attr.compute_efficiency', y='attr.memory_efficiency', z='latency.avg_time',
        color='total_footprint_bytes', size='latency.avg_time', size_max=18,
        opacity=0.7)

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        title="Compute-efficiency vs Memory-efficiency vs Latency",
        title_x=0.5)

    trex_base_layout(fig)
    fig.show()


def report_card_perf_scatter(plan: pd.DataFrame):
    def render_scatter(choice, x, y, color, size):
        convs = plan.get_layers_by_type('Convolution')
        fig = px.scatter(convs, x=x, y=y, color=color, size=size, size_max=18, opacity=0.7)
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=50),
            title=choice, title_x=0.5)
        fig.show()

    dropdown_choices = {
        "Compute-efficiency vs. FMAs (size = foorprint)": (
            'attr.compute_efficiency', 'attr.macs', 'latency.avg_time', 'total_footprint_bytes',),

        "Memory-efficiency vs. footprint (size = FMAs)": (
            'attr.memory_efficiency', 'total_footprint_bytes', 'latency.avg_time', 'attr.macs',),

        "Compute-efficiency vs. memory-efficiency (size = AI)": (
            'attr.compute_efficiency', 'attr.memory_efficiency', 'latency.avg_time', 'attr.arithmetic_intensity'),

        "Memory footprint vs FMAs (size = AI)": (
            'total_footprint_bytes', 'attr.macs', 'latency.avg_time', 'attr.arithmetic_intensity'),

        "Arithmetic-intensity vs. compute-efficiency (size = memory-efficiency)": (
            'attr.arithmetic_intensity', 'attr.compute_efficiency', 'latency.avg_time', 'attr.memory_efficiency'),

        "Arithmetic-intensity vs. memory-efficiency (size = compute-efficiency)": (
            'attr.arithmetic_intensity', 'attr.memory_efficiency', 'latency.avg_time', 'attr.compute_efficiency'),
    }

    InteractiveDiagram(render_scatter, dropdown_choices, 'Diagram')
