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
This file contains interaction cells for the compare_engines.ipynb notebook.
"""


import copy
import numpy as np
from typing import List, Tuple, Dict
from functools import partial
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trex.misc as misc
import trex.plotting as plotting
import trex.colors as colors
import trex.notebook as notebook
import trex.interactive as interactive
import trex.engine_plan as engine_plan
import trex.activations as activations


def get_plans_names(plans: List[engine_plan.EnginePlan]):
    """Create unique plans names"""
    engine_names = [plan.name for plan in plans]
    if len(set(engine_names)) != len(plans):
        engine_names = [plan.name + str(i) for i,plan in enumerate(plans)]
    return engine_names


def compare_engines_overview(plans: List[engine_plan.EnginePlan]):
    """A dropdown widget to choose from several diagrams
    that compare 2 or more engine plans.
    """
    engine_names = get_plans_names(plans)

    # Get throughtput data.
    throughtput = [plan.performance_summary.get('Throughput', 0) for plan in plans]
    have_throughput_data = all([tp > 0 for tp in throughtput])

    def throughput_per_plan(title: str):
        y = [plan.performance_summary.get('Throughput', 0) for plan in plans]
        x = [plan.name for plan in plans]
        fig = px.bar(x=y, y=x, orientation='h')
        plotting.trex_base_layout(fig)
        fig.update_layout({
            'xaxis_title': "Throughput (inferences / sec)",
            'yaxis_title': "Engine"})
        fig.show()

    time_by_type = [plan.df.groupby(['type'])[
        ['latency.pct_time', 'latency.avg_time']].sum().reset_index() for plan in plans]

    cnt_by_type = [misc.group_count(plan.df, 'type') for plan in plans]

    # Normalize timings by the batch-size.
    df_list_bs_normalized = [copy.deepcopy(plan.df) for plan in plans]
    for i, plan in enumerate(plans):
        inputs, outputs = plan.get_bindings()
        bs = inputs[0].shape[0]
        df_list_bs_normalized[i]['latency.avg_time'] /= bs

    time_by_type_bs_normalized = [df.groupby(['type'])
        [['latency.pct_time', 'latency.avg_time']].sum().reset_index() for df in df_list_bs_normalized]

    def latency_per_type(title):
        stacked_latencies_bars = partial(
            plotting.stacked_bars,
            title,
            bar_names=engine_names,
            df_list=time_by_type,
            names_col='type',
            values_col='latency.avg_time',
            colormap=colors.layer_colormap,
            display_tbl=False,
            xaxis_title="Engine",
            yaxis_title="Latency (ms)")

        if have_throughput_data:
            real_latency = [plan.performance_summary.get('Latency', [0]*5)[2] for plan in plans]

            # Display throughput scatter plot together with the latencies bars.
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            stacked_latencies_bars(fig=fig)
            fig.add_trace(
                go.Scatter(
                    x=engine_names,
                    y=throughtput,
                    name="Throughput (IPS)",
                    marker=dict(size=12, color='#FFBF00'),),
                secondary_y=True)
            fig.add_trace(
                go.Scatter(
                    x=engine_names,
                    y=real_latency,
                    name="Real Latency",
                    marker=dict(size=12, color='#DE3163'),),
                secondary_y=False)
            plotting.trex_base_layout(fig)
            fig.update_yaxes(title_text="Throughput (inferences / sec)", secondary_y=True)
            fig.show()
        else:
            stacked_latencies_bars()

        df = plotting.stacked_tabular_df(
            engine_names, time_by_type, 'type', 'latency.avg_time', empty_symbol=np.NaN)
        # Compute the speedup of the last engine vs. the first engine.
        df['speedup'] = df[engine_names[0]] / df[engine_names[-1]]
        print(f"\'speedup\' refers to the speedup of \"{engine_names[-1]}\" relative to \"{engine_names[0]}\"")
        notebook.display_df(df, range_highlights=speedup_range_highlights(
            col_name='speedup', threshold=0.03))

    latency_per_type_bs_normalized = partial(
        plotting.stacked_bars,
        bar_names=engine_names,
        df_list=time_by_type_bs_normalized,
        names_col='type',
        values_col='latency.avg_time',
        empty_symbol=np.NaN,
        colormap=colors.layer_colormap,
        xaxis_title="Engine",
        yaxis_title="Latency (ms)")

    d = {engine_name:df for engine_name, df in zip(engine_names, time_by_type)}
    latency_per_type_comparison = partial(
        plotting.plotly_bar2,
        df=d,
        values_col='latency.avg_time',
        names_col='type',
        orientation='h',
        showlegend=True)

    d = {engine_name:df for engine_name, df in zip(engine_names, cnt_by_type)}
    count_comparison = partial(
        plotting.plotly_bar2,
        df=d,
        values_col='count',
        names_col='type',
        orientation='h',
        showlegend=True)

    time_by_precision = [plan.df.groupby(['precision']) \
        [['latency.avg_time']].sum().reset_index() for plan in plans]

    stacked_layers_by_precision = partial(
        plotting.stacked_bars,
        bar_names=engine_names,
        df_list=time_by_precision,
        names_col='precision',
        values_col='latency.avg_time',
        colormap=colors.precision_colormap)

    precision_subplots = [(
        misc.group_count(
            plan.df, 'precision'),
            plan.name, 'count', 'precision'
        ) for plan in plans]
    precision_cnts = partial(
        plotting.plotly_pie2,
        charts=precision_subplots,
        colormap=colors.precision_colormap)

    output_precision_subplots = [(
        misc.group_count(
            plan.df, 'output_precision'),
            plan.name, 'count', 'output_precision'
        ) for plan in plans]
    output_precision_cnts = partial(
        plotting.plotly_pie2,
        charts=output_precision_subplots,
        colormap=colors.precision_colormap)

    precision_subplots = [(
        misc.group_sum_attr(
            plan.df, grouping_attr='precision',
            reduced_attr='latency.pct_time'),
            plan.name, 'latency.pct_time', 'precision'
        ) for plan in plans]
    precision_latency = partial(
        plotting.plotly_pie2,
        charts=precision_subplots,
        colormap=colors.precision_colormap)

    dropdown_choices = {
        "Stacked latencies by layer type": latency_per_type,
        "Stacked latencies by layer type (BS normalized)": latency_per_type_bs_normalized,
        "Layer count comparison": count_comparison,
        "Layer-type latency comparison": latency_per_type_comparison,
        "Layer latency by precision": stacked_layers_by_precision,
        "Precision counts": precision_cnts,
        "Output precision counts": output_precision_cnts,
        "Precision latency": precision_latency,
    }
    if have_throughput_data:
        dropdown_choices["Throughput"] = throughput_per_plan

    interactive.InteractiveDiagram_2(dropdown_choices, 'Diagram:')


def compare_engines_summaries_tbl(plans: List[engine_plan.EnginePlan], orientation: str='vertical'):
    """Display a tabular comparison of several engine plans."""

    merged_summaries = {}
    summary_dicts_list = (
        [engine_plan.summary_dict(plan) for plan in plans],
        [plan.performance_summary for plan in plans],
        [plan.device_properties for plan in plans],
        [plan.builder_cfg for plan in plans]
    )

    for d in summary_dicts_list:
        merged_summaries.update(misc.stack_dicts(d, empty_placeholder=""))

    if orientation == 'vertical':
        df = pd.DataFrame.from_dict(
            merged_summaries, orient='index', columns=get_plans_names(plans))
        df['attribute'] = list(merged_summaries.keys())
        df = plotting.rotate_columns(df)
        df.set_index('attribute')
    else:
        df = pd.DataFrame.from_dict(merged_summaries)
        df['plan'] = get_plans_names(plans)
        df = plotting.rotate_columns(df)
    print(("\"Average time\": "
        "refers to the sum of the layer latencies, when profiling layers separately"))
    print(("\"Latency\": "
        "refers to the [min, max, mean, median, 99% percentile] of the engine latency "
        "measurements, when timing the engine w/o profiling layers."))
    notebook.display_df(df)


# Code to align and compare two plans

def get_io_dimensions(layer: pd.Series, use_all_tensors: bool) -> tuple:
    """Return a tuple containing all the dimensions of layer's
    inputs and outputs.

    The first dimension (batch) of each input/output tensor is not included
    so that the batch-size is not a cause for a mismatch.

    For an exact (conservative) matching set `use_all_tensors` to True.
    To match using only the first input and output, set to False.
    """
    inputs, outputs = activations.create_activations(layer)
    if not use_all_tensors:
        inputs = [inputs[0],]
        outputs = [outputs[0],]

    dims_dict = {
        'inputs': [t.shape[1:] for t in inputs],
        'outputs': [t.shape[1:] for t in outputs]
    }
    return dims_dict

def get_io_formats(layer: pd.Series) -> tuple:
    """Return a string representation of the inputs and outputs."""
    inputs, outputs = activations.create_activations(layer)
    in_formats = "in: " + ", ".join((f"{t.format}:{t.shape}" for t in inputs))
    out_formats = "out: " + ", ".join((f"{t.format}:{t.shape}" for t in outputs))
    return in_formats + "\t" + out_formats


def get_io_precisions(layer: pd.Series) -> tuple:
    """Return two tuples representing the precisions of layer's inputs and outputs."""
    if layer is None:
        return "", ""
    inputs, outputs = activations.create_activations(layer)
    assert len(inputs) > 0 and len(outputs) > 0
    p_in = ", ".join((t.precision for t in inputs))
    p_out = ", ".join((t.precision for t in outputs))
    return p_in, p_out


def match_layers(
    plan1: engine_plan.EnginePlan, plan2: engine_plan.EnginePlan, exact_matching: bool
) -> List[Tuple]:
    """Align two plans by their layers.

    When comparing the layers of two engine plans, we want to find pairs of layers,
    one from each plan, that correspond to the same trt.Network layer. This is
    not trivial since the plans may have a different number of layers, the layers
    may be fused differently, and they may have different names.

    A heuristic assigns a `signature` to each layer in the two plans. To determine
    if two layers correspond to the same trt.Network layer, their signatures are
    compared.

    Aligining two plans is the task of finding pairs of layers with the same
    signature.
    This function returns a list of index pairs.
    """
    def signature(layer: pd.Series, exact: bool) -> Dict:
        """Returns the heuristic layer signature.

        The signature is composed of the layer's type and dimensions.
        """
        sig = get_io_dimensions(layer, exact)
        sig['type'] = layer['type']
        return sig


    def clamp_indexes(i1: int, i2: int) -> Tuple:
        i1 = min(i1, len(plan1.df) - 1)
        i2 = min(i2, len(plan2.df) - 1)
        return i1, i2

    def are_equal(s1: Dict, s2: Dict) -> bool:
        assert list(s1.keys()) == list(s2.keys()), "Internal error: signature are corrupt"
        for k in s1.keys():
            if s1[k] != s2[k]:
                return False
        return True

    def is_aligned(i1: int, i2: int, exact_matching: bool) -> bool:
        """Return True if row `i1` of plan1 is aligned
        with row `i2` of plan2. """
        def pointwise_same(s1: Dict, s2: Dict):
            """Special signatures comparison for pointwise layers.

            When comparing PointWise layers allow the inputs to be connected in
            reverse order."""
            same = False
            types_ok = s1['type'] == s2['type'] == "PointWise"
            in_lengths_ok = len(s1['inputs']) == 2 and len(s2['inputs']) == 2
            out_lengths_ok = len(s1['outputs']) == 1 and len(s2['outputs']) == 1
            if types_ok and in_lengths_ok and out_lengths_ok:
                same = s1['inputs'][0] == s2['inputs'][1] and s1['inputs'][1] == s2['inputs'][0]
            return same

        i1, i2 = clamp_indexes(i1, i2)
        s1 = signature(plan1.df.loc[i1], exact_matching)
        s2 = signature(plan2.df.loc[i2], exact_matching)
        aligned = are_equal(s1, s2)
        if not aligned:
            aligned = pointwise_same(s1, s2)
        return aligned

    def beam_search(beam_size, unprocessed_indices, list_id):
        """Shine a search beam and look for a match in the other list.
        """
        i1 = unprocessed_indices[0][0]
        i2 = unprocessed_indices[1][0]
        for s in range(beam_size):
            # clamp
            idx = min(s, len(unprocessed_indices[list_id]) - 1)
            if list_id == 1:
                i2 = unprocessed_indices[list_id][idx]
            else:
                i1 = unprocessed_indices[list_id][idx]
            if is_aligned(i1, i2, exact_matching):
                return i1, i2
        if list_id == 1:
            return i1, None
        else:
            return None, i2

    def debug_print(i1: int, i2: int):
        return # disable print
        t1 = plan1.df.loc[i1]['type'] if i1 is not None else "None"
        t2 = plan2.df.loc[i2]['type'] if i2 is not None else "None"
        print(f"{i1}: {t1}  {i2}: {t2}")

    matched_indices_pairs = []
    unprocessed_indices_1 = [*range(len(plan1.df))]
    unprocessed_indices_2 = [*range(len(plan2.df))]
    while unprocessed_indices_1 and unprocessed_indices_2:
        beam_size = max(len(unprocessed_indices_1), len(unprocessed_indices_2))
        for list_id in (1, 0):
            i1, i2 = beam_search(beam_size,
                (unprocessed_indices_1, unprocessed_indices_2), list_id)
            debug_print(i1, i2)
            matched_indices_pairs.append((i1, i2))
            if i1 is not None:
                unprocessed_indices_1.remove(i1)
            if i2 is not None:
                unprocessed_indices_2.remove(i2)
            if not unprocessed_indices_1 or not unprocessed_indices_2:
                break

    # Process "left-over" layers
    for i1 in unprocessed_indices_1:
        matched_indices_pairs.append((i1, None))
    for i2 in unprocessed_indices_2:
        matched_indices_pairs.append((None, i2))
    return matched_indices_pairs


def aligned_merge_plans(
    plan1: engine_plan.EnginePlan,
    plan2: engine_plan.EnginePlan,
    matched_indices_pairs: List[Tuple]
) -> pd.DataFrame:
    """Return a dataframe containing merged layers from the two plans, after
    their layers have been aligned.
    """
    def append_layer(merged: List, layer1: pd.Series, layer2: pd.Series):
        p1_in, p1_out = get_io_precisions(layer1)
        p2_in, p2_out = get_io_precisions(layer2)
        merged.append((
            layer1['type'] if layer1 is not None else layer2['type'],
            layer1['latency.avg_time'] if layer1 is not None else 0,
            layer2['latency.avg_time'] if layer2 is not None else 0,
            layer1['latency.avg_time']/layer2['latency.avg_time'] if layer1 is not None and layer2 is not None else np.NaN,
            p1_in,
            p2_in,
            p1_out,
            p2_out,
            layer1['tactic'] if layer1 is not None else "",
            layer2['tactic'] if layer2 is not None else "",
            get_io_formats(layer1) if layer1 is not None else "",
            get_io_formats(layer2) if layer2 is not None else "",
            layer1['Name'] if layer1 is not None else "",
            layer2['Name'] if layer2 is not None else "",
        ))

    merged = []
    for pair in matched_indices_pairs:
        # A pair of matched indices
        m1 = pair[0]
        m2 = pair[1]

        # Add the pair of matched layers.
        layer1 = plan1.df.loc[m1] if m1 is not None else None
        layer2 = plan2.df.loc[m2] if m2 is not None else None
        append_layer(merged, layer1, layer2)

    df = pd.DataFrame(merged, columns=(
        'type','avg_time (1)', 'avg_time (2)', 'speedup (2)',
        'in-p (1)', 'in-p (2)', 'out-p (1)', 'out-p (2)',
        'tactic (1)', 'tactic (2)',
        'formats (1)', 'formats (2)',
        plan1.name, plan2.name))
    return df


def aligned_layers(
    plan1: engine_plan.EnginePlan,
    plan2: engine_plan.EnginePlan,
    matched_indices_pairs:List[Tuple],
    layer_type: str=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the dataframes of the two plans, after their layers
    have been aligned.

    Where the two plans do not align, insert space-holder rows.
    """
    def append_layer(layers_tbl, layer):
        if layer is None:
            # Add a "space-holder"
            layers_tbl.append((len(layers_tbl), "", "", 0))
        else:
            layers_tbl.append((
                len(layers_tbl),
                layer['Name'],
                layer['type'],
                layer['latency.avg_time']))

    # Create a table of layers for each engine.
    # Empty rows are inserted to an engine's table as space-holders when there's
    # no matching layer in the other engine.
    layers_tbl1, layers_tbl2 = [], []

    def filer_layer(layer, layer_type):
        ignore = layer is not None and (layer_type is None or layer['type'] == layer_type)
        return ignore

    for pair in matched_indices_pairs:
        # A pair of matched indices
        m1, m2  = pair[0], pair[1]

        # Add the pair of matched layers.
        layer1 = plan1.df.loc[m1] if m1 is not None else None
        layer2 = plan2.df.loc[m2] if m2 is not None else None

        if layer1 is not None and layer2 is not None:
            if (layer_type is None or layer1['type'] == layer_type):
                append_layer(layers_tbl1, layer1)
                append_layer(layers_tbl2, layer2)
        else:
            if filer_layer(layer1, layer_type):
                append_layer(layers_tbl2, None)
                append_layer(layers_tbl1, layer1)
            if filer_layer(layer2, layer_type):
                append_layer(layers_tbl1, None)
                append_layer(layers_tbl2, layer2)

    df1 = pd.DataFrame(layers_tbl1, columns=('id', 'name', 'type', 'latency.avg_time'))
    df2 = pd.DataFrame(layers_tbl2, columns=('id', 'name', 'type', 'latency.avg_time'))
    return df1, df2


def speedup_range_highlights(col_name, threshold: float):
    light_yellow = {'r': 255, 'g': 245, 'b': 157, 'a': 0}
    green = {'r': 0, 'g': 255, 'b': 0, 'a': 1}
    orange = {'r': 245, 'g': 166, 'b': 35, 'a': 1}
    range_highlights = {
          col_name: {
            'active': True,
            'equals': {'active': True, 'value': 1, 'color': light_yellow},
            'greaterThan': {'active': True, 'value': 1. + threshold, 'color': green},
            'lessThan': {'active': True, 'value': 1. - threshold, 'color': orange},
          }
        }
    return range_highlights


def compare_engines_layer_latencies(
    plan1: engine_plan.EnginePlan,
    plan2: engine_plan.EnginePlan,
    threshold: float,
    exact_matching: bool
):
    """Display a table and a bar diagram comparing the latencies of layers from
    two plans.
    """
    def render_diagram(choice: str, ignore):
        if plan1.name == plan2.name:
            plan1.name += ".0"
            plan2.name += ".1"

        matched_indices_pairs = match_layers(plan1, plan2, exact_matching)

        # Display a table comparison
        df = aligned_merge_plans(plan1, plan2, matched_indices_pairs)
        if choice not in ('All', 'Precision Mismatch', 'Tactic Mismatch'):
            df = df.query(f"type == \"{choice}\"")
        if choice  == 'Precision Mismatch':
            df = df[(df['in-p (1)'] != df['in-p (2)']) | (df['out-p (1)'] != df['out-p (2)'])]
        if choice == 'Tactic Mismatch':
            df = df[(df['tactic (1)'] != df['tactic (2)'])]
        print(f"Legend:\n\t1: {plan1.name}\n\t2: {plan2.name}")
        print("\"in-p (1)\" are the input precisions of the layer in "
             f"{plan1.name}. Similarly,")
        print("\"out-p (2)\" are the output precisions of the layer in " + plan2.name)
        notebook.display_df(df, range_highlights=speedup_range_highlights(
            'speedup (2)', threshold))

        # Display a bar diagram comparison
        layer_type = None if choice=='All' else choice
        df1, df2 = aligned_layers(plan1, plan2, matched_indices_pairs, layer_type)

        latency_str = lambda name, df: f"\n\t{name}: {df['latency.avg_time'].sum():.3f} ms"
        print(f"Latencies:{latency_str(plan1.name, df1)}{latency_str(plan2.name, df2)}")

        d = {plan1.name: df1, plan2.name: df2}
        plotting.plotly_bar2(title=f"Layer Latency Comparison (Layer Type={choice})",
            df=d,
            values_col='latency.avg_time',
            names_col='id',
            orientation='v',
            showlegend=True)

    types = ['All', 'Precision Mismatch', 'Tactic Mismatch']
    types += list(set(plan1.df['type'].tolist() + plan2.df['type'].tolist()))
    dropdown_choices = {t: t for t in types}
    interactive.InteractiveDiagram(render_diagram, dropdown_choices, 'Dataframe')


def compare_engines_layer_details(
    plan1: engine_plan.EnginePlan,
    plan2: engine_plan.EnginePlan,
):
    """Compare the details of two layers from two aligned plans.

    Align plan1 and plan2 and display a drop-down list of the layers
    of the combined dataframe. This allows easier comparison of
    individual layers.
    """
    def render_diagram(choice: str, row_id: int):
        if plan1.name == plan2.name:
            plan1.name += ".0"
            plan2.name += ".1"
        row = df.iloc[row_id]
        d = {
            'name': (row[plan1.name], row[plan2.name]),
            'avg_time': (row['avg_time (1)'], row['avg_time (2)']),
            'tactic': (row['tactic (1)'], row['tactic (2)']),
            'in-p': (row['in-p (1)'], row['in-p (2)']),
            'out-p': (row['out-p (1)'], row['out-p (2)']),
            'format': (row['formats (1)'], row['formats (2)']),
        }

        df2 = pd.DataFrame.from_dict(
            d, orient='index', columns=(plan1.name, plan2.name,))
        speedup = row['avg_time (1)'] / row['avg_time (2)']
        tactic = "Same" if row['tactic (1)'] == row['tactic (2)'] else "Different"
        inp_precision = "Same" if row['in-p (1)'] == row['in-p (2)'] else "Different"
        out_precision = "Same" if row['out-p (1)'] == row['out-p (2)'] else "Different"
        formats = "Same" if row['formats (1)'] == row['formats (2)'] else "Different"
        df2['comparison'] = (
            '', speedup, tactic, inp_precision, out_precision, formats)
        df2 = plotting.rotate_columns(df2)
        df2['attribute'] = (
            'name', 'avg_time', 'tactic', 'input precision', 'output precision', 'formats')
        df2 = plotting.rotate_columns(df2)
        df2.set_index('attribute')
        notebook.display_df(df2)

    matched_indices_pairs = match_layers(plan1, plan2, exact_matching=True)
    df = aligned_merge_plans(plan1, plan2, matched_indices_pairs)
    dropdown_choices = {f"{t}: {df.iloc[t]['type']}": t for t in range(len(df))}
    interactive.InteractiveDiagram(render_diagram, dropdown_choices, 'Choose Layer:')
