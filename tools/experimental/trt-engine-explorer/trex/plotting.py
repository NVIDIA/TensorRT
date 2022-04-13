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
This file contains pyplot plotting wrappers.
"""


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
from collections import defaultdict
import functools
import pandas as pd
import math
from .notebook import display_df
from .misc import stack_dataframes

NVDA_GREEN = '#76b900'
UNKNOWN_KEY_COLOR = 'gray'


# pallete = px.colors.qualitative.G10
# https://medialab.github.io/iwanthue/
default_pallete = [
    "#a11350",
    "#008619",
    "#4064ec",
    "#ffb519",
    "#8f1a8e",
    "#b2b200",
    "#64b0ff",
    "#e46d00",
    "#02d2ba",
    "#ef393d",
    "#f1b0f7",
    "#7e4401",
    UNKNOWN_KEY_COLOR]


# Set a color for each precision datatype.
precision_colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR, {
    'INT8':  NVDA_GREEN,
    'FP32':  'red',
    'FP16':  'orange',
    'INT32': 'lightgray',
})


# Set a color for each layer type.
layer_colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR, {
    # https://htmlcolorcodes.com/
    "Convolution":    "#2E86C1",
    "MatrixMultiply": "#1B4F72",
    "Scale":          "#a11350",
    "Shuffle":        "#FAD7A0",
    "Pooling":        "#008619",
    "Reformat":       "#F8C471",
    "SoftMax":        "#8f1a8e",
    "PluginV2":       "#b2b200",
    "PointWise":      "#1ABC9C",
    "ElementWise":    "#1ABC9C",
    "Myelin":         "#6C3483",
    "Reduce":         "#ABEBC6",
    "Slice":          "#7e4401",
})


def _categorical_colormap(df: pd.DataFrame, color_col:str):
    # Protect against index-out-of-range
    max_idx = len(default_pallete) - 1
    colormap = {category:
            default_pallete[min(i,max_idx)] for i,category in enumerate(set(df[color_col]))}
    return colormap


def _create_categorical_marker(df: pd.DataFrame, color_col: str, colormap: Dict=None):
    if not colormap:
        colormap = _categorical_colormap(df, color_col)
    # Make colormap robust to unknown keys
    colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR, colormap)
    color_list = [colormap[key] for key in df[color_col]]
    marker = dict(color=color_list)
    return marker


def create_layout(
    title: str,
    size:Tuple,
    x_title: str,
    y_title: str,
    orientation: str,
    show_axis_ticks: Tuple[bool]=(True,True)
):
    grid_color = 'rgba(114, 179, 24, 0.3)'
    y_grid = None if orientation == 'h' else grid_color
    x_grid = None if orientation == 'v' else grid_color

    if orientation == 'h':
        x_title, y_title = y_title, x_title

    layout = go.Layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'bottom'},
        width=size[0],
        height=size[1],
        xaxis={
            'visible': True,
            'showticklabels': show_axis_ticks[0],
            'title': x_title,
            'gridcolor': x_grid,},
        yaxis={
            'visible': True,
            'showticklabels': show_axis_ticks[1],
            'title': y_title,
            'gridcolor': y_grid,
            'tickformat': "%{y:$.2f}"},
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return layout


def plotly_bar2(
    df: pd.DataFrame,
    title: str,
    values_col: str,
    names_col: str,
    orientation: str='v',
    color: str=None,
    size: Tuple=(None, None),
    use_slider: bool=False,
    colormap: Dict=None,
    show_axis_ticks:Tuple=(False, True),
    showlegend: bool=False,
):
    def categorical_color(df, color) -> bool:
        if df[color].dtype in [float, int]:
            return False
        return True

    def add_bar(df, name, color, colormap, showlegend):
        if orientation == 'v':
            x, y = (names_col, values_col)
            hover_txt = f"{x}: " + "%{x}" + f"<br>{y}: " + "%{y:.1f}"
        else:
            x, y = (values_col, names_col)
            hover_txt = f"{x}: " + "%{x:.1f}" + f"<br>{y}: " + "%{y}"

        is_categorical = False
        colorbar = None
        if color is not None:
            assert isinstance(color, str)
            if df[color].dtype in [float, int]:
                # Continuous.
                colorbar = dict(title=color)
                color = df[color]
            else:
                is_categorical = True

        if not is_categorical:
            marker = dict(color=color, colorbar=colorbar)
        else:
            marker = _create_categorical_marker(df, color, colormap)

        texttemplate = "%{value:.2f}" if df[values_col].dtype == float else '%{value}'
        bar = go.Bar(
            x=df[x], y=df[y],
            orientation=orientation,
            marker=marker,
            text=df[values_col],
            texttemplate=texttemplate,
            hovertemplate=hover_txt,
            name=name,
            showlegend=showlegend)
        return bar

    layout = create_layout(
        title,
        size,
        x_title=names_col,
        y_title=values_col,
        orientation=orientation,
        show_axis_ticks=show_axis_ticks)
    fig = go.Figure(layout=layout)

    if not isinstance(df, Dict):
        df_dict = {None: df}
    else:
        df_dict = df

    for name, df in df_dict.items():
        bar = add_bar(df, name, color, colormap, showlegend=showlegend)
        fig.add_traces(bar)

    if use_slider:
        fig.update_xaxes(rangeslider_visible=True)
    fig.show()


def _create_bar(
    df: pd.DataFrame,
    names: str,
    values: str,
    color_col: str,
    orientation: str
) -> go.Bar:
    """Create a single bar diagram."""

    if orientation == 'v':
        x, y = (names, values)
        hover_txt = f"{x}: " + "%{x}: " + f"<br>{y}: " + "%{y:.1f}"
    else:
        x, y = (values, names)
        hover_txt = f"{x}: " + "%{x:.1f}: " + f"<br>{y}: " + "%{y}"

    is_categorical = False
    colorbar = None
    if color_col is None:
        color = None
    else:
        color = color_col
        assert isinstance(color, str)
        if df[color].dtype in [float, int]:
            # Continuous.
            colorbar = dict(title=color_col)
            color = df[color_col]
        else:
            is_categorical = True

    if is_categorical:
        marker = _create_categorical_marker(df, color)
    else:
        #marker = create_continuous_marker(df, color)
        marker = dict(color=color, colorbar=colorbar)

    bar = go.Bar(
            x=df[x], y=df[y],
            orientation=orientation,
            marker=marker,
            text=df[values],
            texttemplate = "%{value:.2f}" if df[values].dtype == float else '%{value}',
            textposition='outside',
            hovertemplate = hover_txt)
    return bar


def plotly_multi_bar(
    title: str,
    bar_sources: List[Tuple[pd.DataFrame, str]],
    names: str,
    orientation: str='v',
    color: str=None,
    size: Tuple[int, int]=(None, None),
    use_slider: bool=False,
    colormap=None,
    use_subplots: bool=False
):
    """Draw several bar diagrams."""

    if orientation == 'v':
        x_title, y_title = names, bar_sources[0][1]
    else:
        x_title, y_title = bar_sources[0][1], names

    if use_subplots:
        fig = make_subplots(rows=1, cols=len(bar_sources))
    else:
        fig = go.Figure()
    layout = create_layout(title, size, x_title, y_title, orientation=orientation)
    fig.update_layout(layout)

    for col, source in enumerate(bar_sources, start=1):
        df = source[0]
        values = source[1]
        bar = _create_bar(df, names, values, color, orientation)
        if use_subplots:
            if orientation == 'v':
                x_title, y_title = names, bar_sources[col-1][1]
            else:
                x_title, y_title = bar_sources[col-1][1], names
            fig.update_yaxes(title_text=y_title, row=1, col=col)
            fig.update_xaxes(title_text=x_title, row=1, col=col)
            fig.add_trace(bar, row=1, col=col)
        else:
            fig.add_trace(bar)

    fig.update_layout(showlegend=False)
    if use_slider:
        fig.update_xaxes(rangeslider_visible=True)
    fig.show()


def rotate_columns(df: pd.DataFrame):
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df


def stacked_tabular(
    bar_names: List[str],
    df_list: List[pd.DataFrame],
    names_col: str,
    values_col: str
):
    stacked = stack_dataframes(df_list, names_col, values_col)
    df = pd.DataFrame.from_dict(stacked, orient='index', columns=bar_names)
    df[values_col] = stacked.keys()
    df = rotate_columns(df)
    display_df(df)


def stacked_bars(
    title: str,
    bar_names: List[str],
    df_list: List[pd.DataFrame],
    names_col: str,
    values_col: str,
    empty_symbol: object=0,
    colormap: Dict=None,
    display_tbl: bool=True,
):
    """Stack a list of dataframes.
    Each df in df_list has a `names` column and a 'values` column.
    This function returns a dictionary indexed by each name in the
    set of all names from all dfs in df_list.
    For each `name` key the dictionary value is a list of all values from all dfs.
    If a df[name] does not exist, we create an `empty_symbol` entry for it.
    """

    stacked = stack_dataframes(df_list, names_col, values_col)
    if colormap:
        bars = [go.Bar(name=k, x=bar_names, y=v, marker_color=colormap[k], text=v) for k,v in stacked.items()]
    else:
        bars = [go.Bar(name=k, x=bar_names, y=v, text=v) for k,v in stacked.items()]
    layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(layout=layout, data=bars)
    fig.update_layout(title=title, title_x=0.5, font_size=15,)
    fig.update_layout(barmode='stack')
    fig.update_layout(showlegend=colormap is not None)
    fig.update_traces(texttemplate='%{text:.4f}')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()
    if display_tbl:
        stacked_tabular(bar_names, df_list, names_col, values_col)


def plotly_hist(
    df: pd.DataFrame,
    title: str,
    values_col: str,
    xaxis_title: str,
    color: str,
    colormap=None):
    """Draw a histogram diagram."""

    nvgreen = [NVDA_GREEN]*len(df)
    fig = px.histogram(
        df, x=values_col, title=title, nbins=len(df),
        histfunc="count",
        color=color,
        template="simple_white",
        color_discrete_map=colormap)
    fig.update_layout(xaxis={"title": xaxis_title}, bargap=0.05)
    fig.show()


def plotly_pie(
    df: pd.DataFrame,
    title: str,
    values: str,
    names: str,
    colormap: Dict[str,str]=None):
    """Draw a pie diagram."""

    fig = go.Figure(data=[go.Pie(
        labels=df[names], values=df[values], hole=.0)])

    pie_pallette = default_pallete
    if colormap:
        pie_pallette = [colormap[key] for key in df[names]]

    marker = dict(colors=pie_pallette, line=dict(color=NVDA_GREEN, width=1))
    fig.update_traces(marker=marker)
    fig.update_traces(
        hoverinfo='label+percent', textinfo='value', textfont_size=20,)
    fig.update_layout(title=title, title_x=0.5, font_size=20,)
    fig.show()


def plotly_pie2(
    title: str,
    charts: list,
    max_cols: int=3,
    colormap: Dict[str,str]=None):
    """Draw a pie diagram."""

    main_title = title
    n_charts = len(charts)
    n_cols = min(max_cols, n_charts)
    n_rows = math.ceil(n_charts/n_cols)
    specs = [[{'type':'domain'}] * n_cols] * n_rows

    subtitles = [chart[1] for chart in charts]
    fig = make_subplots(rows=n_rows, cols=n_cols, specs=specs,
                        subplot_titles=subtitles)

    pie_pallette = None
    for i, chart in enumerate(charts):
        df, title, values, names = chart
        row, col = i // n_cols, i % n_cols
        texttemplate = "%{value:.1f}" if df[values].dtype == float else '%{value}'
        if colormap is not None:
            pie_pallette = [colormap[key] for key in df[names]]
        marker = dict(colors=pie_pallette, line=dict(color=NVDA_GREEN, width=1))

        fig.add_trace(go.Pie(
            labels=df[names],
            values=df[values],
            name=title,
            marker=marker,
            texttemplate=texttemplate,), row+1, col+1)

    if pie_pallette is None:
        pie_pallette = default_pallete

    fig.update_traces(
        hoverinfo='label+percent',
        textinfo='value',
        textfont_size=20,
        hole=0.4,
        textposition='inside',)
    fig.update_layout(title=main_title, title_x=0.5, font_size=20,)
    fig.show()
