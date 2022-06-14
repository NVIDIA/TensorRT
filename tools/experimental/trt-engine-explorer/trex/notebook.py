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
Miscellanous functions used in Jupyter notebooks
"""


import pandas as pd
import dtale
from IPython.core.display import display, HTML
from ipyfilechooser import FileChooser
import qgrid
import pandas as pd

dtale.global_state.set_app_settings(dict(max_column_width=600)) # pixels


def section_header(title):
    style = "text-align:center;background:#76b900;padding:20px;color:#ffffff;font-size:2em;"
    return HTML('<div style="{}">{}</div>'.format(style, title))


def set_wide_display(width_pct: int=90):
    """Configure a wider rendering of the notebook
    (for easier viewing of wide tables and graphs).
    """
    display(HTML(f"<style>.container {{width:{width_pct}% !important;}}</style>"))


def display_df_qgrid(df: pd.DataFrame):
    """Display a Pandas dataframe using a qgrid widget"""
    grid = qgrid.show_grid(df,
        grid_options={'forceFitColumns': False, 'fullWidthRows': True},
        column_options={'resizable': True, },
        column_definitions={'index': {'maxWidth': 0, 'minWidth': 0, 'width': 0},
                            'Name': {'maxwidth': 400}})
    display(grid)


def display_df_dtale(
    df: pd.DataFrame,
    range_highlights: dict=None,
    nan_display: str='...',
    precision: int=4):
    """Display a Pandas dataframe using a dtale widget"""
    d = dtale.show(
        df,
        drop_index=True,
        allow_cell_edits=False,
        precision=precision,
        nan_display=nan_display)
    if range_highlights is not None:
        d.update_settings(range_highlights=range_highlights, background_mode='range')
    display(d)


def display_df(df: pd.DataFrame, **kwargs):
    display_df_dtale(df, **kwargs)


def display_filechooser(rootdir: str) -> FileChooser:
    """Create and display a FileChooser widget"""
    fc = FileChooser(rootdir)
    fc.filter_pattern = '*.engine'
    fc.title = 'Press Select to choose an engine file'
    display(fc)
    return fc
