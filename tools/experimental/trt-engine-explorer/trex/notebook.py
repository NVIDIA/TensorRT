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
Miscellanous functions used in Jupyter notebooks
"""


from typing import Callable
import pandas as pd
import dtale
from IPython.display import display, HTML
import pandas as pd
import time
import logging


dtale.global_state.set_app_settings(dict(max_column_width=600)) # pixels


def section_header(title):
    style = "text-align:center;background:#76b900;padding:20px;color:#ffffff;font-size:2em;"
    return HTML('<div style="{}">{}</div>'.format(style, title))


def set_wide_display(width_pct: int=90):
    """Configure a wider rendering of the notebook
    (for easier viewing of wide tables and graphs).
    """
    display(HTML(f"<style>.container {{width:{width_pct}% !important;}}</style>"))


def display_df_dtale(
    df: pd.DataFrame,
    range_highlights: dict=None,
    nan_display: str='...',
    precision: int=4,
    **kwargs
):
    """Display a Pandas dataframe using a dtale widget"""

    start = time.time()
    d = dtale.show(
        df,
        drop_index=True,
        allow_cell_edits=False,
        precision=precision,
        nan_display=nan_display,
        **kwargs)

    if d and range_highlights is not None:
        d.update_settings(range_highlights=range_highlights, background_mode='range')

    # The dtale notebook client may fail to connect to the dtale (Flask) server
    # without an error indication. If the connection setup duration is longer than
    # some threshold (DTALE_TIMEOUT) we flag an error.
    DTALE_TIMEOUT = 30 # seconds
    abnormal_connection_duration = time.time() - start >= DTALE_TIMEOUT
    if not d or abnormal_connection_duration:
        logging.warning(
            "It seems like the host is taking too long to respond. Please try explicitly "
            "setting the Jupyter notebook server's IP address or name in "
            "\'dtale.app.ACTIVE_HOST = <host address>\'.\n"
            "Make sure to place this line at the top of your notebook.")
    else:
        display(d)


# Control how to display tables in notebooks
table_display_backend = display


def set_table_display_backend(tbl_display_fn: Callable):
    global table_display_backend
    table_display_backend = tbl_display_fn


def display_df(df: pd.DataFrame, **kwargs):
    try:
        table_display_backend(df, **kwargs)
    except TypeError:
        table_display_backend(df)

