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
This file contains code to generate an Excel summary spreadsheet.
"""

import pandas as pd
import tempfile
import plotly
import xlsxwriter
from trex import EnginePlan
from trex.report_card import report_card_perf_overview, report_card_memory_footprint
from typing import Dict
import logging

MAX_PERMISSIBLE_NAME_LENGTH = 31

class ExcelSummary:
    def __init__(self, plan: EnginePlan, path: str="engine_summary.xlsx"):
        self.plan = plan
        self.path = path
        self.Excelwriter = pd.ExcelWriter(path, engine="xlsxwriter")

        # Report generation is currently only supported for those reports
        # that provide dropdown_choices in `report_card.py`
        self.supported_reports = {
            'report_card_perf_overview': report_card_perf_overview,
            'report_card_memory_footprint': report_card_memory_footprint,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Allow context manager to automatically call `save`
        """
        self.save()

    def _add_df(self, df: pd.DataFrame, sheet_name: str):
        """
        Adds a single dataframe to the Excel file. This function is not
        exposed to the user. Users may use "add_dataframes" instead
        """

        df.to_excel(self.Excelwriter, sheet_name=sheet_name, index=False, startrow=3, startcol=1)

    def add_dataframes(self, df_dict: Dict[str, pd.DataFrame]):
        """
        Adds multiple dataframes to the Excel file
        """

        for sheet_name, df in df_dict.items():
            self._add_df(df, sheet_name)

    def _add_image(self, image: str, sheet_name: str):
        """
        Adds a single image to the Excel file. This function is not
        exposed to the user. Users may use "add_images" instead
        """

        worksheet = self.Excelwriter.book.add_worksheet(sheet_name)
        worksheet.insert_image('A1', image)

    def add_images(self, image_dict):
        """
        Adds multiple images to the Excel file
        """

        for sheet_name, image_fname in image_dict.items():
            self._add_image(image_fname, sheet_name)

    def _add_report(self, report_name: str):
        """
        Adds a report to the Excel file
        """

        if report_name not in self.supported_reports:
            raise ValueError(f"{report_name} is not a supported report: {list(self.supported_reports)}")

        files = []
        workbook  = self.Excelwriter.book

        report_choices = self.supported_reports[report_name](self.plan)
        for stat_name in report_choices.keys():
            # Excel sheets have a constraint on length of sheet name
            worksheet = workbook.add_worksheet(stat_name[:MAX_PERMISSIBLE_NAME_LENGTH])
            figure = report_choices[stat_name](title=stat_name, do_show=False)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as fp:
                files.append(fp)
                # Currently only supports generating images for plotly figures
                if isinstance(figure, (plotly.graph_objs._figure.Figure)):
                    figure.write_image(fp)
                    worksheet.insert_image('C5', fp.name)

        for fp in files:
            fp.close()

    def generate_default_summary(self, save: bool=True):
        """
        Generate a default summary with all the available data
        """

        self.add_dataframes({"Plan":self.plan.df})
        reports_to_call = [
            'report_card_perf_overview',
            'report_card_memory_footprint'
        ]

        for report in reports_to_call:
            self._add_report(report)

        if save:
            self.save()

    def save(self):
        """
        Saves the Excel file.

        Once the file is saved, it cannot be updated using the same instance
        of the class. A new instance may have to be created to overwrite the
        Excel file
        """

        self.Excelwriter.close()
        print(f"{self.path} has been saved")
