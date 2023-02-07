#!/usr/bin/env python3
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
Simple printing utils

Utils to print traces and profiles in CSV format
"""


from __future__ import print_function


def combineDescriptions(prolog, features, descriptions):
    """Combine features with their descriptions"""

    fullDescription = prolog
    sep = " "
    for feature, description in zip(features, descriptions):
        fullDescription += sep + feature + " (" + description + ")"
        sep = ", "

    return fullDescription


def printHeader(allFeatures, selection, gp=False, count=False):
    """Print table header"""

    if gp:
        sep = "#"
        if count:
            sep += "count, "
    else:
        sep = ""

    for feature in allFeatures:
        if feature in selection:
            print(sep + feature, end="")
            sep = ", "

    print("")


def printCsv(data, count=False):
    """Print trace in CSV format"""

    c = 0
    for row in data:
        if count:
            print(c, end="")
            c += 1
            sep = ", "
        else:
            sep = ""
        for r in row:
            if isinstance(r, str):
                print(sep + r, end="")
            else:
                print("{}{:.6}".format(sep, float(r)), end="")
            sep = ", "
        print("")


def filterData(data, allFeatures, selection):
    """Drop features not in the given set"""

    filteredData = []
    for d in data:
        row = []
        for f in allFeatures:
            if f in selection:
                if f in d:
                    row.append(d[f])
                else:
                    row.append("")
        filteredData.append(row)

    return filteredData
