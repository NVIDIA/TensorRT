#!/usr/bin/env python3
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

import trex.engine_plan
from trex import group_count, group_sum_attr, group_mean_attr
from tabulate import tabulate
from typing import Literal


def summarize_engine(engine_json_fname: str,
    profiling_json_fname: str,
    group_tactics: bool = False,
    sort_key: str | Literal['count', 'latency', 'id'] = 'count',
):
    plan = trex.engine_plan.EnginePlan(engine_json_fname, profiling_file=profiling_json_fname)

    if group_tactics:
        # The hash distinguishes individual tactics
        remove_hash = lambda tactic: tactic[: tactic.find("_0x")] if tactic.find("_0x") > 0 else tactic
        plan.df['tactic'] = plan.df['tactic'].apply(remove_hash)

        tactic_cnt = group_count(plan.df, 'tactic')
        tactic_latency = group_sum_attr(plan.df, 'tactic', 'latency.pct_time')
        df = tactic_latency
        df['count'] = tactic_cnt['count']
        df.reset_index(drop=True, inplace=True)
        if sort_key == 'id':
            sort_key = 'count'

    else:
        tactic_mean_latency = group_mean_attr(plan.df, 'tactic', 'latency.pct_time')
        df = tactic_mean_latency
        if sort_key == 'count':
            sort_key = 'latency'

    if sort_key == 'latency':
        sort_key = 'latency.pct_time'
    if sort_key != 'id':
        df = df.sort_values(by=[sort_key], ascending=False)
    df = df.rename(columns={'latency.pct_time': 'latency %'})
    print(tabulate(df, headers='keys', tablefmt='psql'))


def make_subcmd_parser(subparsers):
    summarize = lambda args: summarize_engine(
        engine_json_fname=args.input,
        profiling_json_fname=args.profiling_json,
        group_tactics=args.group_tactics,
        sort_key=args.sort_key,
    )
    summarize_parser = subparsers.add_parser("summary", help="Summarize a TensorRT engine.")
    summarize_parser.set_defaults(func=summarize)
    _make_parser(summarize_parser)


def _make_parser(parser):
    parser.add_argument("input", help="name of engine JSON file to draw.")
    parser.add_argument("--profiling_json", "-pj",
        default=None, help="name of engine JSON file to draw")
    parser.add_argument("--sort_key", "-sk",
        choices=["count", "latency", "id"],
        default="count", help="the key to use for sorting the tactics"),
    parser.add_argument("--group_tactics", "-gt",
        action='store_true',
        default=False, help="group the tactics by type")
