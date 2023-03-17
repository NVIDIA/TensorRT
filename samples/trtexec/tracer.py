#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Print a trtexec timing trace from a JSON file

Given a JSON file containing a trtexec timing trace,
this program prints the trace in CSV table format.
Each row represents an entry point in the trace.

The columns, as indicated by the header, respresent
one of the metric recorded. The output format can
be optionally converted to a format suitable for
GNUPlot.
"""

import sys
import json
import argparse
import prn_utils as pu


timestamps = ["startInMs", "endInMs", "startComputeMs", "endComputeMs", "startOutMs", "endOutMs"]

intervals = ["inMs", "computeMs", "outMs", "latencyMs", "endToEndMs"]

allMetrics = timestamps + intervals

defaultMetrics = ",".join(allMetrics)

descriptions = [
    "start input",
    "end input",
    "start compute",
    "end compute",
    "start output",
    "end output",
    "input",
    "compute",
    "output",
    "latency",
    "end to end latency",
]

metricsDescription = pu.combineDescriptions("Possible metrics (all in ms) are:", allMetrics, descriptions)


def skipTrace(trace, start):
    """Skip trace entries until start time"""

    for t in range(len(trace)):
        if trace[t]["startComputeMs"] >= start:
            return trace[t:]

    return []


def hasTimestamp(metrics):
    """Check if features have at least one timestamp"""

    for timestamp in timestamps:
        if timestamp in metrics:
            return True
    return False


def avgData(data, avg, times):
    """Average trace entries (every avg entries)"""

    averaged = []
    accumulator = []
    r = 0

    for row in data:
        if r == 0:
            for m in row:
                accumulator.append(m)
        else:
            for m in row[times:]:
                accumulator[t] += m

        r += 1
        if r == avg:
            for t in range(times, len(row)):
                accumulator[t] /= avg
            averaged.append(accumulator)
            accumulator = []
            r = 0

    return averaged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        metavar="M[,M]*",
        default=defaultMetrics,
        help="Comma separated list of metrics to print. " + metricsDescription,
    )
    parser.add_argument("--avg", metavar="N", type=int, default=1, help="Print average every N records.")
    parser.add_argument(
        "--start",
        metavar="T",
        type=float,
        default=0,
        help="Start trace at time T (drop records with compute start before T ms).",
    )
    parser.add_argument("--gp", action="store_true", help="Print GNUPlot format.")
    parser.add_argument("--no-header", action="store_true", help="Omit the header row.")
    parser.add_argument("name", metavar="filename", help="Trace file.")
    args = parser.parse_args()

    metrics = args.metrics.split(",")
    count = args.gp and (not hasTimestamp(metrics) or len(metrics) == 1)

    if not args.no_header:
        pu.printHeader(allMetrics, metrics, args.gp, count)

    with open(args.name) as f:
        trace = json.load(f)

    if args.start > 0:
        trace = skipTrace(trace, args.start)

    trace = pu.filterData(trace, allMetrics, metrics)

    if args.avg > 1:
        trace = avgData(trace, args.avg, hasTimestamp(metrics))

    pu.printCsv(trace, count)


if __name__ == "__main__":
    sys.exit(main())
