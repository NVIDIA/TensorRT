#!/usr/bin/env python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

'''
Print a trtexec timing trace from a JSON file

Given a JSON file containing a trtexec timing trace,
this program prints the trace in CSV table format.
Each row represents an entry point in the trace.

The columns, as indicated by the header, respresent
one of the metric recorded. The output format can
be optionally converted to a format suitable for
GNUPlot.
'''

import sys
import json
import argparse
import prn_utils


timestamps = ['startInMs', 'endInMs', 'startComputeMs', 'endComputeMs', 'startOutMs', 'endOutMs']

intervals = ['inMs', 'computeMs', 'outMs', 'latencyMs', 'endToEndMs']

all_metrics = timestamps + intervals

default_metrics = ",".join(all_metrics)

descriptions = ['start input', 'end input', 'start compute', 'end compute', 'start output',
                'end output', 'input', 'compute', 'output', 'latency', 'end to end latency']

metrics_description = prn_utils.combine_descriptions('Possible metrics (all in ms) are:',
                                                     all_metrics, descriptions)



def skip_trace(trace, start):
    ''' Skip trace entries until start time '''

    trailing = []

    for t in trace:
        if t['start compute'] >= start:
            trailing.append(t)

    return trailing



def hasTimestamp(metric_set):
    ''' Check if features have at least one timestamp '''

    for timestamp in timestamps:
        if timestamp in metric_set:
            return True
    return False;



def avg_data(data, avg, times):
    ''' Average trace entries (every avg entries) '''

    averaged = []
    accumulator = []
    r = 0

    for row in data:
        if r == 0:
            for t in range(len(row)):
                accumulator.append(row[t])
        else:
            for t in range(times, len(row)):
                accumulator[t] += row[t]

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
    parser.add_argument('--metrics', metavar='M[,M]*', default=default_metrics,
                        help='Comma separated list of metrics to print. ' + metrics_description)
    parser.add_argument('--avg', metavar='N', type=int, default=1, help='Print average every N records.')
    parser.add_argument('--start', metavar='T', type=float, default=0, help='Start trace at time T (drop records with compute start before T).')
    parser.add_argument('--gp', action='store_true', help='Print GNUPlot format.')
    parser.add_argument('--no-header', action='store_true', help='Omit the header row.')
    parser.add_argument('name', metavar='filename', help='Trace file.')
    args = parser.parse_args()

    metric_set = args.metrics.split(',')
    count = args.gp and ( not hasTimestamp(metric_set) or len(metric_set) == 1)

    if not args.no_header:
        prn_utils.print_header(all_metrics, metric_set, args.gp, count)

    with open(args.name) as f:
        trace = json.load(f)

        if args.start > 0:
            trace = skip_trace(trace, args.start)

        data = prn_utils.filter_data(trace, all_metrics, metric_set)

        if args.avg > 1:
            data = avg_data(data, args.avg, hasTimestamp(metric_set))

        prn_utils.print_csv(data, count)


if __name__ == '__main__':
    sys.exit(main())
