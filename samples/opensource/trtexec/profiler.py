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
Print a trtexec profile from a JSON file

Given a JSON file containing a trtexec profile,
this program prints the profile in CSV table format.
Each row represents a layer in the profile.

The output format can be optionally converted to a
format suitable for GNUPlot.
'''

import sys
import json
import argparse
import prn_utils


all_features = ['name', 'timeMs', 'averageMs', 'percentage']

default_features = ",".join(all_features)

descriptions = ['layer name', 'total layer time', 'average layer time', 'percentage of total time']

features_description = prn_utils.combine_descriptions('Features are (times in ms):',
                                                      all_features, descriptions)



def hasNames(feature_set):
    ''' Check if the name is included in the set '''

    return 'name' in feature_set



def total_data(data, names):
    ''' Add row at the bottom with the total '''

    accumulator = []

    if names:
        start = 1
        accumulator.append('total')
    else:
        start = 0
    for f in range(start, len(data[0])):
        accumulator.append(0)

    for row in data:
        for f in range(start, len(row)):
             accumulator[f] += row[f]

    data.append(accumulator)

    return data



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', metavar='F[,F]*', default='name,timeMs,averageMs,percentage',
                        help='Comma separated list of features to print. ' + features_description)
    parser.add_argument('--total', action='store_true', help='Add total time row.')
    parser.add_argument('--gp', action='store_true', help='Print GNUPlot format.')
    parser.add_argument('--no-header', action='store_true', help='Omit the header row.')
    parser.add_argument('name', metavar='filename', help='Profile file.')
    args = parser.parse_args()

    feature_set = args.features.split(',')
    count = args.gp and not hasNames(feature_set)

    if not args.no_header:
        prn_utils.print_header(all_features, feature_set, args.gp, count)

    with open(args.name) as f:
        profile = json.load(f)

        data = prn_utils.filter_data(profile[1:], all_features, feature_set)

        if args.total:
            data = total_data(data, hasNames(feature_set))

        prn_utils.print_csv(data, count)


if __name__ == '__main__':
    sys.exit(main())
