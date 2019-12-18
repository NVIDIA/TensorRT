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
Simple printing utils

Utils to print traces and profiles in CSV format
'''


from __future__ import print_function

def combine_descriptions(prolog, features, descriptions):
    ''' Combine features with their descriptions '''

    full_description = prolog
    sep = ' '
    for feature, description in zip(features, descriptions):
        full_description += sep + feature + ' (' + description + ')'
        sep = ', '

    return full_description



def print_header(allFeatures, features, gp, count):
    ''' Print table header '''

    if gp:
        sep = '#'
        if count:
            sep += 'count, '
    else:
        sep = ''

    for feature in allFeatures:
       if feature in features:
           print(sep + feature, end = '')
           sep = ', '

    print('')



def print_csv(data, count):
    ''' Print trace in CSV format '''

    c = 0
    for row in data:
        if count:
            print(c, end = '')
            c += 1
            sep = ', '
        else:
            sep = ''
        for r in row:
            print('{}{:.6}'.format(sep, float(r)), end = '')
            sep = ', '
        print('')



def filter_data(data, all_features, feature_set):
    ''' Drop features not in the given set '''

    filteredData = []

    for d in data:
        row = []
        for f in all_features:
            if f in feature_set:
                row.append(d[f])
        filteredData.append(row)

    return filteredData
