#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import prn_utils as pu



allFeatures = ['name', 'timeMs', 'averageMs', 'percentage']

defaultFeatures = ",".join(allFeatures)

descriptions = ['layer name', 'total layer time', 'average layer time', 'percentage of total time']

featuresDescription = pu.combineDescriptions('Features are (times in ms):', allFeatures, descriptions)



def hasNames(features):
    ''' Check if the name is included in the set '''

    return 'name' in features



def totalData(features, profile):
    ''' Add row at the bottom with the total '''

    accumulator = {}
    for f in features:
        accumulator[f] = 0
    accumulator['name'] = 'total'

    for row in profile:
        for f in features:
            if f in row and not f == 'name':
                accumulator[f] += row[f]

    return accumulator



def findAndRemove(profile, name):
    ''' Find named row in profile and remove '''

    for r in range(len(profile)):
        if profile[r]['name'] == name:
            row = profile[r]
            del profile[r]
            return row

    return None



def refName(name):
    ''' Add prefix ref to name '''

    return 'ref' + name[0].capitalize() + name[1:]


def refFeatures(names):
    ''' Add prefix ref to features names '''

    refNames = []
    for name in names:
        refNames.append(refName(name))
    return refNames



def mergeHeaders(features, skipFirst = True):
    ''' Duplicate feature names for reference and target profile '''

    if skipFirst:
        return [features[0]] + refFeatures(features[1:]) + features[1:] + ['% difference']
    return refFeatures(features) + features + ['% difference']



def addReference(row, reference):
    ''' Add reference results to results dictionary '''

    for k,v in reference.items():
        if k == 'name':
            if k in row:
                continue
        else:
            k = refName(k)
        row[k] = v



def mergeRow(reference, profile, diff):
    ''' Merge reference and target profile results into a single row '''

    row = {}
    if profile:
        row = profile
    if reference:
        addReference(row, reference)
    if diff:
        row['% difference'] = diff;

    return row



def alignData(reference, profile, threshold):
    ''' Align and merge reference and target profiles '''

    alignedData = []
    for ref in reference:
        prof = findAndRemove(profile, ref['name'])

        if prof:
            diff = (prof['averageMs'] / ref['averageMs'] - 1)*100
            if abs(diff) >= threshold:
                alignedData.append(mergeRow(ref, prof, diff))
        else:
            alignedData.append(mergeRow(ref, None, None))

    for prof in profile:
        alignedData.append(mergeRow(None, prof, None))

    return alignedData



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', metavar='F[,F]*', default=defaultFeatures,
                        help='Comma separated list of features to print. ' + featuresDescription)
    parser.add_argument('--total', action='store_true', help='Add total time row.')
    parser.add_argument('--gp', action='store_true', help='Print GNUPlot format.')
    parser.add_argument('--no-header', action='store_true', help='Omit the header row.')
    parser.add_argument('--threshold', metavar='T', default=0.0, type=float,
                        help='Threshold of percentage difference.')
    parser.add_argument('--reference', metavar='R', help='Reference profile file name.')
    parser.add_argument('name', metavar='filename', help='Profile file.')
    args = parser.parse_args()

    global allFeatures
    features = args.features.split(',')
    for f in features:
        if not f in allFeatures:
            print('Feature {} not recognized'.format(f))
            return

    count = args.gp and not hasNames(features)

    profile = None
    reference = None

    with open(args.name) as f:
        profile = json.load(f)
        profileCount = profile[0]['count']
        profile = profile[1:]

    if args.reference:
        with open(args.reference) as f:
            reference = json.load(f)
            referenceCount = reference[0]['count']
            reference = reference[1:]
        allFeatures = mergeHeaders(allFeatures)
        features = mergeHeaders(features, hasNames(features))

    if not args.no_header:
        if reference:
            comment = '#' if args.gp else ''
            print(comment + 'reference count: {} - profile count: {}'.format(referenceCount, profileCount))
        pu.printHeader(allFeatures, features, args.gp, count)

    if reference:
        profile = alignData(reference, profile, args.threshold)

    if args.total:
        profile.append(totalData(allFeatures, profile))
        if reference:
            total = profile[len(profile) - 1]
            total['% difference'] = (total['averageMs'] / total['refAverageMs'] - 1)*100

    profile = pu.filterData(profile, allFeatures, features)

    pu.printCsv(profile, count)



if __name__ == '__main__':
    sys.exit(main())
