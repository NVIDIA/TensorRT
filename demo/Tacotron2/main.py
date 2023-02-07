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

import argparse
from train import main as main_train
from inference_perf import main as main_infer

def parse_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('--bench-class',  type=str, choices=['train', 'perf-infer', 'perf-train'], required=True, help='Choose test class')

    return parser

def main():

    parser = argparse.ArgumentParser(description='PyTorch Tacotron 2 Testing')
    parser = parse_args(parser)
    args, unknown_args = parser.parse_known_args()

    if "train" in args.bench_class:
        main_train()
    else:
        main_infer()

if __name__ == '__main__':
    main()
