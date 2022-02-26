#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import os
import utils.model as model_utils  # UFF conversion
from utils.paths import PATHS  # Path management

# Model used for inference
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'

def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(description='Run object detection inference on input image.')
    parser.add_argument('-w', '--workspace_dir',
        help='sample workspace directory')
    parser.add_argument('-d', '--data',
        help="Specify the data directory where it is saved in. $TRT_DATA_DIR will be overwritten by this argument.")

    args, _ = parser.parse_known_args()

    data_dir = os.environ.get('TRT_DATA_DIR', None) if args.data is None else args.data
    if data_dir is None:
        raise ValueError("Data directory must be specified by either `-d $DATA` or environment variable $TRT_DATA_DIR.")
    PATHS.set_data_dir_path(data_dir)

    # Set workspace dir path if passed by user
    if args.workspace_dir:
        PATHS.set_workspace_dir_path(args.workspace_dir)

    try:
        os.makedirs(PATHS.get_workspace_dir_path())
    except:
        pass

    # Verify Paths after adjustments. This also exits script if verification fails
    PATHS.verify_all_paths()

    return args

def main():
    # Parse command line arguments
    args = parse_commandline_arguments()

    # Fetch .uff model path
    ssd_model_uff_path = PATHS.get_model_uff_path(MODEL_NAME)
    # convert from .pb if needed, using prepare_ssd_model
    if not os.path.exists(ssd_model_uff_path):
        model_utils.prepare_ssd_model(MODEL_NAME)

if __name__ == '__main__':
    main()
