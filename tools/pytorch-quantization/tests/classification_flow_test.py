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


"""Tests of the classification flow"""
import os
import subprocess
import sys
from os import path
import glob
import pytest
# pylint:disable=missing-docstring, no-self-use

class TestClassificationFlow():

    def test_resnet18(self, request, pytestconfig):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dataset_dir = pytestconfig.getoption('--data-dir')

        # skip if the data dir flag was not set
        if not dataset_dir:
            pytest.skip("Prepare required dataset and use --data-dir option to enable")

        # Verify data dir exists
        if not path.exists(dataset_dir):
            print("Dataset path %s doesn't exist"%(dataset_dir), file=sys.stderr)
            assert path.exists(dataset_dir)

        # Append required paths to PYTHONPATH
        test_env = os.environ.copy()
        if 'PYTHONPATH' not in test_env:
            test_env['PYTHONPATH'] = ""

        # Add project root and torchvision to the path (assuming running in nvcr.io/nvidia/pytorch:20.08-py3)
        test_env['PYTHONPATH'] += ":/opt/pytorch/vision/references/classification/:%s/../"%(dir_path)

        # Add requirement egg files manually to path since we're spawning a new process (downloaded by setuptools)
        for egg in glob.glob(dir_path + "/../.eggs/*.egg"):
            test_env['PYTHONPATH'] += ":%s"%(egg)

        # Run in a subprocess to avoid contaminating the module state for other test cases
        ret = subprocess.run(
            [
                'python3', dir_path + '/../examples/torchvision/classification_flow.py',
                '--data-dir', dataset_dir,
                '--model', 'resnet18', '--pretrained',
                '-t', '0.5',
                '--num-finetune-epochs', '2',
                '--evaluate-onnx',
            ],
            env=test_env,
            check=False, stdout=subprocess.PIPE)

        # If the test failed dump the output to stderr for better logging
        if ret.returncode != 0:
            print(ret.stdout, file=sys.stderr)

        assert ret.returncode == 0
