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


import json
import argparse


def device_info(json_fname: str):
    def install_pycuda(json_fname: str):
        '''trex does not install pycuda by default, because pycuda requires a CUDA
        installation, which limits where trex can be used. However, pycuda is
        mandatory for querying device information.'''

        print("pycuda is not installed - attempting installation.\n")
        import subprocess
        try:
            subprocess.run(f"python3 -m pip install pycuda".split(), check=True)
        except subprocess.CalledProcessError:
            print("\nFailed to install pycuda.")
            print("If CUDA is installed on your machine, try adding to $PATH ", end="")
            print("the path to nvcc (e.g. try `which nvcc`)")
            print(f"\nExiting: file {json_fname} was not generated!")
            exit(1)

    try:
        import pycuda.driver as drv
    except ImportError:
        install_pycuda(json_fname)
        import pycuda.driver as drv

    drv.init()
    print (f"Device info: {drv.Device.count()} device(s) found")

    devices_metadata = []
    for i in range(drv.Device.count()):
        dev = drv.Device(i)
        metadata = {
            'Name': dev.name(),
            'ComputeCapability': dev.compute_capability(),
            'TotalMemory': dev.total_memory(),
            "CUDA_VERSION": drv.get_version(),
        }
        for k, v in dev.get_attributes().items():
            metadata[str(k)] = v
        devices_metadata.append(metadata)

    with open(json_fname, 'w') as fout:
        json.dump(devices_metadata , fout)
        print(f"Device info: generated output file {json_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="name of output json file")
    args = parser.parse_args()
    device_info(json_fname=args.input)
