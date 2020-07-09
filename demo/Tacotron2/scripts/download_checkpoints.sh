#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Prepare the download directory
mkdir -p checkpoints && cd checkpoints

# Download the Tacotron2 and Waveglow checkpoints
if [ ! -f "checkpoints/tacotron2pyt_fp16_v3/tacotron2_1032590_6000_amp" ]; then
    echo "Downloading Tacotron2 fp16 checkpoint from NGC"
    ngc registry model download-version nvidia/tacotron2pyt_fp16:3
fi;
if [ ! -f "checkpoints/waveglow256pyt_fp16_v2/waveglow_1076430_14000_amp" ]; then
    echo "Downloading Waveglow fp16 checkpoint from NGC"
    ngc registry model download-version nvidia/waveglow256pyt_fp16:2
fi;

cd -
