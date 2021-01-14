#!/bin/bash
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

# Prepare the download directory
mkdir -p checkpoints && cd checkpoints

# Download the Tacotron2 and Waveglow checkpoints
if [ ! -f "checkpoints/tacotron2_pyt_ckpt_amp_v19.09.0/nvidia_tacotron2pyt_fp16_20190427" ]; then
    echo "Downloading Tacotron2 checkpoint from NGC"
    ngc registry model download-version nvidia/tacotron2_pyt_ckpt_amp:19.09.0
fi;
if [ ! -f "checkpoints/waveglow_ckpt_amp_256_v19.10.0/nvidia_waveglow256pyt_fp16" ]; then
    echo "Downloading Waveglow checkpoint from NGC"
    ngc registry model download-version nvidia/waveglow_ckpt_amp_256:19.10.0
fi;

cd -
