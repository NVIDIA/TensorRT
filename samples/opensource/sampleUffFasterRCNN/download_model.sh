#!/bin/bash
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

set -eo pipefail
# check for wget
which wget || { echo 'wget not found, please install.' && exit 1; }
# download
mkdir -p uff_faster_rcnn && \
cd uff_faster_rcnn && \
wget 'https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_4.x_apps/master/models/frcnn/faster_rcnn.pb' && \
wget 'https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_4.x_apps/master/models/frcnn/2015_0502_034830_005_00001_rain_000179.ppm' && \
wget 'https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_4.x_apps/master/models/frcnn/2016_1111_185016_003_00001_night_000441.ppm' && \
wget 'https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_4.x_apps/master/models/frcnn/57ea04a57823530017bf15bf_000000.ppm' && \
wget 'https://raw.githubusercontent.com/NVIDIA-AI-IOT/deepstream_4.x_apps/master/models/frcnn/57ea04a57823530017bf15bf_001008.ppm' && \
ls *.ppm | cut -d. -f1 >> list.txt && \
echo 'Model downloading finished !' && \
cd ..
