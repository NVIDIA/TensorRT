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
set(OPENSOURCE_SAMPLES_LIST
    sampleCharRNN
    sampleDynamicReshape
    sampleFasterRCNN
    sampleGoogleNet
    sampleINT8
    sampleINT8API
    sampleMLP
    sampleMNIST
    sampleMNISTAPI
    sampleMovieLens
    sampleMovieLensMPS
    sampleNMT
    sampleOnnxMNIST
    samplePlugin
    sampleReformatFreeIO
    sampleSSD
    sampleUffFasterRCNN
    sampleUffMaskRCNN
    sampleUffMNIST
    sampleUffPluginV2Ext
    sampleUffSSD
    trtexec
)

foreach(SAMPLE_ITER ${OPENSOURCE_SAMPLES_LIST})
    add_subdirectory(${SAMPLE_ITER})
endforeach(SAMPLE_ITER)
