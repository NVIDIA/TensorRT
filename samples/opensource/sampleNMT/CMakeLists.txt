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
SET(SAMPLE_SOURCES
    sampleNMT.cpp 
    trtUtil.cpp
) 
include(../../CMakeSamplesTemplate.txt)

# define SAMPLE_NMT_DATA_SOURCES and SAMPLE_NMT_MODEL_SOURCES
set(SAMPLE_NMT_MODEL_SOURCES 
    model/beamSearchPolicy.cpp
    model/componentWeights.cpp
    model/contextNMT.cpp
    model/debugUtil.cpp
    model/lstmDecoder.cpp
    model/lstmEncoder.cpp
    model/multiplicativeAlignment.cpp
    model/slpAttention.cpp
    model/slpEmbedder.cpp
    model/slpProjection.cpp
    model/softmaxLikelihood.cpp
)

set(SAMPLE_NMT_DATA_SOURCES
    data/benchmarkWriter.cpp
    data/bleuScoreWriter.cpp
    data/dataWriter.cpp
    data/limitedSamplesDataReader.cpp
    data/textReader.cpp
    data/textWriter.cpp
    data/vocabulary.cpp
)

set(TARGET_NAME ${SAMPLE_NAME})
target_sources(${TARGET_NAME}
PRIVATE 
    ${SAMPLE_NMT_MODEL_SOURCES}
    ${SAMPLE_NMT_DATA_SOURCES}
)
