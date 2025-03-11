#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_plugin_source(
    allClassNMS.cu
    bboxDeltas2Proposals.cu
    common.cu
    cropAndResizeKernel.cu
    decodeBbox3DKernels.cu
    decodeBBoxes.cu
    detectionForward.cu
    extractFgScores.cu
    gatherTopDetections.cu
    generateAnchors.cu
    gridAnchorLayer.cu
    kernel.cpp
    kernel.h
    lReLU.cu
    maskRCNNKernels.cu
    maskRCNNKernels.h
    nmsLayer.cu
    normalizeLayer.cu
    permuteData.cu
    pillarScatterKernels.cu
    priorBoxLayer.cu
    proposalKernel.cu
    proposalsForward.cu
    reducedMathPlugin.h
    regionForward.cu
    reorgForward.cu
    roiPooling.cu
    rproiInferenceFused.cu
    saturate.h
    sortScoresPerClass.cu
    sortScoresPerImage.cu
    voxelGeneratorKernels.cu
)
