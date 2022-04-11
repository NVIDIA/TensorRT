/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TRT_CAFFE_PARSER_WEIGHT_TYPE_H
#define TRT_CAFFE_PARSER_WEIGHT_TYPE_H

namespace nvcaffeparser1
{
enum class WeightType
{
    // types for convolution, deconv, fully connected
    kGENERIC = 0, // typical weights for the layer: e.g. filter (for conv) or matrix weights (for innerproduct)
    kBIAS = 1,    // bias weights

    // These enums are for BVLCCaffe, which are incompatible with nvCaffe enums below.
    // See batch_norm_layer.cpp in BLVC source of Caffe
    kMEAN = 0,
    kVARIANCE = 1,
    kMOVING_AVERAGE = 2,

    // These enums are for nvCaffe, which are incompatible with BVLCCaffe enums above
    // See batch_norm_layer.cpp in NVidia fork of Caffe
    kNVMEAN = 0,
    kNVVARIANCE = 1,
    kNVSCALE = 3,
    kNVBIAS = 4
};
} //namespace nvcaffeparser1
#endif //TRT_CAFFE_PARSER_WEIGHT_TYPE_H
