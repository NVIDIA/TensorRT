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

#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1
{
ILayer* parseCrop(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    // To crop, elements of the first bottom are selected to fit the dimensions
    // of the second, reference bottom. The crop is configured by
    // - the crop `axis` to pick the dimensions for cropping
    // - the crop `offset` to set the shift for all/each dimension
    // to align the cropped bottom with the reference bottom.
    // All dimensions up to but excluding `axis` are preserved, while
    // the dimensions including and trailing `axis` are cropped.
    // If only one `offset` is set, then all dimensions are offset by this amount.
    // Otherwise, the number of offsets must equal the number of cropped axes to
    // shift the crop in each dimension accordingly.
    // Note: standard dimensions are N,C,H,W so the default is a spatial crop,
    // and `axis` may be negative to index from the end (e.g., -1 for the last
    // axis).

    if (!checkBlobs(msg, 2, 1))
    {
        return nullptr;
    }

    // ONLY IMPLEMENT SPATIAL CROPPING
    // IF CROP LAYER IS NOT SPATIAL CROP, ABORT
    const trtcaffe::CropParameter& p = msg.crop_param();
    Dims3 inputDims = parserutils::getCHW(tensors[msg.bottom(0)]->getDimensions());
    Dims3 refDims = parserutils::getCHW(tensors[msg.bottom(1)]->getDimensions());
    bool hasAxis = p.has_axis();         // optional parameter
    int axis = hasAxis ? p.axis() : 2;   // default is 2 - spatial crop
    axis = (axis < 0) ? 4 + axis : axis; // axis negative number correction

    // acceptable axis values: 2, 3, -1, -2
    // unacceptable axis values: 0, 1, -3, -4 and anything else
    // acceptable corrected axis values: 2, 3
    // unacceptable corrected axis values: 0, 1 and anything else
    // protect against "garbage" input arguments
    bool axis_abort = (axis != 2 && axis != 3);

    // must be at least one offset
    // if only one offset, the same offset applies to all the dimensions
    // including the chosen axis and trailing it
    // if more than one offset, the number of offsets must match the number
    // of dimensions consisting of the axis and all the dimensions trailing it
    int num_offsets = p.offset_size();

    // 1 + (3 - axis) = 4 - axis
    // this is only valid for acceptable corrected axis values
    // if !axis_abort then invariant that num_dims == 1 || num_dims == 2
    int num_dims = 4 - axis;
    bool offset_abort = (num_offsets != 0 && num_offsets != 1 && num_offsets != num_dims);

    if (axis_abort)
    {
        std::cout << "Caffe Parser: Invalid axis in crop layer - only spatial cropping is supported" << std::endl;
        return nullptr;
    }

    if (offset_abort)
    {
        std::cout << "Caffe Parser: Invalid number of offsets in crop layer" << std::endl;
        return nullptr;
    }

    // get the offsets
    // the offsets are zero by default (in case no offset is specified)
    int offsetHeight = 0;
    int offsetWidth = 0;

    if (num_offsets != 0)
    {
        // offsetHeight will only be specified if the H channel is the chosen axis
        // in this case, regardless of whether there are one or multiple offsets
        // offsetHeight should always be the zero-indexed offset
        offsetHeight = axis == 2 ? p.offset(0) : 0;
        // offsetWidth should always be specified
        // if there is only one offset, use the zero-indexed offset
        // otherwise, use the one-indexed offset since the zero-indexed offet
        // is for offsetHeight
        offsetWidth = num_offsets == 1 ? p.offset(0) : p.offset(1);
    }

    // now compute the prePadding and postPadding required to perform the crop
    // so that the first bottom is the same spatial size as the second bottom
    // prePadding is the padding to the left/bottom (assuming origin is lower-left).
    // postPadding is the padding to the right/top.
    // - ( inputDims.h() - refDims.h() - offsetHeight ) = -inputDims.h() + refDims.h() + offsetHeight
    // - ( inputDims.w() - refDims.w() - offsetWidth ) = -inputDims.w() + refDims.w() + offsetWidth
    int prePadHeight = -offsetHeight;
    int prePadWidth = -offsetWidth;
    int postPadHeight = -inputDims.d[1] + refDims.d[1] + offsetHeight;
    int postPadWidth = -inputDims.d[2] + refDims.d[2] + offsetWidth;

    Dims prePadding = parserutils::toDims(prePadHeight, prePadWidth);
    Dims postPadding = parserutils::toDims(postPadHeight, postPadWidth);
    return network.addPaddingNd(*tensors[msg.bottom(0)], prePadding, postPadding);
}
} //namespace nvcaffeparser1
