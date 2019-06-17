/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
#ifndef NV_INFER_PLUGIN_UTILS_H
#define NV_INFER_PLUGIN_UTILS_H
//!
//! \file NvPluginUtils.h
//!
//! This is the API for the Nvidia provided TensorRT plugin utilities.
//! It lists all the parameters utilized by the TensorRT plugins.
//!

namespace nvinfer1
{
namespace plugin
{

//!
//! \brief RPROIParams is used to create the RPROIPlugin instance.
//! It contains:
//! \param poolingH Height of the output in pixels after ROI pooling on feature map.
//! \param poolingW Width of the output in pixels after ROI pooling on feature map.
//! \param featureStride Feature stride; ratio of input image size to feature map size. Assuming that max pooling layers in neural network use square filters.
//! \param preNmsTop Number of proposals to keep before applying NMS.
//! \param nmsMaxOut Number of remaining proposals after applying NMS.
//! \param anchorsRatioCount Number of anchor box ratios.
//! \param anchorsScaleCount Number of anchor box scales.
//! \param iouThreshold IoU (Intersection over Union) threshold used for the NMS step.
//! \param minBoxSize Minimum allowed bounding box size before scaling, used for anchor box calculation.
//! \param spatialScale Spatial scale between the input image and the last feature map.
//!
struct RPROIParams
{
    int poolingH;
    int poolingW;
    int featureStride;
    int preNmsTop;
    int nmsMaxOut;
    int anchorsRatioCount;
    int anchorsScaleCount;
    float iouThreshold;
    float minBoxSize;
    float spatialScale;
};

} // end plugin namespace
} // end nvinfer1 namespace
#endif
