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

// This file contains all bindings related to TensorRT INetworkDefinition.
#include "ForwardDeclarations.h"
#include "utils.h"
#include <pybind11/stl.h>

#if ENABLE_INETWORK_SERIALIZE
#include "NvInferSerialize.h"
#endif

#include "infer/pyGraphDoc.h"

// clang-format off
namespace tensorrt
{
    using namespace nvinfer1;
    // Long lambda functions should go here rather than being inlined into the bindings (1 liners are OK).
    namespace lambdas
    {
        Weights optionalWeights(Weights* weights)
        {
            if (weights)
            {
                return *weights;
            }
            return Weights{DataType::kFLOAT, nullptr, 0};
        }

        static const auto get_dynamic_range = [] (ITensor const& self) -> py::object {
            if (self.dynamicRangeIsSet()) {
                return py::make_tuple(self.getDynamicRangeMin(), self.getDynamicRangeMax());
            } else {
                return py::none{};
            }
        };

        static const auto set_dynamic_range = [] (ITensor& self, std::vector<float> const& range) {
            PY_ASSERT_VALUE_ERROR(range.size() == 2, "Dynamic range must contain exactly 2 elements");
            PY_ASSERT_VALUE_ERROR(self.setDynamicRange(range[0], range[1]),
                "Error in set dynamic range");
        };

        // For permutation
        static const auto permutation_vector_constructor = [] (std::vector<int32_t> const& in) {
            // Static casts are required here, so that MAX_DIMS is resolved at compile/link time.
            int32_t const maxDims{static_cast<int32_t const>(Dims::MAX_DIMS)};
            PY_ASSERT_VALUE_ERROR(in.size() <= maxDims,
                "Invalid input length. Max expected length is " + std::to_string(maxDims));
            Permutation* self = new Permutation{};
            for (int32_t i = 0; i < in.size(); ++i)
                self->order[i] = in[i];
            return self;
        };

        static const auto permutation_to_str = [] (Permutation const& self) {
            int32_t const maxDims = static_cast<int32_t const>(Dims::MAX_DIMS);
            std::string temp = "(";
            for (int32_t i = 0; i < maxDims - 1; ++i)
                temp += std::to_string(self.order[i]) + ", ";
            temp += std::to_string(self.order[maxDims - 1]) + ")";
            return temp;
        };

        // TODO: Add slicing support?
        static const auto permutation_getter = [] (Permutation const& self, int32_t const pyIndex) {
            PY_ASSERT_INDEX_ERROR(pyIndex < static_cast<int32_t const>(Dims::MAX_DIMS));
            int32_t const index{(pyIndex < 0) ? static_cast<int32_t const>(Dims::MAX_DIMS) + pyIndex : pyIndex};
            // Static cast is REQUIRED here, or chaos ensues as MAX_DIMS is not pulled in at link time.
            PY_ASSERT_INDEX_ERROR(index >= 0 && index < static_cast<int32_t const>(Dims::MAX_DIMS));
            return self.order[index];
        };

        static const auto permutation_setter = [] (Permutation& self, int32_t const pyIndex, int32_t const item) {
            PY_ASSERT_INDEX_ERROR(pyIndex < static_cast<int32_t const>(Dims::MAX_DIMS));
            int32_t const index = (pyIndex < 0) ? static_cast<int32_t const>(Dims::MAX_DIMS) + pyIndex : pyIndex;
            // Static cast is REQUIRED here, or chaos ensues as MAX_DIMS is not pulled in at link time.
            PY_ASSERT_INDEX_ERROR(index >= 0 && index < static_cast<int32_t const>(Dims::MAX_DIMS));
            self.order[index] = item;
        };

        static const auto permutation_len = [] (Permutation const& self) {
            return static_cast<int32_t const>(Dims::MAX_DIMS);
        };

        // For INetworkDefinition
        // Need a ptr to const-ptr to ITensor.
        static const auto add_concatenation = [] (INetworkDefinition& self, std::vector<ITensor*> const& inputs) {
            return self.addConcatenation(inputs.data(), inputs.size());
        };

        // Need a ptr to const-ptr to ITensor.
        static const auto add_plugin_v2 = [] (INetworkDefinition& self, std::vector<ITensor*> const& inputs, IPluginV2& plugin) {
            return self.addPluginV2(inputs.data(), inputs.size(), plugin);
        };

        IConvolutionLayer* add_convolution(INetworkDefinition& self, ITensor& input, int32_t numOutputMaps, DimsHW kernelSize, Weights kernel, Weights* bias)
        {
            return self.addConvolution(input, numOutputMaps, kernelSize, kernel, optionalWeights(bias));
        };

        static const auto add_convolution_nd = [](INetworkDefinition& self, ITensor& input, int32_t numOutputMaps, Dims kernelSize, Weights kernel, Weights* bias)
        {
            return self.addConvolutionNd(input, numOutputMaps, kernelSize, kernel, optionalWeights(bias));
        };

        IFullyConnectedLayer* add_fully_connected(INetworkDefinition& self, ITensor& input, int32_t numOutputs, Weights kernel, Weights* bias)
        {
            return self.addFullyConnected(input, numOutputs, kernel, optionalWeights(bias));
        };

        IGridSampleLayer* add_grid_sample(INetworkDefinition& self, ITensor& input, ITensor& grid)
        {
            return self.addGridSample(input, grid);
        };

        static const auto add_scale = [](INetworkDefinition& self, ITensor& input, ScaleMode mode, Weights* shift, Weights* scale, Weights* power)
        {
            return self.addScale(input, mode, optionalWeights(shift), optionalWeights(scale), optionalWeights(power));
        };

        static const auto add_scale_nd = [](INetworkDefinition& self, ITensor& input, ScaleMode mode, Weights* shift, Weights* scale, Weights* power, int32_t channelAxis)
        {
            return self.addScaleNd(input, mode, optionalWeights(shift), optionalWeights(scale), optionalWeights(power), channelAxis);
        };

        static const auto add_quantize = [](INetworkDefinition& self, ITensor& input, ITensor& scale)
        {
            return self.addQuantize(input, scale);
        };

        static const auto add_dequantize = [](INetworkDefinition& self, ITensor& input, ITensor& scale)
        {
            return self.addDequantize(input, scale);
        };

        IDeconvolutionLayer* add_deconvolution(INetworkDefinition& self, ITensor& input, int32_t numOutputMaps, DimsHW kernelSize, Weights kernel, Weights* bias)
        {
            return self.addDeconvolution(input, numOutputMaps, kernelSize, kernel, optionalWeights(bias));
        };

        static const auto add_scatter = [](INetworkDefinition& self, ITensor& data, ITensor& indices, ITensor& updates, ScatterMode mode)
        {
            return self.addScatter(data, indices, updates, mode);
        };

        static const auto add_deconvolution_nd = [](INetworkDefinition& self, ITensor& input, int32_t numOutputMaps, Dims kernelSize, Weights kernel, Weights* bias)
        {
            return self.addDeconvolutionNd(input, numOutputMaps, kernelSize, kernel, optionalWeights(bias));
        };

        static const auto add_einsum = [] (INetworkDefinition& self, const std::vector<ITensor*>& inputs, const char* equation) {
            return self.addEinsum(inputs.data(), inputs.size(), equation);
        };

#if ENABLE_INETWORK_SERIALIZE
        // Serialization
        static const auto network_serialize = [] (INetworkDefinition& self) {
            return serialize::serializeNetwork(self);
        };
#endif

        // TODO: Need to ensure that these are returning by reference rather than by copy.
        // NumPy getters for layers.
        static const auto conv_get_kernel = [](IConvolutionLayer& self) { auto w = self.getKernelWeights(); return utils::weights_to_numpy(w); };
        static const auto conv_get_bias = [](IConvolutionLayer& self) { auto w = self.getBiasWeights(); return utils::weights_to_numpy(w); };

        static const auto fc_get_kernel = [](IFullyConnectedLayer& self) { auto w = self.getKernelWeights(); return utils::weights_to_numpy(w); };
        static const auto fc_get_bias = [](IFullyConnectedLayer& self) { auto w = self.getBiasWeights(); return utils::weights_to_numpy(w); };

        static const auto scale_get_shift = [](IScaleLayer& self) { auto w = self.getShift(); return utils::weights_to_numpy(w); };
        static const auto scale_get_scale = [](IScaleLayer& self) { auto w = self.getScale(); return utils::weights_to_numpy(w); };
        static const auto scale_get_power = [](IScaleLayer& self) { auto w = self.getPower(); return utils::weights_to_numpy(w); };

        static const auto deconv_get_kernel = [](IDeconvolutionLayer& self) { auto w = self.getKernelWeights(); return utils::weights_to_numpy(w); };
        static const auto deconv_get_bias = [](IDeconvolutionLayer& self) { auto w = self.getBiasWeights(); return utils::weights_to_numpy(w); };

        static const auto rnnv2_get_weights = [](IRNNv2Layer& self, int32_t index, RNNGateType gate, bool isW) {
            auto w = self.getWeightsForGate(index, gate, isW); return utils::weights_to_numpy(w);
        };
        static const auto rnnv2_get_bias = [](IRNNv2Layer& self, int32_t index, RNNGateType gate, bool isW) {
            auto w = self.getBiasForGate(index, gate, isW); return utils::weights_to_numpy(w);
        };

        static const auto constant_get_weights = [](IConstantLayer& self) { auto w = self.getWeights(); return utils::weights_to_numpy(w); };

        // TODO: Add slicing support?
        static const auto network_getitem = [](INetworkDefinition& self, int32_t pyIndex) {
            // Support python's negative indexing
            size_t index = (pyIndex < 0) ? self.getNbLayers() + pyIndex : pyIndex;
            PY_ASSERT_INDEX_ERROR(index < self.getNbLayers());
            return self.getLayer(index);
        };

        static const auto resize_set_scales = [](IResizeLayer& self, const std::vector<float>& scales) { self.setScales(scales.data(), scales.size()); };
        static const auto resize_get_scales = [](IResizeLayer& self)
        {
            size_t nbScales = self.getScales(0, nullptr);
            // nbScales of -1 signifies that scales are unused for resize caluclation. Return an empty vector here.
            if (nbScales == -1)
            {
                return std::vector<float>();
            }
            std::vector<float> scales(nbScales, 1.0f);
            self.getScales(nbScales, scales.data());
            return scales;
        };
    } /* lambdas */

    void bindGraph(py::module& m)
    {
        // Bind to a Python enum called LayerType.
        py::enum_<LayerType>(m, "LayerType", LayerTypeDoc::descr)
            .value("CONVOLUTION", LayerType::kCONVOLUTION, LayerTypeDoc::CONVOLUTION)
            .value("FULLY_CONNECTED", LayerType::kFULLY_CONNECTED, LayerTypeDoc::FULLY_CONNECTED)
            .value("GRID_SAMPLE", LayerType::kGRID_SAMPLE, LayerTypeDoc::GRID_SAMPLE)
            .value("NMS", LayerType::kNMS, LayerTypeDoc::NMS)
            .value("ACTIVATION", LayerType::kACTIVATION, LayerTypeDoc::ACTIVATION)
            .value("POOLING", LayerType::kPOOLING, LayerTypeDoc::POOLING)
            .value("LRN", LayerType::kLRN, LayerTypeDoc::LRN)
            .value("SCALE", LayerType::kSCALE, LayerTypeDoc::SCALE)
            .value("SOFTMAX", LayerType::kSOFTMAX, LayerTypeDoc::SOFTMAX)
            .value("DECONVOLUTION", LayerType::kDECONVOLUTION, LayerTypeDoc::DECONVOLUTION)
            .value("CONCATENATION", LayerType::kCONCATENATION, LayerTypeDoc::CONCATENATION)
            .value("ELEMENTWISE", LayerType::kELEMENTWISE, LayerTypeDoc::ELEMENTWISE)
            .value("PLUGIN", LayerType::kPLUGIN, LayerTypeDoc::PLUGIN)
            .value("UNARY", LayerType::kUNARY, LayerTypeDoc::UNARY)
            .value("PADDING", LayerType::kPADDING, LayerTypeDoc::PADDING)
            .value("SHUFFLE", LayerType::kSHUFFLE, LayerTypeDoc::SHUFFLE)
            .value("REDUCE", LayerType::kREDUCE, LayerTypeDoc::REDUCE)
            .value("TOPK", LayerType::kTOPK, LayerTypeDoc::TOPK)
            .value("GATHER", LayerType::kGATHER, LayerTypeDoc::GATHER)
            .value("MATRIX_MULTIPLY", LayerType::kMATRIX_MULTIPLY, LayerTypeDoc::MATRIX_MULTIPLY)
            .value("RAGGED_SOFTMAX", LayerType::kRAGGED_SOFTMAX, LayerTypeDoc::RAGGED_SOFTMAX)
            .value("CONSTANT", LayerType::kCONSTANT, LayerTypeDoc::CONSTANT)
            .value("RNN_V2", LayerType::kRNN_V2, LayerTypeDoc::RNN_V2)
            .value("IDENTITY", LayerType::kIDENTITY, LayerTypeDoc::IDENTITY)
            .value("PLUGIN_V2", LayerType::kPLUGIN_V2, LayerTypeDoc::PLUGIN_V2)
            .value("SLICE", LayerType::kSLICE, LayerTypeDoc::SLICE)
            .value("SHAPE", LayerType::kSHAPE, LayerTypeDoc::SHAPE)
            .value("PARAMETRIC_RELU", LayerType::kPARAMETRIC_RELU, LayerTypeDoc::PARAMETRIC_RELU)
            .value("RESIZE", LayerType::kRESIZE, LayerTypeDoc::RESIZE)
            .value("TRIP_LIMIT", LayerType::kTRIP_LIMIT, LayerTypeDoc::TRIP_LIMIT)
            .value("RECURRENCE", LayerType::kRECURRENCE, LayerTypeDoc::RECURRENCE)
            .value("ITERATOR", LayerType::kITERATOR, LayerTypeDoc::ITERATOR)
            .value("LOOP_OUTPUT", LayerType::kLOOP_OUTPUT, LayerTypeDoc::LOOP_OUTPUT)
            .value("SELECT", LayerType::kSELECT, LayerTypeDoc::SELECT)
            .value("ASSERTION", LayerType::kASSERTION, LayerTypeDoc::ASSERTION)
            .value("FILL", LayerType::kFILL, LayerTypeDoc::FILL)
            .value("QUANTIZE", LayerType::kQUANTIZE, LayerTypeDoc::QUANTIZE)
            .value("DEQUANTIZE", LayerType::kDEQUANTIZE, LayerTypeDoc::DEQUANTIZE)
            .value("CONDITION", LayerType::kCONDITION, LayerTypeDoc::CONDITION)
            .value("CONDITIONAL_INPUT", LayerType::kCONDITIONAL_INPUT, LayerTypeDoc::CONDITIONAL_INPUT)
            .value("CONDITIONAL_OUTPUT", LayerType::kCONDITIONAL_OUTPUT, LayerTypeDoc::CONDITIONAL_OUTPUT)
            .value("SCATTER", LayerType::kSCATTER, LayerTypeDoc::SCATTER)
            .value("EINSUM", LayerType::kEINSUM, LayerTypeDoc::EINSUM)
            .value("ONE_HOT", LayerType::kONE_HOT, LayerTypeDoc::ONE_HOT)
            .value("NON_ZERO", LayerType::kNON_ZERO, LayerTypeDoc::NON_ZERO)
        ; // LayerType

        // Bind to a Python enum called TensorLocation.
        py::enum_<TensorLocation>(m, "TensorLocation", TensorLocationDoc::descr)
            .value("DEVICE", TensorLocation::kDEVICE, TensorLocationDoc::DEVICE)
            .value("HOST", TensorLocation::kHOST, TensorLocationDoc::HOST)
        ; // TensorLocation

        py::enum_<TensorFormat>(m, "TensorFormat", TensorFormatDoc::descr, py::arithmetic{})
            .value("LINEAR", TensorFormat::kLINEAR, TensorFormatDoc::LINEAR)
            .value("CHW2", TensorFormat::kCHW2, TensorFormatDoc::CHW2)
            .value("HWC8", TensorFormat::kHWC8, TensorFormatDoc::HWC8)
            .value("CHW4", TensorFormat::kCHW4, TensorFormatDoc::CHW4)
            .value("CHW16", TensorFormat::kCHW16, TensorFormatDoc::CHW16)
            .value("CHW32", TensorFormat::kCHW32, TensorFormatDoc::CHW32)
            .value("DHWC8", TensorFormat::kDHWC8, TensorFormatDoc::DHWC8)
            .value("CDHW32", TensorFormat::kCDHW32, TensorFormatDoc::CDHW32)
            .value("HWC", TensorFormat::kHWC, TensorFormatDoc::HWC)
            .value("DLA_LINEAR", TensorFormat::kDLA_LINEAR, TensorFormatDoc::DLA_LINEAR)
            .value("DLA_HWC4", TensorFormat::kDLA_HWC4, TensorFormatDoc::DLA_HWC4)
            .value("HWC16", TensorFormat::kHWC16, TensorFormatDoc::HWC16)
        ; // TensorFormat

        // ITensor
        py::class_<ITensor, std::unique_ptr<ITensor, py::nodelete>>(m, "ITensor", ITensorDoc::descr)
            .def_property("name", &ITensor::getName, &ITensor::setName)
            .def_property("shape", &ITensor::getDimensions, &ITensor::setDimensions)
            .def_property("dtype", &ITensor::getType, &ITensor::setType)
            .def_property("broadcast_across_batch", &ITensor::getBroadcastAcrossBatch, &ITensor::setBroadcastAcrossBatch)
            .def_property("location", &ITensor::getLocation, &ITensor::setLocation)
            .def_property("allowed_formats", &ITensor::getAllowedFormats, &ITensor::setAllowedFormats)
            .def_property_readonly("is_network_input", &ITensor::isNetworkInput)
            .def_property_readonly("is_network_output", &ITensor::isNetworkOutput)
            .def_property_readonly("is_shape_tensor", &ITensor::isShapeTensor)
            .def_property_readonly("is_execution_tensor", &ITensor::isExecutionTensor)
            .def_property("dynamic_range", lambdas::get_dynamic_range, lambdas::set_dynamic_range)
            .def_property("allowed_formats", &ITensor::getAllowedFormats, &ITensor::setAllowedFormats)
            .def("set_dynamic_range", &ITensor::setDynamicRange, "min"_a, "max"_a, ITensorDoc::set_dynamic_range)
            .def("reset_dynamic_range", &ITensor::resetDynamicRange, ITensorDoc::reset_dynamic_range)
            .def("set_dimension_name", &ITensor::setDimensionName, "index"_a, "name"_a, ITensorDoc::set_dimension_name)
            .def("get_dimension_name", &ITensor::getDimensionName, "index"_a, ITensorDoc::get_dimension_name)
        ;

        py::class_<ILayer, std::unique_ptr<ILayer, py::nodelete>>(m, "ILayer", ILayerDoc::descr)
            .def_property("name", &ILayer::getName, &ILayer::setName)
            .def_property_readonly("type", &ILayer::getType)
            .def_property_readonly("num_inputs", &ILayer::getNbInputs)
            .def_property_readonly("num_outputs", &ILayer::getNbOutputs)
            .def_property("precision", &ILayer::getPrecision, &ILayer::setPrecision)
            .def_property_readonly("precision_is_set", &ILayer::precisionIsSet)
            .def("set_input", &ILayer::setInput, "index"_a, "tensor"_a, ILayerDoc::set_input)
            .def("get_input", &ILayer::getInput, "index"_a, ILayerDoc::get_input)
            .def("get_output", &ILayer::getOutput, "index"_a, ILayerDoc::get_output)
            .def("reset_precision", &ILayer::resetPrecision, ILayerDoc::reset_precision)
            .def("set_output_type", &ILayer::setOutputType, "index"_a, "dtype"_a, ILayerDoc::set_output_type)
            .def("get_output_type", &ILayer::getOutputType, "index"_a, ILayerDoc::get_output_type)
            .def("output_type_is_set", &ILayer::outputTypeIsSet, "index"_a, ILayerDoc::output_type_is_set)
            .def("reset_output_type", &ILayer::resetOutputType, "index"_a, ILayerDoc::reset_output_type)
        ;

        py::enum_<PaddingMode>(m, "PaddingMode", PaddingModeDoc::descr)
            .value("EXPLICIT_ROUND_DOWN", PaddingMode::kEXPLICIT_ROUND_DOWN, PaddingModeDoc::EXPLICIT_ROUND_DOWN)
            .value("EXPLICIT_ROUND_UP", PaddingMode::kEXPLICIT_ROUND_UP, PaddingModeDoc::EXPLICIT_ROUND_UP)
            .value("SAME_UPPER", PaddingMode::kSAME_UPPER, PaddingModeDoc::SAME_UPPER)
            .value("SAME_LOWER", PaddingMode::kSAME_LOWER, PaddingModeDoc::SAME_LOWER)
            .value("CAFFE_ROUND_DOWN", PaddingMode::kCAFFE_ROUND_DOWN, PaddingModeDoc::CAFFE_ROUND_DOWN)
            .value("CAFFE_ROUND_UP", PaddingMode::kCAFFE_ROUND_UP, PaddingModeDoc::CAFFE_ROUND_UP)
        ;

        py::class_<IConvolutionLayer, ILayer, std::unique_ptr<IConvolutionLayer, py::nodelete>>(m, "IConvolutionLayer", IConvolutionLayerDoc::descr)
            .def_property("kernel_size", utils::deprecateMember(&IConvolutionLayer::getKernelSize, "kernel_size_nd"), utils::deprecateMember(&IConvolutionLayer::setKernelSize, "kernel_size_nd"))
            .def_property("num_output_maps", &IConvolutionLayer::getNbOutputMaps, &IConvolutionLayer::setNbOutputMaps)
            .def_property("stride", utils::deprecateMember(&IConvolutionLayer::getStride, "stride_nd"), utils::deprecateMember(&IConvolutionLayer::setStride, "stride_nd"))
            .def_property("padding", utils::deprecateMember(&IConvolutionLayer::getPadding, "padding_nd"), utils::deprecateMember(&IConvolutionLayer::setPadding, "padding_nd"))
            .def_property("pre_padding", &IConvolutionLayer::getPrePadding, &IConvolutionLayer::setPrePadding)
            .def_property("post_padding", &IConvolutionLayer::getPostPadding, &IConvolutionLayer::setPostPadding)
            .def_property("padding_mode", &IConvolutionLayer::getPaddingMode, &IConvolutionLayer::setPaddingMode)
            .def_property("num_groups", &IConvolutionLayer::getNbGroups, &IConvolutionLayer::setNbGroups)
            // Return numpy arrays instead of weights.
            .def_property("kernel", lambdas::conv_get_kernel, py::cpp_function(&IConvolutionLayer::setKernelWeights, py::keep_alive<1, 2>{}))
            .def_property("bias", lambdas::conv_get_bias, py::cpp_function(&IConvolutionLayer::setBiasWeights, py::keep_alive<1, 2>{}))
            .def_property("dilation", utils::deprecateMember(&IConvolutionLayer::getDilation, "dilation_nd"), utils::deprecateMember(&IConvolutionLayer::setDilation, "dilation_nd"))
            .def_property("kernel_size_nd", &IConvolutionLayer::getKernelSizeNd, &IConvolutionLayer::setKernelSizeNd)
            .def_property("stride_nd", &IConvolutionLayer::getStrideNd, &IConvolutionLayer::setStrideNd)
            .def_property("padding_nd", &IConvolutionLayer::getPaddingNd, &IConvolutionLayer::setPaddingNd)
            .def_property("dilation_nd", &IConvolutionLayer::getDilationNd, &IConvolutionLayer::setDilationNd)
        ;

        py::class_<IFullyConnectedLayer, ILayer, std::unique_ptr<IFullyConnectedLayer, py::nodelete>>(m, "IFullyConnectedLayer", IFullyConnectedLayerDoc::descr)
            .def_property("num_output_channels", &IFullyConnectedLayer::getNbOutputChannels, &IFullyConnectedLayer::setNbOutputChannels)
            .def_property("kernel", lambdas::fc_get_kernel, py::cpp_function(&IFullyConnectedLayer::setKernelWeights, py::keep_alive<1, 2>{}))
            .def_property("bias", lambdas::fc_get_bias, py::cpp_function(&IFullyConnectedLayer::setBiasWeights, py::keep_alive<1, 2>{}))
        ;

        // Bind to a Python enum called ActivationType.
        py::enum_<ActivationType>(m, "ActivationType", ActivationTypeDoc::descr)
            .value("RELU", ActivationType::kRELU, ActivationTypeDoc::RELU)
            .value("SIGMOID", ActivationType::kSIGMOID, ActivationTypeDoc::SIGMOID)
            .value("TANH", ActivationType::kTANH, ActivationTypeDoc::TANH)
            .value("LEAKY_RELU", ActivationType::kLEAKY_RELU, ActivationTypeDoc::LEAKY_RELU)
            .value("ELU", ActivationType::kELU, ActivationTypeDoc::ELU)
            .value("SELU", ActivationType::kSELU, ActivationTypeDoc::SELU)
            .value("SOFTSIGN", ActivationType::kSOFTSIGN, ActivationTypeDoc::SOFTSIGN)
            .value("SOFTPLUS", ActivationType::kSOFTPLUS, ActivationTypeDoc::SOFTPLUS)
            .value("CLIP", ActivationType::kCLIP, ActivationTypeDoc::CLIP)
            .value("HARD_SIGMOID", ActivationType::kHARD_SIGMOID, ActivationTypeDoc::HARD_SIGMOID)
            .value("SCALED_TANH", ActivationType::kSCALED_TANH, ActivationTypeDoc::SCALED_TANH)
            .value("THRESHOLDED_RELU", ActivationType::kTHRESHOLDED_RELU, ActivationTypeDoc::THRESHOLDED_RELU)
        ; // ActivationType

        py::class_<IActivationLayer, ILayer, std::unique_ptr<IActivationLayer, py::nodelete>>(m, "IActivationLayer", IActivationLayerDoc::descr)
            .def_property("type", &IActivationLayer::getActivationType, &IActivationLayer::setActivationType)
            .def_property("alpha", &IActivationLayer::getAlpha, &IActivationLayer::setAlpha)
            .def_property("beta", &IActivationLayer::getBeta, &IActivationLayer::setBeta)
        ;

        // Bind to a Python enum called PoolingType.
        py::enum_<PoolingType>(m, "PoolingType", PoolingTypeDoc::descr)
            .value("MAX", PoolingType::kMAX, PoolingTypeDoc::MAX)
            .value("AVERAGE", PoolingType::kAVERAGE, PoolingTypeDoc::AVERAGE)
            .value("MAX_AVERAGE_BLEND", PoolingType::kMAX_AVERAGE_BLEND, PoolingTypeDoc::MAX_AVERAGE_BLEND)
        ; // PoolingType

        py::class_<IPoolingLayer, ILayer, std::unique_ptr<IPoolingLayer, py::nodelete>>(m, "IPoolingLayer", IPoolingLayerDoc::descr)
            .def_property("type", &IPoolingLayer::getPoolingType, &IPoolingLayer::setPoolingType)
            .def_property("window_size", utils::deprecateMember(&IPoolingLayer::getWindowSize, "windnow_size_nd"), utils::deprecateMember(&IPoolingLayer::setWindowSize, "windnow_size_nd"))
            .def_property("stride", utils::deprecateMember(&IPoolingLayer::getStride, "stride_nd"), utils::deprecateMember(&IPoolingLayer::setStride, "stride_nd"))
            .def_property("padding", utils::deprecateMember(&IPoolingLayer::getPadding, "padding_nd"), utils::deprecateMember(&IPoolingLayer::setPadding, "padding_nd"))
            .def_property("pre_padding", &IPoolingLayer::getPrePadding, &IPoolingLayer::setPrePadding)
            .def_property("post_padding", &IPoolingLayer::getPostPadding, &IPoolingLayer::setPostPadding)
            .def_property("padding_mode", &IPoolingLayer::getPaddingMode, &IPoolingLayer::setPaddingMode)
            .def_property("blend_factor", &IPoolingLayer::getBlendFactor, &IPoolingLayer::setBlendFactor)
            .def_property("average_count_excludes_padding", &IPoolingLayer::getAverageCountExcludesPadding, &IPoolingLayer::setAverageCountExcludesPadding)
            .def_property("window_size_nd", &IPoolingLayer::getWindowSizeNd, &IPoolingLayer::setWindowSizeNd)
            .def_property("stride_nd", &IPoolingLayer::getStrideNd, &IPoolingLayer::setStrideNd)
            .def_property("padding_nd", &IPoolingLayer::getPaddingNd, &IPoolingLayer::setPaddingNd)
        ;

        py::class_<ILRNLayer, ILayer, std::unique_ptr<ILRNLayer, py::nodelete>>(m, "ILRNLayer", ILRNLayerDoc::descr)
            .def_property("window_size", &ILRNLayer::getWindowSize, &ILRNLayer::setWindowSize)
            .def_property("alpha", &ILRNLayer::getAlpha, &ILRNLayer::setAlpha)
            .def_property("beta", &ILRNLayer::getBeta, &ILRNLayer::setBeta)
            .def_property("k", &ILRNLayer::getK, &ILRNLayer::setK)
        ;

        // Bind to a Python enum called ScaleMode.
        py::enum_<ScaleMode>(m, "ScaleMode", ScaleModeDoc::descr)
            .value("UNIFORM", ScaleMode::kUNIFORM, ScaleModeDoc::UNIFORM)
            .value("CHANNEL", ScaleMode::kCHANNEL, ScaleModeDoc::CHANNEL)
            .value("ELEMENTWISE", ScaleMode::kELEMENTWISE, ScaleModeDoc::ELEMENTWISE)
        ; // ScaleMode

        py::class_<IScaleLayer, ILayer, std::unique_ptr<IScaleLayer, py::nodelete>>(m, "IScaleLayer", IScaleLayerDoc::descr)
            .def_property("mode", &IScaleLayer::getMode, &IScaleLayer::setMode)
            .def_property("shift", lambdas::scale_get_shift, py::cpp_function(&IScaleLayer::setShift, py::keep_alive<1, 2>{}))
            .def_property("scale", lambdas::scale_get_scale, py::cpp_function(&IScaleLayer::setScale, py::keep_alive<1, 2>{}))
            .def_property("power", lambdas::scale_get_power, py::cpp_function(&IScaleLayer::setPower, py::keep_alive<1, 2>{}))
            .def_property("channel_axis", &IScaleLayer::getChannelAxis, &IScaleLayer::setChannelAxis)
        ;

        py::class_<IQuantizeLayer, ILayer, std::unique_ptr<IQuantizeLayer, py::nodelete>>(m, "IQuantizeLayer", IQuantizeLayerDoc::descr)
            .def_property("axis", &IQuantizeLayer::getAxis, &IQuantizeLayer::setAxis)
        ;

        py::class_<IDequantizeLayer, ILayer, std::unique_ptr<IDequantizeLayer, py::nodelete>>(m, "IDequantizeLayer", IDequantizeLayerDoc::descr)
            .def_property("axis", &IDequantizeLayer::getAxis, &IDequantizeLayer::setAxis)
        ;

        py::class_<ISoftMaxLayer, ILayer, std::unique_ptr<ISoftMaxLayer, py::nodelete>>(m, "ISoftMaxLayer", ISoftMaxLayerDoc::descr)
            .def_property("axes", &ISoftMaxLayer::getAxes, &ISoftMaxLayer::setAxes)
        ;

        py::class_<IConcatenationLayer, ILayer, std::unique_ptr<IConcatenationLayer, py::nodelete>>(m, "IConcatenationLayer", IConcatenationLayerDoc::descr)
            .def_property("axis", &IConcatenationLayer::getAxis, &IConcatenationLayer::setAxis)
        ;

        py::class_<IDeconvolutionLayer, ILayer, std::unique_ptr<IDeconvolutionLayer, py::nodelete>>(m, "IDeconvolutionLayer", IDeconvolutionLayerDoc::descr)
            .def_property("kernel_size", utils::deprecateMember(&IDeconvolutionLayer::getKernelSize, "kernel_size_nd"), utils::deprecateMember(&IDeconvolutionLayer::setKernelSize, "kernel_size_nd"))
            .def_property("stride", utils::deprecateMember(&IDeconvolutionLayer::getStride, "stride_nd"), utils::deprecateMember(&IDeconvolutionLayer::setStride, "stride_nd"))
            .def_property("padding", utils::deprecateMember(&IDeconvolutionLayer::getPadding, "padding_nd"), utils::deprecateMember(&IDeconvolutionLayer::setPadding, "padding_nd"))
            .def_property("num_output_maps", &IDeconvolutionLayer::getNbOutputMaps, &IDeconvolutionLayer::setNbOutputMaps)
            .def_property("pre_padding", &IDeconvolutionLayer::getPrePadding, &IDeconvolutionLayer::setPrePadding)
            .def_property("post_padding", &IDeconvolutionLayer::getPostPadding, &IDeconvolutionLayer::setPostPadding)
            .def_property("padding_mode", &IDeconvolutionLayer::getPaddingMode, &IDeconvolutionLayer::setPaddingMode)
            .def_property("num_groups", &IDeconvolutionLayer::getNbGroups, &IDeconvolutionLayer::setNbGroups)
            .def_property("kernel", lambdas::deconv_get_kernel, py::cpp_function(&IDeconvolutionLayer::setKernelWeights, py::keep_alive<1, 2>{}))
            .def_property("bias", lambdas::deconv_get_bias, py::cpp_function(&IDeconvolutionLayer::setBiasWeights, py::keep_alive<1, 2>{}))
            .def_property("kernel_size_nd", &IDeconvolutionLayer::getKernelSizeNd, &IDeconvolutionLayer::setKernelSizeNd)
            .def_property("stride_nd", &IDeconvolutionLayer::getStrideNd, &IDeconvolutionLayer::setStrideNd)
            .def_property("padding_nd", &IDeconvolutionLayer::getPaddingNd, &IDeconvolutionLayer::setPaddingNd)
            .def_property("dilation_nd", &IDeconvolutionLayer::getDilationNd, &IDeconvolutionLayer::setDilationNd)
        ;

        // Bind to a Python enum called ElementWiseOperation.
        py::enum_<ElementWiseOperation>(m, "ElementWiseOperation", ElementWiseOperationDoc::descr)
            .value("SUM", ElementWiseOperation::kSUM, ElementWiseOperationDoc::SUM)
            .value("PROD", ElementWiseOperation::kPROD, ElementWiseOperationDoc::PROD)
            .value("MAX", ElementWiseOperation::kMAX, ElementWiseOperationDoc::MAX)
            .value("MIN", ElementWiseOperation::kMIN, ElementWiseOperationDoc::MIN)
            .value("SUB", ElementWiseOperation::kSUB, ElementWiseOperationDoc::SUB)
            .value("DIV", ElementWiseOperation::kDIV, ElementWiseOperationDoc::DIV)
            .value("POW", ElementWiseOperation::kPOW, ElementWiseOperationDoc::POW)
            .value("FLOOR_DIV", ElementWiseOperation::kFLOOR_DIV, ElementWiseOperationDoc::FLOOR_DIV)
            .value("AND", ElementWiseOperation::kAND, ElementWiseOperationDoc::AND)
            .value("OR", ElementWiseOperation::kOR, ElementWiseOperationDoc::OR)
            .value("XOR", ElementWiseOperation::kXOR, ElementWiseOperationDoc::XOR)
            .value("EQUAL", ElementWiseOperation::kEQUAL, ElementWiseOperationDoc::EQUAL)
            .value("GREATER", ElementWiseOperation::kGREATER, ElementWiseOperationDoc::GREATER)
            .value("LESS", ElementWiseOperation::kLESS, ElementWiseOperationDoc::LESS)
        ; // ElementWiseOperation

        py::class_<IElementWiseLayer, ILayer, std::unique_ptr<IElementWiseLayer, py::nodelete>>(m, "IElementWiseLayer", IElementWiseLayerDoc::descr)
            .def_property("op", &IElementWiseLayer::getOperation, &IElementWiseLayer::setOperation)
        ;

        // Bind to a Python enum called ScatterMode.
        py::enum_<ScatterMode>(m, "ScatterMode", ScatterModeDoc::descr)
            .value("ELEMENT", ScatterMode::kELEMENT, ScatterModeDoc::ELEMENT)
            .value("ND", ScatterMode::kND, ScatterModeDoc::ND)
        ; // ScatterMode

        py::class_<IScatterLayer, ILayer, std::unique_ptr<IScatterLayer,py::nodelete>>(m, "IScatterLayer", IScatterLayerDoc::descr)
            .def_property("axis", &IScatterLayer::getAxis, &IScatterLayer::setAxis)
            .def_property("mode", &IScatterLayer::getMode, &IScatterLayer::setMode);

        py::class_<IGatherLayer, ILayer, std::unique_ptr<IGatherLayer, py::nodelete>>(m, "IGatherLayer", IGatherLayerDoc::descr)
            .def_property("axis", &IGatherLayer::getGatherAxis, &IGatherLayer::setGatherAxis)
            .def_property("num_elementwise_dims", &IGatherLayer::getNbElementWiseDims, &IGatherLayer::setNbElementWiseDims)
            .def_property("mode", &IGatherLayer::getMode, &IGatherLayer::setMode)
        ;

        py::enum_<GatherMode>(m, "GatherMode", GatherModeDoc::descr)
            .value("DEFAULT", GatherMode::kDEFAULT, GatherModeDoc::DEFAULT)
            .value("ELEMENT", GatherMode::kELEMENT, GatherModeDoc::ELEMENT)
            .value("ND", GatherMode::kND, GatherModeDoc::ND)
        ;

        py::enum_<RNNOperation>(m, "RNNOperation", RNNOperationDoc::descr)
            .value("RELU", RNNOperation::kRELU, RNNOperationDoc::RELU)
            .value("TANH", RNNOperation::kTANH, RNNOperationDoc::TANH)
            .value("LSTM", RNNOperation::kLSTM, RNNOperationDoc::LSTM)
            .value("GRU", RNNOperation::kGRU, RNNOperationDoc::GRU)
        ;

        py::enum_<RNNDirection>(m, "RNNDirection", RNNDirectionDoc::descr)
            .value("UNIDIRECTION", RNNDirection::kUNIDIRECTION, RNNDirectionDoc::UNIDIRECTION)
            .value("BIDIRECTION", RNNDirection::kBIDIRECTION, RNNDirectionDoc::BIDIRECTION)
        ;

        py::enum_<RNNInputMode>(m, "RNNInputMode", RNNInputModeDoc::descr)
            .value("LINEAR", RNNInputMode::kLINEAR, RNNInputModeDoc::LINEAR)
            .value("SKIP", RNNInputMode::kSKIP, RNNInputModeDoc::SKIP)
        ;

        py::enum_<RNNGateType>(m, "RNNGateType", RNNGateTypeDoc::descr)
            .value("INPUT", RNNGateType::kINPUT, RNNGateTypeDoc::INPUT)
            .value("OUTPUT", RNNGateType::kOUTPUT, RNNGateTypeDoc::OUTPUT)
            .value("FORGET", RNNGateType::kFORGET, RNNGateTypeDoc::FORGET)
            .value("UPDATE", RNNGateType::kUPDATE, RNNGateTypeDoc::UPDATE)
            .value("RESET", RNNGateType::kRESET, RNNGateTypeDoc::RESET)
            .value("CELL", RNNGateType::kCELL, RNNGateTypeDoc::CELL)
            .value("HIDDEN", RNNGateType::kHIDDEN, RNNGateTypeDoc::HIDDEN)
        ;

        py::class_<IRNNv2Layer, ILayer, std::unique_ptr<IRNNv2Layer, py::nodelete>>(m, "IRNNv2Layer", IRNNv2LayerDoc::descr)
            .def_property_readonly("num_layers", &IRNNv2Layer::getLayerCount)
            .def_property_readonly("hidden_size", &IRNNv2Layer::getHiddenSize)
            .def_property_readonly("max_seq_length", &IRNNv2Layer::getMaxSeqLength)
            .def_property_readonly("data_length", &IRNNv2Layer::getDataLength)
            .def_property("seq_lengths", &IRNNv2Layer::getSequenceLengths, &IRNNv2Layer::setSequenceLengths)
            .def_property("op", &IRNNv2Layer::getOperation, &IRNNv2Layer::setOperation)
            .def_property("input_mode", &IRNNv2Layer::getInputMode, &IRNNv2Layer::setInputMode)
            .def_property("direction", &IRNNv2Layer::getDirection, &IRNNv2Layer::setDirection)
            .def("set_weights_for_gate", &IRNNv2Layer::setWeightsForGate, "layer_index"_a, "gate"_a, "is_w"_a, "weights"_a, IRNNv2LayerDoc::set_weights_for_gate, py::keep_alive<1, 5>{})
            .def("get_weights_for_gate", lambdas::rnnv2_get_weights, "layer_index"_a, "gate"_a, "is_w"_a, IRNNv2LayerDoc::get_weights_for_gate)
            .def("set_bias_for_gate", &IRNNv2Layer::setBiasForGate, "layer_index"_a, "gate"_a, "is_w"_a, "bias"_a, IRNNv2LayerDoc::set_bias_for_gate, py::keep_alive<1, 5>{})
            .def("get_bias_for_gate", lambdas::rnnv2_get_bias, "layer_index"_a, "gate"_a, "is_w"_a, IRNNv2LayerDoc::get_bias_for_gate)
            .def_property("hidden_state", &IRNNv2Layer::getHiddenState, py::cpp_function(&IRNNv2Layer::setHiddenState, py::keep_alive<1, 2>{}))
            .def_property("cell_state", &IRNNv2Layer::getCellState, py::cpp_function(&IRNNv2Layer::setCellState, py::keep_alive<1, 2>{}))
        ;

        py::class_<IPluginV2Layer, ILayer, std::unique_ptr<IPluginV2Layer, py::nodelete>>(m, "IPluginV2Layer", IPluginV2LayerDoc::descr)
            .def_property_readonly("plugin", &IPluginV2Layer::getPlugin)
        ;

        py::enum_<UnaryOperation>(m, "UnaryOperation", UnaryOperationDoc::descr)
            .value("EXP", UnaryOperation::kEXP, UnaryOperationDoc::EXP)
            .value("LOG", UnaryOperation::kLOG, UnaryOperationDoc::LOG)
            .value("SQRT", UnaryOperation::kSQRT, UnaryOperationDoc::SQRT)
            .value("RECIP", UnaryOperation::kRECIP, UnaryOperationDoc::RECIP)
            .value("ABS", UnaryOperation::kABS, UnaryOperationDoc::ABS)
            .value("NEG", UnaryOperation::kNEG, UnaryOperationDoc::NEG)
            .value("SIN", UnaryOperation::kSIN, UnaryOperationDoc::SIN)
            .value("COS", UnaryOperation::kCOS, UnaryOperationDoc::COS)
            .value("TAN", UnaryOperation::kTAN, UnaryOperationDoc::TAN)
            .value("SINH", UnaryOperation::kSINH, UnaryOperationDoc::SINH)
            .value("COSH", UnaryOperation::kCOSH, UnaryOperationDoc::COSH)
            .value("ASIN", UnaryOperation::kASIN, UnaryOperationDoc::ASIN)
            .value("ACOS", UnaryOperation::kACOS, UnaryOperationDoc::ACOS)
            .value("ATAN", UnaryOperation::kATAN, UnaryOperationDoc::ATAN)
            .value("ASINH", UnaryOperation::kASINH, UnaryOperationDoc::ASINH)
            .value("ACOSH", UnaryOperation::kACOSH, UnaryOperationDoc::ACOSH)
            .value("ATANH", UnaryOperation::kATANH, UnaryOperationDoc::ATANH)
            .value("CEIL", UnaryOperation::kCEIL, UnaryOperationDoc::CEIL)
            .value("FLOOR", UnaryOperation::kFLOOR, UnaryOperationDoc::FLOOR)
            .value("ERF", UnaryOperation::kERF, UnaryOperationDoc::ERF)
            .value("NOT", UnaryOperation::kNOT, UnaryOperationDoc::NOT)
            .value("SIGN", UnaryOperation::kSIGN, UnaryOperationDoc::SIGN)
            .value("ROUND", UnaryOperation::kROUND, UnaryOperationDoc::ROUND)
        ;

        py::class_<IUnaryLayer, ILayer, std::unique_ptr<IUnaryLayer, py::nodelete>>(m, "IUnaryLayer", IUnaryLayerDoc::descr)
            .def_property("op", &IUnaryLayer::getOperation, &IUnaryLayer::setOperation)
        ;

        py::enum_<ReduceOperation>(m, "ReduceOperation", ReduceOperationDoc::descr)
            .value("SUM", ReduceOperation::kSUM, ReduceOperationDoc::SUM)
            .value("PROD", ReduceOperation::kPROD, ReduceOperationDoc::PROD)
            .value("MAX", ReduceOperation::kMAX, ReduceOperationDoc::MAX)
            .value("MIN", ReduceOperation::kMIN, ReduceOperationDoc::MIN)
            .value("AVG", ReduceOperation::kAVG, ReduceOperationDoc::AVG)
        ;

        py::class_<IReduceLayer, ILayer, std::unique_ptr<IReduceLayer, py::nodelete>>(m, "IReduceLayer", IReduceLayerDoc::descr)
            .def_property("op", &IReduceLayer::getOperation, &IReduceLayer::setOperation)
            .def_property("axes", &IReduceLayer::getReduceAxes, &IReduceLayer::setReduceAxes)
            .def_property("keep_dims", &IReduceLayer::getKeepDimensions, &IReduceLayer::setKeepDimensions)
        ;

        py::class_<IPaddingLayer, ILayer, std::unique_ptr<IPaddingLayer, py::nodelete>>(m, "IPaddingLayer", IPaddingLayerDoc::descr)
            .def_property("pre_padding", utils::deprecateMember(&IPaddingLayer::getPrePadding, "pre_padding_nd"), utils::deprecateMember(&IPaddingLayer::setPrePadding, "pre_padding_nd"))
            .def_property("post_padding", utils::deprecateMember(&IPaddingLayer::getPostPadding, "post_padding_nd"), utils::deprecateMember(&IPaddingLayer::setPostPadding, "post_padding_nd"))
            .def_property("pre_padding_nd", &IPaddingLayer::getPrePaddingNd, &IPaddingLayer::setPrePaddingNd)
            .def_property("post_padding_nd", &IPaddingLayer::getPostPaddingNd, &IPaddingLayer::setPostPaddingNd)
        ;

        py::class_<Permutation>(m, "Permutation", PermutationDoc::descr)
            .def(py::init<>())
            .def(py::init(lambdas::permutation_vector_constructor))
            // Allow for string representations (displays like a python tuple).
            .def("__str__", lambdas::permutation_to_str)
            .def("__repr__", lambdas::permutation_to_str)
            // Allows for iteration.
            .def("__getitem__", lambdas::permutation_getter)
            .def("__setitem__", lambdas::permutation_setter)
            .def("__len__", lambdas::permutation_len)
        ;

        // Make it possible to use tuples/lists in Python in place of Permutation.
        py::implicitly_convertible<std::vector<int32_t>, Permutation>();

        py::class_<IShuffleLayer, ILayer, std::unique_ptr<IShuffleLayer, py::nodelete>>(m, "IShuffleLayer", IShuffleLayerDoc::descr)
            .def_property("first_transpose", &IShuffleLayer::getFirstTranspose, &IShuffleLayer::setFirstTranspose)
            .def_property("reshape_dims", &IShuffleLayer::getReshapeDimensions, &IShuffleLayer::setReshapeDimensions)
            .def_property("second_transpose", &IShuffleLayer::getSecondTranspose, &IShuffleLayer::setSecondTranspose)
            .def_property("zero_is_placeholder", &IShuffleLayer::getZeroIsPlaceholder, &IShuffleLayer::setZeroIsPlaceholder)
            .def("set_input", &IShuffleLayer::setInput, "index"_a, "tensor"_a, IShuffleLayerDoc::set_input)
        ;

        py::class_<ISliceLayer, ILayer, std::unique_ptr<ISliceLayer, py::nodelete>>(m, "ISliceLayer", ISliceLayerDoc::descr)
            .def_property("start", &ISliceLayer::getStart, &ISliceLayer::setStart)
            .def_property("shape", &ISliceLayer::getSize, &ISliceLayer::setSize)
            .def_property("stride", &ISliceLayer::getStride, &ISliceLayer::setStride)
            .def_property("mode", &ISliceLayer::getMode, &ISliceLayer::setMode)
            .def("set_input", &ISliceLayer::setInput, "index"_a, "tensor"_a, ISliceLayerDoc::set_input)
        ;

        py::enum_<InterpolationMode>(m, "InterpolationMode", InterpolationModeDoc::descr)
            .value("NEAREST", InterpolationMode::kNEAREST, InterpolationModeDoc::NEAREST)
            .value("LINEAR", InterpolationMode::kLINEAR, InterpolationModeDoc::LINEAR)
            .value("CUBIC", InterpolationMode::kCUBIC, InterpolationModeDoc::CUBIC)
        ;

        py::enum_<SampleMode>(m, "SampleMode", SampleModeDoc::descr)
            .value("STRICT_BOUNDS", SampleMode::kSTRICT_BOUNDS, SampleModeDoc::STRICT_BOUNDS)
            .value("DEFAULT", SampleMode::kDEFAULT, SampleModeDoc::DEFAULT)
            .value("WRAP", SampleMode::kWRAP, SampleModeDoc::WRAP)
            .value("CLAMP", SampleMode::kCLAMP, SampleModeDoc::CLAMP)
            .value("FILL", SampleMode::kFILL, SampleModeDoc::FILL)
            .value("REFLECT", SampleMode::kREFLECT, SampleModeDoc::REFLECT)
        ;

        py::class_<IShapeLayer, ILayer, std::unique_ptr<IShapeLayer, py::nodelete>>(m, "IShapeLayer", IShapeLayerDoc::descr);

        py::enum_<TopKOperation>(m, "TopKOperation", TopKOperationDoc::descr)
            .value("MAX", TopKOperation::kMAX, TopKOperationDoc::MAX)
            .value("MIN", TopKOperation::kMIN, TopKOperationDoc::MIN)
        ;

        py::class_<ITopKLayer, ILayer, std::unique_ptr<ITopKLayer, py::nodelete>>(m, "ITopKLayer", ITopKLayerDoc::descr)
            .def_property("op", &ITopKLayer::getOperation, &ITopKLayer::setOperation)
            .def_property("k", &ITopKLayer::getK, &ITopKLayer::setK)
            .def_property("axes", &ITopKLayer::getReduceAxes, &ITopKLayer::setReduceAxes)
        ;

        py::enum_<MatrixOperation>(m, "MatrixOperation", MatrixOperationDoc::descr)
            .value("NONE", MatrixOperation::kNONE, MatrixOperationDoc::NONE)
            .value("TRANSPOSE", MatrixOperation::kTRANSPOSE, MatrixOperationDoc::TRANSPOSE)
            .value("VECTOR", MatrixOperation::kVECTOR, MatrixOperationDoc::VECTOR)
        ;

        py::class_<IMatrixMultiplyLayer, ILayer, std::unique_ptr<IMatrixMultiplyLayer, py::nodelete>>(m, "IMatrixMultiplyLayer", IMatrixMultiplyLayerDoc::descr)
            .def_property("op0", [](IMatrixMultiplyLayer& self) {return self.getOperation(0);}, [](IMatrixMultiplyLayer& self, MatrixOperation op) {return self.setOperation(0, op);})
            .def_property("op1", [](IMatrixMultiplyLayer& self) {return self.getOperation(1);}, [](IMatrixMultiplyLayer& self, MatrixOperation op) {return self.setOperation(1, op);})
        ;

        py::class_<IRaggedSoftMaxLayer, ILayer, std::unique_ptr<IRaggedSoftMaxLayer, py::nodelete>>(m, "IRaggedSoftMaxLayer", IRaggedSoftMaxLayerDoc::descr);

        py::class_<IIdentityLayer, ILayer, std::unique_ptr<IIdentityLayer, py::nodelete>>(m, "IIdentityLayer", IIdentityLayerDoc::descr);

        py::class_<IConstantLayer, ILayer, std::unique_ptr<IConstantLayer, py::nodelete>>(m, "IConstantLayer", IConstantLayerDoc::descr)
            .def_property("weights", lambdas::constant_get_weights, py::cpp_function(&IConstantLayer::setWeights, py::keep_alive<1, 2>{}))
            .def_property("shape", &IConstantLayer::getDimensions, &IConstantLayer::setDimensions)
        ;

        py::class_<IParametricReLULayer, ILayer, std::unique_ptr<IParametricReLULayer, py::nodelete>>(m, "IParametricReLULayer", IParametricReLULayerDoc::descr);

        py::enum_<ResizeCoordinateTransformation>(m, "ResizeCoordinateTransformation", ResizeCoordinateTransformationDoc::descr)
            .value("ALIGN_CORNERS", ResizeCoordinateTransformation::kALIGN_CORNERS, ResizeCoordinateTransformationDoc::ALIGN_CORNERS)
            .value("ASYMMETRIC", ResizeCoordinateTransformation::kASYMMETRIC, ResizeCoordinateTransformationDoc::ASYMMETRIC)
            .value("HALF_PIXEL", ResizeCoordinateTransformation::kHALF_PIXEL, ResizeCoordinateTransformationDoc::HALF_PIXEL)
        ; // ResizeCoordinateTransformation

        py::enum_<ResizeSelector>(m, "ResizeSelector", ResizeSelectorDoc::descr)
            .value("FORMULA", ResizeSelector::kFORMULA,ResizeSelectorDoc::FORMULA)
            .value("UPPER", ResizeSelector::kUPPER, ResizeSelectorDoc::UPPER)
        ; // ResizeSelector

        py::enum_<ResizeRoundMode>(m, "ResizeRoundMode", ResizeRoundModeDoc::descr)
            .value("HALF_UP", ResizeRoundMode::kHALF_UP,ResizeRoundModeDoc::HALF_UP)
            .value("HALF_DOWN", ResizeRoundMode::kHALF_DOWN, ResizeRoundModeDoc::HALF_DOWN)
            .value("FLOOR", ResizeRoundMode::kFLOOR,ResizeRoundModeDoc::FLOOR)
            .value("CEIL", ResizeRoundMode::kCEIL, ResizeRoundModeDoc::CEIL)
        ; // ResizeRoundMode

        py::class_<IResizeLayer, ILayer, std::unique_ptr<IResizeLayer, py::nodelete>>(m, "IResizeLayer", IResizeLayerDoc::descr)
            .def_property("shape", &IResizeLayer::getOutputDimensions, &IResizeLayer::setOutputDimensions)
            .def_property("scales", lambdas::resize_get_scales, lambdas::resize_set_scales)
            .def_property("resize_mode", &IResizeLayer::getResizeMode, &IResizeLayer::setResizeMode)
            .def_property("coordinate_transformation", &IResizeLayer::getCoordinateTransformation, &IResizeLayer::setCoordinateTransformation)
            .def_property("selector_for_single_pixel", &IResizeLayer::getSelectorForSinglePixel, &IResizeLayer::setSelectorForSinglePixel)
            .def_property("nearest_rounding", &IResizeLayer::getNearestRounding, &IResizeLayer::setNearestRounding)
            .def_property("exclude_outside", &IResizeLayer::getExcludeOutside, &IResizeLayer::setExcludeOutside)
            .def_property("cubic_coeff", &IResizeLayer::getCubicCoeff, &IResizeLayer::setCubicCoeff)
            .def("set_input", &IResizeLayer::setInput, "index"_a, "tensor"_a, IResizeLayerDoc::set_input)
        ;

        py::enum_<LoopOutput>(m, "LoopOutput", LoopOutputDoc::descr)
            .value("LAST_VALUE", LoopOutput::kLAST_VALUE, LoopOutputDoc::LAST_VALUE)
            .value("CONCATENATE", LoopOutput::kCONCATENATE, LoopOutputDoc::CONCATENATE)
            .value("REVERSE", LoopOutput::kREVERSE, LoopOutputDoc::REVERSE)
        ;

        py::enum_<TripLimit>(m, "TripLimit", TripLimitDoc::descr)
            .value("COUNT", TripLimit::kCOUNT, TripLimitDoc::COUNT)
            .value("WHILE", TripLimit::kWHILE, TripLimitDoc::WHILE)
        ;

        py::class_<ILoopBoundaryLayer, ILayer, std::unique_ptr<ILoopBoundaryLayer, py::nodelete>>(m, "ILoopBoundaryLayer", ILoopBoundaryLayerDoc::descr)
            .def_property_readonly("loop", &ILoopBoundaryLayer::getLoop)
        ;

        py::class_<IRecurrenceLayer, ILoopBoundaryLayer, std::unique_ptr<IRecurrenceLayer, py::nodelete>>(m, "IRecurrenceLayer", IRecurrenceLayerDoc::descr)
            .def("set_input", &IRecurrenceLayer::setInput, "index"_a, "tensor"_a, IRecurrenceLayerDoc::set_input)
        ;

        py::class_<ILoopOutputLayer, ILoopBoundaryLayer, std::unique_ptr<ILoopOutputLayer, py::nodelete>>(m, "ILoopOutputLayer", ILoopOutputLayerDoc::descr)
            .def("set_input", &ILoopOutputLayer::setInput, "index"_a, "tensor"_a, ILoopOutputLayerDoc::set_input)
            .def_property("axis", &ILoopOutputLayer::getAxis, &ILoopOutputLayer::setAxis)
            .def_property_readonly("kind", &ILoopOutputLayer::getLoopOutput)
        ;

        py::class_<ITripLimitLayer, ILoopBoundaryLayer, std::unique_ptr<ITripLimitLayer, py::nodelete>>(m, "ITripLimitLayer", ITripLimitLayerDoc::descr)
            .def_property_readonly("kind", &ITripLimitLayer::getTripLimit)
        ;

        py::class_<IIteratorLayer, ILoopBoundaryLayer, std::unique_ptr<IIteratorLayer, py::nodelete>>(m, "IIteratorLayer", IIteratorLayerDoc::descr)
            .def_property("axis", &IIteratorLayer::getAxis, &IIteratorLayer::setAxis)
            .def_property("reverse", &IIteratorLayer::getReverse, &IIteratorLayer::setReverse)
        ;

        py::class_<ILoop, std::unique_ptr<ILoop, py::nodelete>>(m, "ILoop", ILoopDoc::descr)
            .def("add_recurrence", &ILoop::addRecurrence, "initial_value"_a, ILoopDoc::add_recurrence)
            .def("add_trip_limit", &ILoop::addTripLimit, "tensor"_a, "kind"_a, ILoopDoc::add_trip_limit)
            .def("add_iterator", &ILoop::addIterator, "tensor"_a, "axis"_a = 0, "reverse"_a = false, ILoopDoc::add_iterator)
            .def("add_loop_output", &ILoop::addLoopOutput, "tensor"_a, "kind"_a, "axis"_a = 0, ILoopDoc::add_loop_output)
            .def_property("name", &ILoop::getName, &ILoop::setName)
        ;

        py::class_<ISelectLayer, ILayer, std::unique_ptr<ISelectLayer, py::nodelete>>(m, "ISelectLayer", ISelectLayerDoc::descr)
        ;

        py::class_<IAssertionLayer, ILayer, std::unique_ptr<IAssertionLayer, py::nodelete>>(m, "IAssertionLayer", IAssertionLayerDoc::descr)
            .def_property("message", &IAssertionLayer::getMessage, &IAssertionLayer::setMessage);
        ;

        py::class_<IGridSampleLayer, ILayer, std::unique_ptr<IGridSampleLayer, py::nodelete>>(m, "IGridSampleLayer", IGridSampleLayerDoc::descr)
            .def_property("interpolation_mode", &IGridSampleLayer::getInterpolationMode, &IGridSampleLayer::setInterpolationMode)
            .def_property("align_corners", &IGridSampleLayer::getAlignCorners, &IGridSampleLayer::setAlignCorners)
            .def_property("sample_mode", &IGridSampleLayer::getSampleMode, &IGridSampleLayer::setSampleMode)
        ;

        py::enum_<BoundingBoxFormat>(m, "BoundingBoxFormat", BoundingBoxFormatDoc::descr)
            .value("CORNER_PAIRS", BoundingBoxFormat::kCORNER_PAIRS, BoundingBoxFormatDoc::CORNER_PAIRS)
            .value("CENTER_SIZES", BoundingBoxFormat::kCENTER_SIZES, BoundingBoxFormatDoc::CENTER_SIZES)
        ;

        py::class_<INMSLayer, ILayer, std::unique_ptr<INMSLayer, py::nodelete>>(m, "INMSLayer", INMSLayerDoc::descr)
            .def_property("bounding_box_format", &INMSLayer::getBoundingBoxFormat, &INMSLayer::setBoundingBoxFormat)
            .def_property("topk_box_limit", &INMSLayer::getTopKBoxLimit, &INMSLayer::setTopKBoxLimit)
            .def("set_input", &INMSLayer::setInput, "index"_a, "tensor"_a, INMSLayerDoc::set_input)
        ;

        py::enum_<FillOperation>(m, "FillOperation", FillOperationDoc::descr)
            .value("LINSPACE", FillOperation::kLINSPACE, FillOperationDoc::LINSPACE)
            .value("RANDOM_UNIFORM", FillOperation::kRANDOM_UNIFORM, FillOperationDoc::RANDOM_UNIFORM)
            .value("RANDOM_NORMAL", FillOperation::kRANDOM_NORMAL, FillOperationDoc::RANDOM_NORMAL)
        ; // FillOperation

        py::class_<IFillLayer, ILayer, std::unique_ptr<IFillLayer, py::nodelete>>(m, "IFillLayer", IFillLayerDoc::descr)
            .def_property("shape", &IFillLayer::getDimensions, &IFillLayer::setDimensions)
            .def_property("operation", &IFillLayer::getOperation, &IFillLayer::setOperation)
            .def_property("alpha", &IFillLayer::getAlpha, &IFillLayer::setAlpha)
            .def_property("beta", &IFillLayer::getBeta, &IFillLayer::setBeta)
            .def("set_input", &IFillLayer::setInput, "index"_a, "tensor"_a, IFillLayerDoc::set_input)
        ;

        py::class_<IIfConditionalBoundaryLayer, ILayer, std::unique_ptr<IIfConditionalBoundaryLayer, py::nodelete>>(m, "IIfConditionalBoundaryLayer", IIfConditionalBoundaryLayerDoc::descr)
            .def_property_readonly("conditional", &IIfConditionalBoundaryLayer::getConditional)
        ;

        py::class_<IIfConditionalOutputLayer, IIfConditionalBoundaryLayer, std::unique_ptr<IIfConditionalOutputLayer, py::nodelete>>(m, "IIfConditionalOutputLayer", IIfConditionalOutputLayerDoc::descr)
        ;

        py::class_<IIfConditionalInputLayer, IIfConditionalBoundaryLayer, std::unique_ptr<IIfConditionalInputLayer, py::nodelete>>(m, "IIfConditionalInputLayer", IIfConditionalInputLayerDoc::descr)
        ;

        py::class_<IConditionLayer, IIfConditionalBoundaryLayer, std::unique_ptr<IConditionLayer, py::nodelete>>(m, "IConditionLayer", IConditionLayerDoc::descr)
        ;

        py::class_<IIfConditional, std::unique_ptr<IIfConditional, py::nodelete>>(m, "IIfConditional", IIfConditionalDoc::descr)
            .def("set_condition", &IIfConditional::setCondition, "condition"_a, IIfConditionalDoc::set_condition)
            .def("add_output", &IIfConditional::addOutput, "true_subgraph_output"_a, "false_subgraph_output"_a, IIfConditionalDoc::add_output)
            .def("add_input", &IIfConditional::addInput, "input"_a, IIfConditionalDoc::add_input)
            .def_property("name", &IIfConditional::getName, &IIfConditional::setName)
        ;

        py::class_<IEinsumLayer, ILayer, std::unique_ptr<IEinsumLayer, py::nodelete>>(m, "IEinsumLayer", IEinsumLayerDoc::descr)
            .def_property("equation", &IEinsumLayer::getEquation, &IEinsumLayer::setEquation)
        ;

        py::class_<IOneHotLayer, ILayer, std::unique_ptr<IOneHotLayer,py::nodelete>>(m, "IOneHotLayer", IOneHotLayerDoc::descr)
            .def_property("axis", &IOneHotLayer::getAxis, &IOneHotLayer::setAxis)
        ;

        py::class_<INonZeroLayer, ILayer, std::unique_ptr<INonZeroLayer,py::nodelete>>(m, "INonZeroLayer", INonZeroLayerDoc::descr)
        ;

        // Weights must be kept alive for the duration of the network. py::keep_alive is critical here!
        // Additionally, we use reference_internal so that pybind11 does not free layers when they go out of scope.
        py::class_<INetworkDefinition>(m, "INetworkDefinition", INetworkDefinitionDoc::descr)
            .def_property("name", &INetworkDefinition::getName, &INetworkDefinition::setName)
            .def_property_readonly("num_layers", &INetworkDefinition::getNbLayers)
            .def_property_readonly("num_inputs", &INetworkDefinition::getNbInputs)
            .def_property_readonly("num_outputs", &INetworkDefinition::getNbOutputs)
            .def_property_readonly("has_implicit_batch_dimension", &INetworkDefinition::hasImplicitBatchDimension)
            .def_property_readonly("has_explicit_precision", &INetworkDefinition::hasExplicitPrecision)
            .def_property("error_recorder", &INetworkDefinition::getErrorRecorder,
                py::cpp_function(&INetworkDefinition::setErrorRecorder, py::keep_alive<1, 2>{}))
            .def("mark_output", &INetworkDefinition::markOutput, "tensor"_a, INetworkDefinitionDoc::mark_output)
            // Layers
            .def("add_input", &INetworkDefinition::addInput, "name"_a, "dtype"_a, "shape"_a,
                INetworkDefinitionDoc::add_input,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_convolution", utils::deprecate(lambdas::add_convolution, "add_convolution_nd"), "input"_a, "num_output_maps"_a, "kernel_shape"_a,
                "kernel"_a, "bias"_a=nullptr, py::keep_alive<1, 5>{}, py::keep_alive<1, 6>{}, INetworkDefinitionDoc::add_convolution,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_convolution_nd", lambdas::add_convolution_nd, "input"_a, "num_output_maps"_a,
                "kernel_shape"_a, "kernel"_a, "bias"_a=nullptr, py::keep_alive<1, 5>{}, py::keep_alive<1, 6>{},
                INetworkDefinitionDoc::add_convolution_nd,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_fully_connected", utils::deprecate(lambdas::add_fully_connected, "add_matrix_multiply"), "input"_a, "num_outputs"_a,
                "kernel"_a, "bias"_a=nullptr, py::keep_alive<1, 4>{}, py::keep_alive<1, 5>{}, INetworkDefinitionDoc::add_fully_connected,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_activation", &INetworkDefinition::addActivation, "input"_a, "type"_a,
                INetworkDefinitionDoc::add_activation,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_pooling", utils::deprecateMember(&INetworkDefinition::addPooling, "add_pooling_nd"), "input"_a, "type"_a, "window_size"_a,
                INetworkDefinitionDoc::add_pooling,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_pooling_nd", &INetworkDefinition::addPoolingNd, "input"_a, "type"_a, "window_size"_a,
                INetworkDefinitionDoc::add_pooling_nd,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_lrn", &INetworkDefinition::addLRN, "input"_a, "window"_a, "alpha"_a, "beta"_a, "k"_a,
                INetworkDefinitionDoc::add_lrn,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_scale", lambdas::add_scale, "input"_a, "mode"_a, "shift"_a=nullptr, "scale"_a=nullptr, "power"_a=nullptr,
                py::keep_alive<1, 4>{}, py::keep_alive<1, 5>{}, py::keep_alive<1, 6>{}, INetworkDefinitionDoc::add_scale,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_scale_nd", lambdas::add_scale_nd, "input"_a, "mode"_a, "shift"_a=nullptr, "scale"_a=nullptr, "power"_a=nullptr, "channel_axis"_a,
                py::keep_alive<1, 4>{}, py::keep_alive<1, 5>{}, py::keep_alive<1, 6>{}, INetworkDefinitionDoc::add_scale_nd,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_softmax", &INetworkDefinition::addSoftMax, "input"_a, INetworkDefinitionDoc::add_softmax,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_concatenation", lambdas::add_concatenation, "inputs"_a, INetworkDefinitionDoc::add_concatenation,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_deconvolution", utils::deprecate(lambdas::add_deconvolution, "add_deconvolution_nd"), "input"_a, "num_output_maps"_a,
                "kernel_shape"_a, "kernel"_a, "bias"_a=nullptr, py::keep_alive<1, 5>{}, py::keep_alive<1, 6>{},
                INetworkDefinitionDoc::add_deconvolution,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_deconvolution_nd", lambdas::add_deconvolution_nd, "input"_a, "num_output_maps"_a,
                "kernel_shape"_a, "kernel"_a, "bias"_a=nullptr, py::keep_alive<1, 5>{}, py::keep_alive<1, 6>{},
                INetworkDefinitionDoc::add_deconvolution_nd,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_elementwise", &INetworkDefinition::addElementWise, "input1"_a, "input2"_a, "op"_a,
                INetworkDefinitionDoc::add_elementwise,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_unary", &INetworkDefinition::addUnary, "input"_a, "op"_a, INetworkDefinitionDoc::add_unary,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_padding", utils::deprecateMember(&INetworkDefinition::addPadding, "add_padding_nd"), "input"_a, "pre_padding"_a, "post_padding"_a,
                INetworkDefinitionDoc::add_padding,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_padding_nd", &INetworkDefinition::addPaddingNd, "input"_a, "pre_padding"_a, "post_padding"_a,
                INetworkDefinitionDoc::add_padding_nd,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_shuffle", &INetworkDefinition::addShuffle, "input"_a, INetworkDefinitionDoc::add_shuffle,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_slice", &INetworkDefinition::addSlice, "input"_a, "start"_a, "shape"_a, "stride"_a,
                INetworkDefinitionDoc::add_slice,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_reduce", &INetworkDefinition::addReduce, "input"_a, "op"_a, "axes"_a, "keep_dims"_a,
                INetworkDefinitionDoc::add_reduce,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_topk", &INetworkDefinition::addTopK, "input"_a, "op"_a, "k"_a, "axes"_a,
                INetworkDefinitionDoc::add_topk,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_gather", &INetworkDefinition::addGather, "input"_a, "indices"_a, "axis"_a,
                INetworkDefinitionDoc::add_gather,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_scatter", &INetworkDefinition::addScatter, "data"_a, "indices"_a, "updates"_a, "mode"_a,
                INetworkDefinitionDoc::add_scatter,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_gather_v2", &INetworkDefinition::addGatherV2, "input"_a, "indices"_a, "mode"_a,
                INetworkDefinitionDoc::add_gather_v2,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_ragged_softmax", &INetworkDefinition::addRaggedSoftMax, "input"_a, "bounds"_a,
                INetworkDefinitionDoc::add_ragged_softmax,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_matrix_multiply",
                static_cast<IMatrixMultiplyLayer* (INetworkDefinition::*)(ITensor&, MatrixOperation, ITensor&, MatrixOperation)>(&INetworkDefinition::addMatrixMultiply),
                "input0"_a, "op0"_a, "input1"_a, "op1"_a, INetworkDefinitionDoc::add_matrix_multiply,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_constant", &INetworkDefinition::addConstant, "shape"_a, "weights"_a,
                py::keep_alive<1, 3>{}, INetworkDefinitionDoc::add_constant,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_rnn_v2", utils::deprecateMember(&INetworkDefinition::addRNNv2, "addLoop"), "input"_a, "layer_count"_a,
                "hidden_size"_a, "max_seq_length"_a, "op"_a,
                py::keep_alive<1, 0>{}, INetworkDefinitionDoc::add_rnn_v2)
            .def("add_identity", &INetworkDefinition::addIdentity, "input"_a,
                INetworkDefinitionDoc::add_identity,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_plugin_v2",  lambdas::add_plugin_v2, "inputs"_a, "plugin"_a,
                INetworkDefinitionDoc::add_plugin_v2,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_parametric_relu", &INetworkDefinition::addParametricReLU, "input"_a,
                "slopes"_a, INetworkDefinitionDoc::add_parametric_relu,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_resize", &INetworkDefinition::addResize, "input"_a, INetworkDefinitionDoc::add_resize,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_loop", &INetworkDefinition::addLoop, INetworkDefinitionDoc::add_loop,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_shape", &INetworkDefinition::addShape, "input"_a, INetworkDefinitionDoc::add_shape,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_select", &INetworkDefinition::addSelect, "condition"_a, "then_input"_a,
                "else_input"_a, INetworkDefinitionDoc::add_select,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_assertion", &INetworkDefinition::addAssertion, "condition"_a, "message"_a,
                 INetworkDefinitionDoc::add_assertion, INetworkDefinitionDoc::add_assertion,
                 py::return_value_policy::reference_internal)
            .def("add_grid_sample", &INetworkDefinition::addGridSample, "input"_a, "grid"_a,
                  INetworkDefinitionDoc::add_grid_sample, py::return_value_policy::reference_internal)
            .def("add_nms", &INetworkDefinition::addNMS, "boxes"_a,
                "scores"_a, "max_output_boxes_per_class"_a, INetworkDefinitionDoc::add_nms,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_fill", &INetworkDefinition::addFill, "shape"_a, "op"_a, INetworkDefinitionDoc::add_fill)
            .def("add_quantize",  &INetworkDefinition::addQuantize, "input"_a, "scale"_a,
                INetworkDefinitionDoc::add_quantize,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_dequantize", &INetworkDefinition::addDequantize, "input"_a, "scale"_a,
                INetworkDefinitionDoc::add_dequantize,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_if_conditional", &INetworkDefinition::addIfConditional, INetworkDefinitionDoc::add_if_conditional,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_einsum", lambdas::add_einsum, "inputs"_a, "equation"_a, INetworkDefinitionDoc::add_einsum,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_one_hot", &INetworkDefinition::addOneHot, "indices"_a, "values"_a, "depth"_a, "axis"_a,
                INetworkDefinitionDoc::add_one_hot,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("add_non_zero", &INetworkDefinition::addNonZero, "input"_a, INetworkDefinitionDoc::add_non_zero,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("remove_tensor", &INetworkDefinition::removeTensor, "tensor"_a, INetworkDefinitionDoc::remove_tensor)
            .def("unmark_output", &INetworkDefinition::unmarkOutput, "tensor"_a, INetworkDefinitionDoc::unmark_output)
            .def("mark_output_for_shapes", &INetworkDefinition::markOutputForShapes, "tensor"_a, INetworkDefinitionDoc::mark_output_for_shapes)
            .def("unmark_output_for_shapes", &INetworkDefinition::unmarkOutputForShapes, "tensor"_a, INetworkDefinitionDoc::unmark_output_for_shapes)
            .def("set_weights_name", &INetworkDefinition::setWeightsName, "weights"_a, "name"_a, INetworkDefinitionDoc::set_weights_name)
            // Getters
            .def("get_layer", &INetworkDefinition::getLayer, "index"_a, INetworkDefinitionDoc::get_layer,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("get_input", &INetworkDefinition::getInput, "index"_a, INetworkDefinitionDoc::get_input,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("get_output", &INetworkDefinition::getOutput, "index"_a, INetworkDefinitionDoc::get_output,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
#if ENABLE_INETWORK_SERIALIZE
            // Serialization
            .def("serialize", lambdas::network_serialize, INetworkDefinitionDoc::serialize)
#endif
            // Allow iteration over the layers of a network
            .def("__len__", &INetworkDefinition::getNbLayers)
            .def("__getitem__", lambdas::network_getitem, py::return_value_policy::reference_internal,
                py::keep_alive<1, 0>{}, py::return_value_policy::reference_internal)
            .def("__del__", &utils::doNothingDel<INetworkDefinition>)
        ;

        //Aliasing deprecated enums
        m.attr("ResizeMode") = m.attr("InterpolationMode");
        m.attr("SliceMode") = m.attr("SampleMode");
    }
} /* tensorrt */
