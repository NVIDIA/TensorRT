/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef NV_INFER_IMPL_H
#define NV_INFER_IMPL_H

#include "NvInferLegacyDims.h"
#include "NvInferRuntimeCommon.h"

// @cond SuppressDoxyWarnings

namespace nvinfer1
{

class IActivationLayer;
class IAlgorithm;
class IAlgorithmContext;
class IAlgorithmIOInfo;
class IAlgorithmSelector;
class IAlgorithmVariant;
class IAssertionLayer;
class IBuilderConfig;
class IConcatenationLayer;
class IIfConditional;
class IConditionLayer;
class IIfConditionalOutputLayer;
class IIfConditionalInputLayer;
class IConstantLayer;
class IConvolutionLayer;
class ICudaEngine;
class IDeconvolutionLayer;
class IDequantizeLayer;
class IDimensionExpr;
class IEinsumLayer;
class IElementWiseLayer;
class IExecutionContext;
class IFillLayer;
class IFullyConnectedLayer;
class IGatherLayer;
class IHostMemory;
class IIdentityLayer;
class IIfConditional;
class IInt8Calibrator;
class IIteratorLayer;
class ILayer;
class ILoop;
class ILoopOutputLayer;
class ILRNLayer;
class IMatrixMultiplyLayer;
class INetworkDefinition;
class IOptimizationProfile;
class IPaddingLayer;
class IParametricReLULayer;
class IPlugin;
class IPluginExt;
class IPluginFactory;
class IPluginLayer;
class IPluginV2Layer;
class IPoolingLayer;
class IProfiler;
class IQuantizeLayer;
class IRaggedSoftMaxLayer;
class IRecurrenceLayer;
class IReduceLayer;
class IResizeLayer;
class IRNNv2Layer;
class IScaleLayer;
class IScatterLayer;
class ISelectLayer;
class IShapeLayer;
class IShuffleLayer;
class ISliceLayer;
class ISoftMaxLayer;
class IEngineInspector;
class ITensor;
class ITimingCache;
class ITopKLayer;
class ITripLimitLayer;
class IUnaryLayer;
struct Permutation;
class Weights;

enum class ActivationType : int32_t;
enum class BuilderFlag : int32_t;
enum class CalibrationAlgoType : int32_t;
enum class DeviceType : int32_t;
enum class DimensionOperation : int32_t;
enum class ElementWiseOperation : int32_t;
enum class EngineCapability : int32_t;
enum class FillOperation : int32_t;
enum class GatherMode : int32_t;
enum class LayerInformationFormat : int32_t;
enum class LayerType : int32_t;
enum class LoopOutput : int32_t;
enum class MatrixOperation : int32_t;
enum class MemoryPoolType : int32_t;
enum class NetworkDefinitionCreationFlag : int32_t;
enum class OptProfileSelector : int32_t;
enum class PaddingMode : int32_t;
enum class PoolingType : int32_t;
enum class ProfilingVerbosity : int32_t;
enum class QuantizationFlag : int32_t;
enum class ReduceOperation : int32_t;
enum class ResizeCoordinateTransformation : int32_t;
enum class ResizeMode : int32_t;
enum class ResizeRoundMode : int32_t;
enum class ResizeSelector : int32_t;
enum class RNNDirection : int32_t;
enum class RNNGateType : int32_t;
enum class RNNInputMode : int32_t;
enum class RNNOperation : int32_t;
enum class ScaleMode : int32_t;
enum class ScatterMode : int32_t;
enum class SliceMode : int32_t;
enum class TensorLocation : int32_t;
enum class TopKOperation : int32_t;
enum class TripLimit : int32_t;
enum class UnaryOperation : int32_t;
enum class WeightsRole : int32_t;

using TacticSources = uint32_t;
using TensorFormats = uint32_t;
using BuilderFlags = uint32_t;
using NetworkDefinitionCreationFlags = uint32_t;
using QuantizationFlags = uint32_t;

//!
//! \file NvInferImpl.h
//!
//! This file contains definitions for API methods that cross the shared library boundary. These
//! methods must not be called directly by applications; they should only be called through the
//! API classes.
//!

namespace apiv
{

class VRoot
{
public:
    virtual ~VRoot() noexcept = default;
};

class VHostMemory : public VRoot
{
public:
    virtual void* data() const noexcept = 0;
    virtual std::size_t size() const noexcept = 0;
    virtual DataType type() const noexcept = 0;
};

class VDimensionExpr : public VRoot
{
public:
    virtual bool isConstant() const = 0;
    virtual int32_t getConstantValue() const = 0;
};

class VExprBuilder : public VRoot
{
public:
    virtual IDimensionExpr const* constant(int32_t value) = 0;
    virtual IDimensionExpr const* operation(
        DimensionOperation op, IDimensionExpr const& first, IDimensionExpr const& second)
        = 0;
};

class VRuntime : public VRoot
{
public:
    virtual nvinfer1::ICudaEngine* deserializeCudaEngine(
        void const* blob, std::size_t size, IPluginFactory* pluginFactory) noexcept = 0;
    virtual void setDLACore(int32_t dlaCore) noexcept = 0;
    virtual int32_t getDLACore() const noexcept = 0;
    virtual int32_t getNbDLACores() const noexcept = 0;
    virtual void setGpuAllocator(IGpuAllocator* allocator) noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual ILogger* getLogger() const noexcept = 0;
    virtual bool setMaxThreads(int32_t maxThreads) noexcept = 0;
    virtual int32_t getMaxThreads() const noexcept = 0;
};

class VRefitter : public VRoot
{
public:
    virtual bool setWeights(char const* layerName, WeightsRole role, const Weights weights) noexcept = 0;
    virtual bool refitCudaEngine() noexcept = 0;
    virtual int32_t getMissing(int32_t size, char const** layerNames, WeightsRole* roles) noexcept = 0;
    virtual int32_t getAll(int32_t size, char const** layerNames, WeightsRole* roles) noexcept = 0;
    virtual bool setDynamicRange(char const* tensorName, float min, float max) noexcept = 0;
    virtual float getDynamicRangeMin(char const* tensorName) const noexcept = 0;
    virtual float getDynamicRangeMax(char const* tensorName) const noexcept = 0;
    virtual int32_t getTensorsWithDynamicRange(int32_t size, char const** tensorNames) const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual bool setNamedWeights(char const* name, Weights weights) noexcept = 0;
    virtual int32_t getMissingWeights(int32_t size, char const** weightsNames) noexcept = 0;
    virtual int32_t getAllWeights(int32_t size, char const** weightsNames) noexcept = 0;
    virtual ILogger* getLogger() const noexcept = 0;
    virtual bool setMaxThreads(int32_t maxThreads) noexcept = 0;
    virtual int32_t getMaxThreads() const noexcept = 0;
};

class VOptimizationProfile : public VRoot
{
public:
    virtual bool setDimensions(char const* inputName, OptProfileSelector select, Dims dims) noexcept = 0;
    virtual Dims getDimensions(char const* inputName, OptProfileSelector select) const noexcept = 0;
    virtual bool setShapeValues(
        char const* inputName, OptProfileSelector select, int32_t const* values, int32_t nbValues) noexcept = 0;
    virtual int32_t getNbShapeValues(char const* inputName) const noexcept = 0;
    virtual int32_t const* getShapeValues(char const* inputName, OptProfileSelector select) const noexcept = 0;
    virtual bool setExtraMemoryTarget(float target) noexcept = 0;
    virtual float getExtraMemoryTarget() const noexcept = 0;
    virtual bool isValid() const noexcept = 0;
};

class VCudaEngine : public VRoot
{
public:
    virtual int32_t getNbBindings() const noexcept = 0;
    virtual int32_t getBindingIndex(char const* name) const noexcept = 0;
    virtual char const* getBindingName(int32_t bindingIndex) const noexcept = 0;
    virtual bool bindingIsInput(int32_t bindingIndex) const noexcept = 0;
    virtual Dims getBindingDimensions(int32_t bindingIndex) const noexcept = 0;
    virtual DataType getBindingDataType(int32_t bindingIndex) const noexcept = 0;
    virtual int32_t getMaxBatchSize() const noexcept = 0;
    virtual int32_t getNbLayers() const noexcept = 0;
    virtual IHostMemory* serialize() const noexcept = 0;
    virtual IExecutionContext* createExecutionContext() noexcept = 0;
    virtual TensorLocation getLocation(int32_t bindingIndex) const noexcept = 0;
    virtual IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept = 0;
    virtual size_t getDeviceMemorySize() const noexcept = 0;
    virtual bool isRefittable() const noexcept = 0;
    virtual int32_t getBindingBytesPerComponent(int32_t bindingIndex) const noexcept = 0;
    virtual int32_t getBindingComponentsPerElement(int32_t bindingIndex) const noexcept = 0;
    virtual TensorFormat getBindingFormat(int32_t bindingIndex) const noexcept = 0;
    virtual char const* getBindingFormatDesc(int32_t bindingIndex) const noexcept = 0;
    virtual int32_t getBindingVectorizedDim(int32_t bindingIndex) const noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual int32_t getNbOptimizationProfiles() const noexcept = 0;
    virtual Dims getProfileDimensions(
        int32_t bindingIndex, int32_t profileIndex, OptProfileSelector select) const noexcept = 0;
    virtual int32_t const* getProfileShapeValues(
        int32_t profileIndex, int32_t inputIndex, OptProfileSelector select) const noexcept = 0;
    virtual bool isShapeBinding(int32_t bindingIndex) const noexcept = 0;
    virtual bool isExecutionBinding(int32_t bindingIndex) const noexcept = 0;
    virtual EngineCapability getEngineCapability() const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual bool hasImplicitBatchDimension() const noexcept = 0;
    virtual TacticSources getTacticSources() const noexcept = 0;
    virtual ProfilingVerbosity getProfilingVerbosity() const noexcept = 0;
    virtual IEngineInspector* createEngineInspector() const noexcept = 0;
};

class VExecutionContext : public VRoot
{
public:
    virtual bool execute(int32_t batchSize, void* const* bindings) noexcept = 0;
    virtual bool enqueue(
        int32_t batchSize, void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept
        = 0;
    virtual void setDebugSync(bool sync) noexcept = 0;
    virtual bool getDebugSync() const noexcept = 0;
    virtual void setProfiler(IProfiler*) noexcept = 0;
    virtual IProfiler* getProfiler() const noexcept = 0;
    virtual ICudaEngine const& getEngine() const noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual void setDeviceMemory(void* memory) noexcept = 0;
    virtual Dims getStrides(int32_t bindingIndex) const noexcept = 0;
    virtual bool setOptimizationProfile(int32_t profileIndex) noexcept = 0;
    virtual int32_t getOptimizationProfile() const noexcept = 0;
    virtual bool setBindingDimensions(int32_t bindingIndex, Dims dimensions) noexcept = 0;
    virtual Dims getBindingDimensions(int32_t bindingIndex) const noexcept = 0;
    virtual bool setInputShapeBinding(int32_t bindingIndex, int32_t const* data) noexcept = 0;
    virtual bool getShapeBinding(int32_t bindingIndex, int32_t* data) const noexcept = 0;
    virtual bool allInputDimensionsSpecified() const noexcept = 0;
    virtual bool allInputShapesSpecified() const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual bool executeV2(void* const* bindings) noexcept = 0;
    virtual bool enqueueV2(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed) noexcept = 0;
    virtual bool setOptimizationProfileAsync(int32_t profileIndex, cudaStream_t stream) noexcept = 0;
    virtual void setEnqueueEmitsProfile(bool enqueueEmitsProfile) noexcept = 0;
    virtual bool getEnqueueEmitsProfile() const noexcept = 0;
    virtual bool reportToProfiler() const noexcept = 0;
};

class VEngineInspector : public VRoot
{
public:
    virtual bool setExecutionContext(IExecutionContext const* context) noexcept = 0;
    virtual IExecutionContext const* getExecutionContext() const noexcept = 0;
    virtual AsciiChar const* getLayerInformation(int32_t layerIndex, LayerInformationFormat format) const noexcept = 0;
    virtual AsciiChar const* getEngineInformation(LayerInformationFormat format) const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
};

class VTensor : public VRoot
{
public:
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual void setDimensions(Dims dimensions) noexcept = 0;
    virtual Dims getDimensions() const noexcept = 0;
    virtual void setType(DataType type) noexcept = 0;
    virtual DataType getType() const noexcept = 0;
    virtual bool setDynamicRange(float min, float max) noexcept = 0;
    virtual bool isNetworkInput() const noexcept = 0;
    virtual bool isNetworkOutput() const noexcept = 0;
    virtual void setBroadcastAcrossBatch(bool broadcastAcrossBatch) noexcept = 0;
    virtual bool getBroadcastAcrossBatch() const noexcept = 0;
    virtual TensorLocation getLocation() const noexcept = 0;
    virtual void setLocation(TensorLocation location) noexcept = 0;
    virtual bool dynamicRangeIsSet() const noexcept = 0;
    virtual void resetDynamicRange() noexcept = 0;
    virtual float getDynamicRangeMin() const noexcept = 0;
    virtual float getDynamicRangeMax() const noexcept = 0;
    virtual void setAllowedFormats(TensorFormats formats) noexcept = 0;
    virtual TensorFormats getAllowedFormats() const noexcept = 0;
    virtual bool isShapeTensor() const noexcept = 0;
    virtual bool isExecutionTensor() const noexcept = 0;
};
class VLayer : public VRoot
{
public:
    virtual LayerType getType() const noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual int32_t getNbInputs() const noexcept = 0;
    virtual ITensor* getInput(int32_t index) const noexcept = 0;
    virtual int32_t getNbOutputs() const noexcept = 0;
    virtual ITensor* getOutput(int32_t index) const noexcept = 0;
    virtual void setInput(int32_t index, ITensor& tensor) noexcept = 0;
    virtual void setPrecision(DataType dataType) noexcept = 0;
    virtual DataType getPrecision() const noexcept = 0;
    virtual bool precisionIsSet() const noexcept = 0;
    virtual void resetPrecision() noexcept = 0;
    virtual void setOutputType(int32_t index, DataType dataType) noexcept = 0;
    virtual DataType getOutputType(int32_t index) const noexcept = 0;
    virtual bool outputTypeIsSet(int32_t index) const noexcept = 0;
    virtual void resetOutputType(int32_t index) noexcept = 0;
};

class VConvolutionLayer : public VRoot
{
public:
    virtual void setKernelSize(DimsHW kernelSize) noexcept = 0;
    virtual DimsHW getKernelSize() const noexcept = 0;
    virtual void setNbOutputMaps(int32_t nbOutputMaps) noexcept = 0;
    virtual int32_t getNbOutputMaps() const noexcept = 0;
    virtual void setStride(DimsHW stride) noexcept = 0;
    virtual DimsHW getStride() const noexcept = 0;
    virtual void setPadding(DimsHW padding) noexcept = 0;
    virtual DimsHW getPadding() const noexcept = 0;
    virtual void setNbGroups(int32_t nbGroups) noexcept = 0;
    virtual int32_t getNbGroups() const noexcept = 0;
    virtual void setKernelWeights(Weights weights) noexcept = 0;
    virtual Weights getKernelWeights() const noexcept = 0;
    virtual void setBiasWeights(Weights weights) noexcept = 0;
    virtual Weights getBiasWeights() const noexcept = 0;
    virtual void setDilation(DimsHW dilation) noexcept = 0;
    virtual DimsHW getDilation() const noexcept = 0;
    virtual void setPrePadding(Dims padding) noexcept = 0;
    virtual Dims getPrePadding() const noexcept = 0;
    virtual void setPostPadding(Dims padding) noexcept = 0;
    virtual Dims getPostPadding() const noexcept = 0;
    virtual void setPaddingMode(PaddingMode paddingMode) noexcept = 0;
    virtual PaddingMode getPaddingMode() const noexcept = 0;
    virtual void setKernelSizeNd(Dims kernelSize) noexcept = 0;
    virtual Dims getKernelSizeNd() const noexcept = 0;
    virtual void setStrideNd(Dims stride) noexcept = 0;
    virtual Dims getStrideNd() const noexcept = 0;
    virtual void setPaddingNd(Dims padding) noexcept = 0;
    virtual Dims getPaddingNd() const noexcept = 0;
    virtual void setDilationNd(Dims dilation) noexcept = 0;
    virtual Dims getDilationNd() const noexcept = 0;
};

class VFullyConnectedLayer : public VRoot
{
public:
    virtual void setNbOutputChannels(int32_t nbOutputs) noexcept = 0;
    virtual int32_t getNbOutputChannels() const noexcept = 0;
    virtual void setKernelWeights(Weights weights) noexcept = 0;
    virtual Weights getKernelWeights() const noexcept = 0;
    virtual void setBiasWeights(Weights weights) noexcept = 0;
    virtual Weights getBiasWeights() const noexcept = 0;
};

class VActivationLayer : public VRoot
{
public:
    virtual void setActivationType(ActivationType type) noexcept = 0;
    virtual ActivationType getActivationType() const noexcept = 0;
    virtual void setAlpha(float alpha) noexcept = 0;
    virtual void setBeta(float beta) noexcept = 0;
    virtual float getAlpha() const noexcept = 0;
    virtual float getBeta() const noexcept = 0;
};

class VPoolingLayer : public VRoot
{
public:
    virtual void setPoolingType(PoolingType type) noexcept = 0;
    virtual PoolingType getPoolingType() const noexcept = 0;
    virtual void setWindowSize(DimsHW windowSize) noexcept = 0;
    virtual DimsHW getWindowSize() const noexcept = 0;
    virtual void setStride(DimsHW stride) noexcept = 0;
    virtual DimsHW getStride() const noexcept = 0;
    virtual void setPadding(DimsHW padding) noexcept = 0;
    virtual DimsHW getPadding() const noexcept = 0;
    virtual void setBlendFactor(float blendFactor) noexcept = 0;
    virtual float getBlendFactor() const noexcept = 0;
    virtual void setAverageCountExcludesPadding(bool exclusive) noexcept = 0;
    virtual bool getAverageCountExcludesPadding() const noexcept = 0;
    virtual void setPrePadding(Dims padding) noexcept = 0;
    virtual Dims getPrePadding() const noexcept = 0;
    virtual void setPostPadding(Dims padding) noexcept = 0;
    virtual Dims getPostPadding() const noexcept = 0;
    virtual void setPaddingMode(PaddingMode paddingMode) noexcept = 0;
    virtual PaddingMode getPaddingMode() const noexcept = 0;
    virtual void setWindowSizeNd(Dims windowSize) noexcept = 0;
    virtual Dims getWindowSizeNd() const noexcept = 0;
    virtual void setStrideNd(Dims stride) noexcept = 0;
    virtual Dims getStrideNd() const noexcept = 0;
    virtual void setPaddingNd(Dims padding) noexcept = 0;
    virtual Dims getPaddingNd() const noexcept = 0;
};

class VLRNLayer : public VRoot
{
public:
    virtual void setWindowSize(int32_t windowSize) noexcept = 0;
    virtual int32_t getWindowSize() const noexcept = 0;
    virtual void setAlpha(float alpha) noexcept = 0;
    virtual float getAlpha() const noexcept = 0;
    virtual void setBeta(float beta) noexcept = 0;
    virtual float getBeta() const noexcept = 0;
    virtual void setK(float k) noexcept = 0;
    virtual float getK() const noexcept = 0;
};

class VScaleLayer : public VRoot
{
public:
    virtual void setMode(ScaleMode mode) noexcept = 0;
    virtual ScaleMode getMode() const noexcept = 0;
    virtual void setShift(Weights shift) noexcept = 0;
    virtual Weights getShift() const noexcept = 0;
    virtual void setScale(Weights scale) noexcept = 0;
    virtual Weights getScale() const noexcept = 0;
    virtual void setPower(Weights power) noexcept = 0;
    virtual Weights getPower() const noexcept = 0;
    virtual int32_t getChannelAxis() const noexcept = 0;
    virtual void setChannelAxis(int32_t channelAxis) noexcept = 0;
};

class VSoftMaxLayer : public VRoot
{
public:
    virtual void setAxes(uint32_t axes) noexcept = 0;
    virtual uint32_t getAxes() const noexcept = 0;
};

class VConcatenationLayer : public VRoot
{
public:
    virtual void setAxis(int32_t axis) noexcept = 0;
    virtual int32_t getAxis() const noexcept = 0;
};

class VDeconvolutionLayer : public VRoot
{
public:
    virtual void setKernelSize(DimsHW kernelSize) noexcept = 0;
    virtual DimsHW getKernelSize() const noexcept = 0;
    virtual void setNbOutputMaps(int32_t nbOutputMaps) noexcept = 0;
    virtual int32_t getNbOutputMaps() const noexcept = 0;
    virtual void setStride(DimsHW stride) noexcept = 0;
    virtual DimsHW getStride() const noexcept = 0;
    virtual void setPadding(DimsHW padding) noexcept = 0;
    virtual DimsHW getPadding() const noexcept = 0;
    virtual void setNbGroups(int32_t nbGroups) noexcept = 0;
    virtual int32_t getNbGroups() const noexcept = 0;
    virtual void setKernelWeights(Weights weights) noexcept = 0;
    virtual Weights getKernelWeights() const noexcept = 0;
    virtual void setBiasWeights(Weights weights) noexcept = 0;
    virtual Weights getBiasWeights() const noexcept = 0;
    virtual void setPrePadding(Dims padding) noexcept = 0;
    virtual Dims getPrePadding() const noexcept = 0;
    virtual void setPostPadding(Dims padding) noexcept = 0;
    virtual Dims getPostPadding() const noexcept = 0;
    virtual void setPaddingMode(PaddingMode paddingMode) noexcept = 0;
    virtual PaddingMode getPaddingMode() const noexcept = 0;
    virtual void setKernelSizeNd(Dims kernelSize) noexcept = 0;
    virtual Dims getKernelSizeNd() const noexcept = 0;
    virtual void setStrideNd(Dims stride) noexcept = 0;
    virtual Dims getStrideNd() const noexcept = 0;
    virtual void setPaddingNd(Dims padding) noexcept = 0;
    virtual Dims getPaddingNd() const noexcept = 0;
    virtual void setDilationNd(Dims dilation) noexcept = 0;
    virtual Dims getDilationNd() const noexcept = 0;
};

class VElementWiseLayer : public VRoot
{
public:
    virtual void setOperation(ElementWiseOperation op) noexcept = 0;
    virtual ElementWiseOperation getOperation() const noexcept = 0;
};

class VGatherLayer : public VRoot
{
public:
    virtual void setGatherAxis(int32_t axis) noexcept = 0;
    virtual int32_t getGatherAxis() const noexcept = 0;
    virtual void setNbElementWiseDims(int32_t k) noexcept = 0;
    virtual int32_t getNbElementWiseDims() const noexcept = 0;
    virtual void setMode(GatherMode mode) noexcept = 0;
    virtual GatherMode getMode() const noexcept = 0;
};

class VRNNv2Layer : public VRoot
{
public:
    virtual int32_t getLayerCount() const noexcept = 0;
    virtual int32_t getHiddenSize() const noexcept = 0;
    virtual int32_t getMaxSeqLength() const noexcept = 0;
    virtual int32_t getDataLength() const noexcept = 0;
    virtual void setSequenceLengths(ITensor& seqLengths) noexcept = 0;
    virtual ITensor* getSequenceLengths() const noexcept = 0;
    virtual void setOperation(RNNOperation op) noexcept = 0;
    virtual RNNOperation getOperation() const noexcept = 0;
    virtual void setInputMode(RNNInputMode op) noexcept = 0;
    virtual RNNInputMode getInputMode() const noexcept = 0;
    virtual void setDirection(RNNDirection op) noexcept = 0;
    virtual RNNDirection getDirection() const noexcept = 0;
    virtual void setWeightsForGate(int32_t layerIndex, RNNGateType gate, bool isW, Weights weights) noexcept = 0;
    virtual Weights getWeightsForGate(int32_t layerIndex, RNNGateType gate, bool isW) const noexcept = 0;
    virtual void setBiasForGate(int32_t layerIndex, RNNGateType gate, bool isW, Weights bias) noexcept = 0;
    virtual Weights getBiasForGate(int32_t layerIndex, RNNGateType gate, bool isW) const noexcept = 0;
    virtual void setHiddenState(ITensor& hidden) noexcept = 0;
    virtual ITensor* getHiddenState() const noexcept = 0;
    virtual void setCellState(ITensor& cell) noexcept = 0;
    virtual ITensor* getCellState() const noexcept = 0;
};

class VPluginLayer : public VRoot
{
public:
    virtual IPlugin& getPlugin() noexcept = 0;
};

class VPluginV2Layer : public VRoot
{
public:
    virtual IPluginV2& getPlugin() noexcept = 0;
};

class VUnaryLayer : public VRoot
{
public:
    virtual void setOperation(UnaryOperation op) noexcept = 0;
    virtual UnaryOperation getOperation() const noexcept = 0;
};

class VReduceLayer : public VRoot
{
public:
    virtual void setOperation(ReduceOperation op) noexcept = 0;
    virtual ReduceOperation getOperation() const noexcept = 0;
    virtual void setReduceAxes(uint32_t reduceAxes) noexcept = 0;
    virtual uint32_t getReduceAxes() const noexcept = 0;
    virtual void setKeepDimensions(bool keepDimensions) noexcept = 0;
    virtual bool getKeepDimensions() const noexcept = 0;
};

class VPaddingLayer : public VRoot
{
public:
    virtual void setPrePadding(DimsHW padding) noexcept = 0;
    virtual DimsHW getPrePadding() const noexcept = 0;
    virtual void setPostPadding(DimsHW padding) noexcept = 0;
    virtual DimsHW getPostPadding() const noexcept = 0;
    virtual void setPrePaddingNd(Dims padding) noexcept = 0;
    virtual Dims getPrePaddingNd() const noexcept = 0;
    virtual void setPostPaddingNd(Dims padding) noexcept = 0;
    virtual Dims getPostPaddingNd() const noexcept = 0;
};

class VShuffleLayer : public VRoot
{
public:
    virtual void setFirstTranspose(Permutation const& permutation) noexcept = 0;
    virtual Permutation const& getFirstTranspose() const noexcept = 0;
    virtual void setReshapeDimensions(Dims dimensions) noexcept = 0;
    virtual Dims getReshapeDimensions() const noexcept = 0;
    virtual void setSecondTranspose(Permutation const& permutation) noexcept = 0;
    virtual Permutation const& getSecondTranspose() const noexcept = 0;
    virtual void setZeroIsPlaceholder(bool zeroIsPlaceholder) = 0;
    virtual bool getZeroIsPlaceholder() const = 0;
};

class VSliceLayer : public VRoot
{
public:
    virtual void setStart(Dims start) noexcept = 0;
    virtual Dims getStart() const noexcept = 0;
    virtual void setSize(Dims size) noexcept = 0;
    virtual Dims getSize() const noexcept = 0;
    virtual void setStride(Dims stride) noexcept = 0;
    virtual Dims getStride() const noexcept = 0;
    virtual void setMode(SliceMode mode) noexcept = 0;
    virtual SliceMode getMode() const noexcept = 0;
};

class VShapeLayer : public VRoot
{
public:
};

class VTopKLayer : public VRoot
{
public:
    virtual void setOperation(TopKOperation op) noexcept = 0;
    virtual TopKOperation getOperation() const noexcept = 0;
    virtual void setK(int32_t k) noexcept = 0;
    virtual int32_t getK() const noexcept = 0;
    virtual void setReduceAxes(uint32_t reduceAxes) noexcept = 0;
    virtual uint32_t getReduceAxes() const noexcept = 0;
};

class VMatrixMultiplyLayer : public VRoot
{
public:
    virtual void setOperation(int32_t index, MatrixOperation op) noexcept = 0;
    virtual MatrixOperation getOperation(int32_t index) const noexcept = 0;
};

class VRaggedSoftMaxLayer : public VRoot
{
public:
};

class VIdentityLayer : public VRoot
{
public:
};

class VConstantLayer : public VRoot
{
public:
    virtual void setWeights(Weights weights) noexcept = 0;
    virtual Weights getWeights() const noexcept = 0;
    virtual void setDimensions(Dims dimensions) noexcept = 0;
    virtual Dims getDimensions() const noexcept = 0;
};

class VParametricReLULayer : public VRoot
{
public:
};

class VResizeLayer : public VRoot
{
public:
    virtual void setOutputDimensions(Dims dimensions) noexcept = 0;
    virtual Dims getOutputDimensions() const noexcept = 0;
    virtual void setScales(float const* scales, int32_t nbScales) noexcept = 0;
    virtual int32_t getScales(int32_t size, float* scales) const noexcept = 0;
    virtual void setResizeMode(ResizeMode resizeMode) noexcept = 0;
    virtual ResizeMode getResizeMode() const noexcept = 0;
    virtual void setAlignCorners(bool alignCorners) noexcept = 0;
    virtual bool getAlignCorners() const noexcept = 0;
    virtual void setCoordinateTransformation(ResizeCoordinateTransformation coordTransform) noexcept = 0;
    virtual ResizeCoordinateTransformation getCoordinateTransformation() const noexcept = 0;
    virtual void setSelectorForSinglePixel(ResizeSelector selector) noexcept = 0;
    virtual ResizeSelector getSelectorForSinglePixel() const noexcept = 0;
    virtual void setNearestRounding(ResizeRoundMode value) noexcept = 0;
    virtual ResizeRoundMode getNearestRounding() const noexcept = 0;
};

class VLoopBoundaryLayer : public VRoot
{
public:
    virtual ILoop* getLoop() const noexcept = 0;
};

class VRecurrenceLayer : public VRoot
{
public:
};

class VLoopOutputLayer : public VRoot
{
public:
    virtual LoopOutput getLoopOutput() const noexcept = 0;
    virtual void setAxis(int32_t axis) noexcept = 0;
    virtual int32_t getAxis() const noexcept = 0;
};

class VTripLimitLayer : public VRoot
{
public:
    virtual TripLimit getTripLimit() const noexcept = 0;
};

class VIteratorLayer : public VRoot
{
public:
    virtual void setAxis(int32_t axis) noexcept = 0;
    virtual int32_t getAxis() const noexcept = 0;
    virtual void setReverse(bool reverse) noexcept = 0;
    virtual bool getReverse() const noexcept = 0;
};
class VLoop : public VRoot
{
public:
    virtual IRecurrenceLayer* addRecurrence(ITensor& initialValue) noexcept = 0;
    virtual ITripLimitLayer* addTripLimit(ITensor& tensor, TripLimit limit) noexcept = 0;
    virtual IIteratorLayer* addIterator(ITensor& tensor, int32_t axis = 0, bool reverse = false) noexcept = 0;
    virtual ILoopOutputLayer* addLoopOutput(ITensor& tensor, LoopOutput outputKind, int32_t axis = 0) noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
};

class VConditionalBoundaryLayer : public VRoot
{
public:
    virtual IIfConditional* getConditional() const noexcept = 0;
};

class VConditionLayer : public VRoot
{
public:
};

class VConditionalInputLayer : public VRoot
{
public:
};

class VConditionalOutputLayer : public VRoot
{
public:
};

class VIfConditional : public VRoot
{
public:
    virtual IConditionLayer* setCondition(ITensor& tensor) noexcept = 0;
    virtual IIfConditionalInputLayer* addInput(ITensor& tensor) noexcept = 0;
    virtual IIfConditionalOutputLayer* addOutput(ITensor& trueTensor, ITensor& falseTensor) noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
};

class VSelectLayer : public VRoot
{
};

class VAssertionLayer : public VRoot
{
public:
    virtual void setMessage(char const* message) noexcept = 0;
    virtual char const* getMessage() const noexcept = 0;
};

class VFillLayer : public VRoot
{
public:
    virtual void setDimensions(Dims dimensions) noexcept = 0;
    virtual Dims getDimensions() const noexcept = 0;
    virtual void setOperation(FillOperation op) noexcept = 0;
    virtual FillOperation getOperation() const noexcept = 0;
    virtual void setAlpha(double alpha) noexcept = 0;
    virtual double getAlpha() const noexcept = 0;
    virtual void setBeta(double beta) noexcept = 0;
    virtual double getBeta() const noexcept = 0;
};

class VQuantizeLayer : public VRoot
{
public:
    virtual int32_t getAxis() const noexcept = 0;
    virtual void setAxis(int32_t axis) noexcept = 0;
};

class VDequantizeLayer : public VRoot
{
public:
    virtual int32_t getAxis() const noexcept = 0;
    virtual void setAxis(int32_t axis) noexcept = 0;
};

class VScatterLayer : public VRoot
{
public:
   virtual void setMode(ScatterMode mode) noexcept = 0;
   virtual ScatterMode getMode() const noexcept = 0;
   virtual void setAxis(int32_t axis) noexcept = 0;
   virtual int32_t getAxis() const noexcept = 0;
}; // class VScatterLayer

class VEinsumLayer : public VRoot
{
public:
    virtual bool setEquation(char const* equation) noexcept = 0;
    virtual char const* getEquation() const noexcept = 0;
};

class VNetworkDefinition : public VRoot
{
public:
    virtual ITensor* addInput(char const* name, DataType type, Dims dimensions) noexcept = 0;
    virtual void markOutput(ITensor& tensor) noexcept = 0;
    virtual IConvolutionLayer* addConvolution(ITensor& input, int32_t nbOutputMaps, DimsHW kernelSize,
        Weights kernelWeights, Weights biasWeights) noexcept = 0;
    virtual IFullyConnectedLayer* addFullyConnected(
        ITensor& input, int32_t nbOutputs, Weights kernelWeights, Weights biasWeights) noexcept
        = 0;
    virtual IActivationLayer* addActivation(ITensor& input, ActivationType type) noexcept = 0;
    virtual IPoolingLayer* addPooling(ITensor& input, PoolingType type, DimsHW windowSize) noexcept = 0;
    virtual ILRNLayer* addLRN(ITensor& input, int32_t window, float alpha, float beta, float k) noexcept = 0;
    virtual IScaleLayer* addScale(ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) noexcept
        = 0;
    virtual ISoftMaxLayer* addSoftMax(ITensor& input) noexcept = 0;
    virtual IConcatenationLayer* addConcatenation(ITensor* const* inputs, int32_t nbInputs) noexcept = 0;
    virtual IDeconvolutionLayer* addDeconvolution(
        ITensor& input, int32_t nbOutputMaps, DimsHW kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
        = 0;
    virtual IElementWiseLayer* addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) noexcept = 0;
    virtual IUnaryLayer* addUnary(ITensor& input, UnaryOperation operation) noexcept = 0;
    virtual IPaddingLayer* addPadding(ITensor& input, DimsHW prePadding, DimsHW postPadding) noexcept = 0;
    virtual IShuffleLayer* addShuffle(ITensor& input) noexcept = 0;
    virtual int32_t getNbLayers() const noexcept = 0;
    virtual ILayer* getLayer(int32_t index) const noexcept = 0;
    virtual int32_t getNbInputs() const noexcept = 0;
    virtual ITensor* getInput(int32_t index) const noexcept = 0;
    virtual int32_t getNbOutputs() const noexcept = 0;
    virtual ITensor* getOutput(int32_t index) const noexcept = 0;
    virtual IReduceLayer* addReduce(
        ITensor& input, ReduceOperation operation, uint32_t reduceAxes, bool keepDimensions) noexcept
        = 0;
    virtual ITopKLayer* addTopK(ITensor& input, TopKOperation op, int32_t k, uint32_t reduceAxes) noexcept = 0;
    virtual IGatherLayer* addGather(ITensor& data, ITensor& indices, int32_t axis) noexcept = 0;
    virtual IRaggedSoftMaxLayer* addRaggedSoftMax(ITensor& input, ITensor& bounds) noexcept = 0;
    virtual IMatrixMultiplyLayer* addMatrixMultiply(
        ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1) noexcept
        = 0;
    virtual IConstantLayer* addConstant(Dims dimensions, Weights weights) noexcept = 0;
    virtual IRNNv2Layer* addRNNv2(
        ITensor& input, int32_t layerCount, int32_t hiddenSize, int32_t maxSeqLen, RNNOperation op) noexcept
        = 0;
    virtual IIdentityLayer* addIdentity(ITensor& input) noexcept = 0;
    virtual void removeTensor(ITensor& tensor) noexcept = 0;
    virtual void unmarkOutput(ITensor& tensor) noexcept = 0;
    virtual IPluginV2Layer* addPluginV2(ITensor* const* inputs, int32_t nbInputs, IPluginV2& plugin) noexcept = 0;
    virtual ISliceLayer* addSlice(ITensor& input, Dims start, Dims size, Dims stride) noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual IShapeLayer* addShape(ITensor& input) noexcept = 0;
    virtual bool hasImplicitBatchDimension() const noexcept = 0;
    virtual bool markOutputForShapes(ITensor& tensor) noexcept = 0;
    virtual bool unmarkOutputForShapes(ITensor& tensor) noexcept = 0;
    virtual IParametricReLULayer* addParametricReLU(ITensor& input, ITensor& slope) noexcept = 0;
    virtual IConvolutionLayer* addConvolutionNd(
        ITensor& input, int32_t nbOutputMaps, Dims kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
        = 0;
    virtual IPoolingLayer* addPoolingNd(ITensor& input, PoolingType type, Dims windowSize) noexcept = 0;
    virtual IDeconvolutionLayer* addDeconvolutionNd(
        ITensor& input, int32_t nbOutputMaps, Dims kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
        = 0;
    virtual IScaleLayer* addScaleNd(
        ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power, int32_t channelAxis) noexcept
        = 0;
    virtual IResizeLayer* addResize(ITensor& input) noexcept = 0;
    virtual bool hasExplicitPrecision() const noexcept = 0;
    virtual ILoop* addLoop() noexcept = 0;
    virtual ISelectLayer* addSelect(ITensor& condition, ITensor& thenInput, ITensor& elseInput) noexcept = 0;
    virtual IFillLayer* addFill(Dims dimensions, FillOperation op) noexcept = 0;
    virtual IPaddingLayer* addPaddingNd(ITensor& input, Dims prePadding, Dims postPadding) noexcept = 0;
    virtual bool setWeightsName(Weights weights, char const* name) noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual IDequantizeLayer* addDequantize(ITensor& input, ITensor& scale) noexcept = 0;
    virtual IQuantizeLayer* addQuantize(ITensor& input, ITensor& scale) noexcept = 0;
    virtual IGatherLayer* addGatherV2(ITensor& data, ITensor& indices, GatherMode mode) noexcept = 0;
    virtual IIfConditional* addIfConditional() noexcept = 0;
    virtual IScatterLayer* addScatter(ITensor& data, ITensor& indices, ITensor& updates, ScatterMode mode) noexcept = 0;
    virtual IEinsumLayer* addEinsum(ITensor* const* inputs, int32_t nbInputs, char const* equation) noexcept = 0;
    virtual IAssertionLayer* addAssertion(ITensor& condition, char const* message) noexcept = 0;
};

class VAlgorithmIOInfo : public VRoot
{
public:
    virtual TensorFormat getTensorFormat() const noexcept = 0;
    virtual DataType getDataType() const noexcept = 0;
    virtual Dims getStrides() const noexcept = 0;
};

class VAlgorithmVariant : public VRoot
{
public:
    virtual int64_t getImplementation() const noexcept = 0;
    virtual int64_t getTactic() const noexcept = 0;
};

class VAlgorithmContext : public VRoot
{
public:
    virtual char const* getName() const noexcept = 0;
    virtual Dims getDimensions(int32_t index, OptProfileSelector select) const noexcept = 0;
    virtual int32_t getNbInputs() const noexcept = 0;
    virtual int32_t getNbOutputs() const noexcept = 0;
};

class VAlgorithm : public VRoot
{
public:
    virtual IAlgorithmIOInfo const& getAlgorithmIOInfo(int32_t index) const noexcept = 0;
    virtual IAlgorithmVariant const& getAlgorithmVariant() const noexcept = 0;
    virtual float getTimingMSec() const noexcept = 0;
    virtual std::size_t getWorkspaceSize() const noexcept = 0;
    virtual IAlgorithmIOInfo const* getAlgorithmIOInfoByIndex(int32_t index) const noexcept = 0;
};

class VTimingCache : public VRoot
{
public:
    virtual nvinfer1::IHostMemory* serialize() const noexcept = 0;
    virtual bool combine(ITimingCache const& inputCache, bool ignoreMismatch) noexcept = 0;
    virtual bool reset() noexcept = 0;
};

class VBuilderConfig : public VRoot
{
public:
    virtual void setMinTimingIterations(int32_t minTiming) noexcept = 0;
    virtual int32_t getMinTimingIterations() const noexcept = 0;
    virtual void setAvgTimingIterations(int32_t avgTiming) noexcept = 0;
    virtual int32_t getAvgTimingIterations() const noexcept = 0;
    virtual void setEngineCapability(EngineCapability capability) noexcept = 0;
    virtual EngineCapability getEngineCapability() const noexcept = 0;
    virtual void setInt8Calibrator(IInt8Calibrator* calibrator) noexcept = 0;
    virtual IInt8Calibrator* getInt8Calibrator() const noexcept = 0;
    virtual void setMaxWorkspaceSize(std::size_t workspaceSize) noexcept = 0;
    virtual std::size_t getMaxWorkspaceSize() const noexcept = 0;
    virtual void setFlags(BuilderFlags builderFlags) noexcept = 0;
    virtual BuilderFlags getFlags() const noexcept = 0;
    virtual void clearFlag(BuilderFlag builderFlag) noexcept = 0;
    virtual void setFlag(BuilderFlag builderFlag) noexcept = 0;
    virtual bool getFlag(BuilderFlag builderFlag) const noexcept = 0;
    virtual void setDeviceType(ILayer const* layer, DeviceType deviceType) noexcept = 0;
    virtual DeviceType getDeviceType(ILayer const* layer) const noexcept = 0;
    virtual bool isDeviceTypeSet(ILayer const* layer) const noexcept = 0;
    virtual void resetDeviceType(ILayer const* layer) noexcept = 0;
    virtual bool canRunOnDLA(ILayer const* layer) const noexcept = 0;
    virtual void setDLACore(int32_t dlaCore) noexcept = 0;
    virtual int32_t getDLACore() const noexcept = 0;
    virtual void setDefaultDeviceType(DeviceType deviceType) noexcept = 0;
    virtual DeviceType getDefaultDeviceType() const noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual void setProfileStream(const cudaStream_t stream) noexcept = 0;
    virtual cudaStream_t getProfileStream() const noexcept = 0;
    virtual int32_t addOptimizationProfile(IOptimizationProfile const* profile) noexcept = 0;
    virtual int32_t getNbOptimizationProfiles() const noexcept = 0;
    virtual void setProfilingVerbosity(ProfilingVerbosity verbosity) noexcept = 0;
    virtual ProfilingVerbosity getProfilingVerbosity() const noexcept = 0;
    virtual void setAlgorithmSelector(IAlgorithmSelector* selector) noexcept = 0;
    virtual IAlgorithmSelector* getAlgorithmSelector() const noexcept = 0;
    virtual bool setCalibrationProfile(IOptimizationProfile const* profile) noexcept = 0;
    virtual IOptimizationProfile const* getCalibrationProfile() noexcept = 0;
    virtual void setQuantizationFlags(QuantizationFlags flags) noexcept = 0;
    virtual QuantizationFlags getQuantizationFlags() const noexcept = 0;
    virtual void clearQuantizationFlag(QuantizationFlag flag) noexcept = 0;
    virtual void setQuantizationFlag(QuantizationFlag flag) noexcept = 0;
    virtual bool getQuantizationFlag(QuantizationFlag flag) const noexcept = 0;
    virtual bool setTacticSources(TacticSources tacticSources) noexcept = 0;
    virtual TacticSources getTacticSources() const noexcept = 0;
    virtual nvinfer1::ITimingCache* createTimingCache(void const* blob, std::size_t size) const noexcept = 0;
    virtual bool setTimingCache(ITimingCache const& cache, bool ignoreMismatch) noexcept = 0;
    virtual nvinfer1::ITimingCache const* getTimingCache() const noexcept = 0;
    virtual void setMemoryPoolLimit(MemoryPoolType pool, std::size_t poolSize) noexcept = 0;
    virtual std::size_t getMemoryPoolLimit(MemoryPoolType pool) const noexcept = 0;
};

class VBuilder : public VRoot
{
public:
    virtual void setMaxBatchSize(int32_t batchSize) noexcept = 0;
    virtual int32_t getMaxBatchSize() const noexcept = 0;
    virtual bool platformHasFastFp16() const noexcept = 0;
    virtual bool platformHasFastInt8() const noexcept = 0;
    virtual int32_t getMaxDLABatchSize() const noexcept = 0;
    virtual int32_t getNbDLACores() const noexcept = 0;
    virtual void setGpuAllocator(IGpuAllocator* allocator) noexcept = 0;
    virtual nvinfer1::IBuilderConfig* createBuilderConfig() noexcept = 0;
    virtual nvinfer1::ICudaEngine* buildEngineWithConfig(INetworkDefinition& network, IBuilderConfig& config) noexcept
        = 0;
    virtual nvinfer1::INetworkDefinition* createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept = 0;
    virtual nvinfer1::IOptimizationProfile* createOptimizationProfile() noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual bool platformHasTf32() const noexcept = 0;
    virtual nvinfer1::IHostMemory* buildSerializedNetwork(INetworkDefinition& network, IBuilderConfig& config) noexcept
        = 0;
    virtual bool isNetworkSupported(INetworkDefinition const& network, IBuilderConfig const& config) const noexcept = 0;
    virtual ILogger* getLogger() const noexcept = 0;
    virtual bool setMaxThreads(int32_t maxThreads) noexcept = 0;
    virtual int32_t getMaxThreads() const noexcept = 0;
};

} // namespace apiv
} // namespace nvinfer1

// @endcond

#endif // NV_INFER_RUNTIME_IMPL_H
