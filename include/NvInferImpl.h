/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NV_INFER_IMPL_H
#define NV_INFER_IMPL_H

#include "NvInferLegacyDims.h"
#include "NvInferRuntimeCommon.h"

// @cond SuppressDoxyWarnings

namespace nvinfer1
{

class ILogger;

namespace v_1_0
{
class IProgressMonitor;
}
using IProgressMonitor = v_1_0::IProgressMonitor;

namespace v_1_0
{
class IAlgorithmSelector;
}
using IAlgorithmSelector = v_1_0::IAlgorithmSelector;

namespace v_1_0
{
class IProfiler;
}
using IProfiler = v_1_0::IProfiler;

namespace v_1_0
{
class IOutputAllocator;
}
using IOutputAllocator = v_1_0::IOutputAllocator;

namespace v_1_0
{
class IDebugListener;
}
using IDebugListener = v_1_0::IDebugListener;

class IActivationLayer;
class IAlgorithm;
class IAlgorithmContext;
class IAlgorithmIOInfo;
class IAlgorithmVariant;
class IAssertionLayer;
class IBuilder;
class IBuilderConfig;
class IConcatenationLayer;
class IConditionLayer;
class IConstantLayer;
class IConvolutionLayer;
class ICudaEngine;
class IDeconvolutionLayer;
class IDequantizeLayer;
class IDimensionExpr;
class IEinsumLayer;
class IElementWiseLayer;
class IEngineInspector;
class IExecutionContext;
class IFillLayer;
class IGatherLayer;
class IGridSampleLayer;
class IHostMemory;
class IIdentityLayer;
class ICastLayer;
class IIfConditional;
class IIfConditionalInputLayer;
class IIfConditionalOutputLayer;
class IInt8Calibrator;
class IIteratorLayer;
class ILayer;
class ILoop;
class ILoopOutputLayer;
class ILRNLayer;
class IMatrixMultiplyLayer;
class INetworkDefinition;
class INormalizationLayer;
class INMSLayer;
class INonZeroLayer;
class IOneHotLayer;
class IOptimizationProfile;
class IPaddingLayer;
class IParametricReLULayer;
class IPlugin;
class IPluginExt;
class IPluginFactory;
class IPluginLayer;
class IPluginRegistry;
class IPluginV2Layer;

namespace v_1_0
{
class IPluginV3;
} // namespace v_1_0
using IPluginV3 = v_1_0::IPluginV3;

namespace v_1_0
{
class IStreamReader;
} // namespace v_1_0
using IStreamReader = v_1_0::IStreamReader;
namespace v_1_0
{
class IStreamReaderV2;
} // namespace v_1_0
using IStreamReaderV2 = v_1_0::IStreamReaderV2;

class IPluginV3Layer;
class IPoolingLayer;
class IQuantizeLayer;
class IRaggedSoftMaxLayer;
class IRecurrenceLayer;
class IReduceLayer;
class IRefitter;
class IResizeLayer;
class IReverseSequenceLayer;
class IRuntime;
class IScaleLayer;
class IScatterLayer;
class ISelectLayer;
class ISerializationConfig;
class IShapeLayer;
class IShuffleLayer;
class ISliceLayer;
class ISoftMaxLayer;
class ISqueezeLayer;
class ITensor;
class ITimingCache;
class ITopKLayer;
class ITripLimitLayer;
class IUnaryLayer;
class IUnsqueezeLayer;
struct Permutation;
class Weights;

enum class ActivationType : int32_t;
enum class BoundingBoxFormat : int32_t;
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
enum class InterpolationMode : int32_t;
enum class ResizeRoundMode : int32_t;
enum class ResizeSelector : int32_t;
enum class ScaleMode : int32_t;
enum class ScatterMode : int32_t;
enum class SampleMode : int32_t;
enum class SerializationFlag : int32_t;
enum class TensorIOMode : int32_t;
enum class TensorLocation : int32_t;
enum class TopKOperation : int32_t;
enum class TripLimit : int32_t;
enum class UnaryOperation : int32_t;
enum class WeightsRole : int32_t;
enum class PreviewFeature : int32_t;
enum class HardwareCompatibilityLevel : int32_t;
enum class ExecutionContextAllocationStrategy : int32_t;
enum class RuntimePlatform : int32_t;

using TacticSources = uint32_t;
using TensorFormats = uint32_t;
using BuilderFlags = uint32_t;
using NetworkDefinitionCreationFlags = uint32_t;
using QuantizationFlags = uint32_t;
using TempfileControlFlags = uint32_t;
using SerializationFlags = uint32_t;

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
    virtual int64_t getConstantValue() const = 0;
    virtual bool isSizeTensor() const = 0;
};

class VExprBuilder : public VRoot
{
public:
    virtual IDimensionExpr const* constant(int64_t value) = 0;
    virtual IDimensionExpr const* operation(
        DimensionOperation op, IDimensionExpr const& first, IDimensionExpr const& second)
        = 0;
    virtual IDimensionExpr const* declareSizeTensor(
        int32_t outputIndex, IDimensionExpr const& opt, IDimensionExpr const& upper)
        = 0;
};

class VRuntime : public VRoot
{
public:
    virtual IRuntime* getPImpl() noexcept = 0;
    virtual nvinfer1::ICudaEngine* deserializeCudaEngine(void const* blob, std::size_t size) noexcept = 0;
    virtual nvinfer1::ICudaEngine* deserializeCudaEngine(IStreamReader& streamReader) noexcept = 0;
    virtual void setDLACore(int32_t dlaCore) noexcept = 0;
    virtual int32_t getDLACore() const noexcept = 0;
    virtual int32_t getNbDLACores() const noexcept = 0;
    virtual void setGpuAllocator(IGpuAllocator* allocator) noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual ILogger* getLogger() const noexcept = 0;
    virtual bool setMaxThreads(int32_t maxThreads) noexcept = 0;
    virtual int32_t getMaxThreads() const noexcept = 0;
    virtual void setTemporaryDirectory(char const*) noexcept = 0;
    virtual char const* getTemporaryDirectory() const noexcept = 0;
    virtual void setTempfileControlFlags(TempfileControlFlags) noexcept = 0;
    virtual TempfileControlFlags getTempfileControlFlags() const noexcept = 0;
    virtual IPluginRegistry& getPluginRegistry() noexcept = 0;
    virtual void setPluginRegistryParent(IPluginRegistry* parent) noexcept = 0;
    virtual IRuntime* loadRuntime(char const* path) noexcept = 0;
    virtual void setEngineHostCodeAllowed(bool allowed) noexcept = 0;
    virtual bool getEngineHostCodeAllowed() const noexcept = 0;
    // Added in TensorRT version 10.7
    virtual nvinfer1::ICudaEngine* deserializeCudaEngineV2(IStreamReaderV2& streamReader) noexcept = 0;
};

class VRefitter : public VRoot
{
public:
    virtual IRefitter* getPImpl() noexcept = 0;
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
    virtual bool setNamedWeightsWithLocation(char const* name, Weights weights, TensorLocation location) noexcept = 0;
    virtual Weights getNamedWeights(char const* weightsName) const noexcept = 0;
    virtual TensorLocation getWeightsLocation(char const* weightsName) const noexcept = 0;
    virtual bool unsetNamedWeights(char const* weightsName) noexcept = 0;
    virtual void setWeightsValidation(bool weightsValidation) noexcept = 0;
    virtual bool getWeightsValidation() const noexcept = 0;
    virtual bool refitCudaEngineAsync(cudaStream_t stream) noexcept = 0;
    virtual Weights getWeightsPrototype(char const* weightsName) const noexcept = 0;
};

class VOptimizationProfile : public VRoot
{
public:
    virtual bool setDimensions(char const* inputName, OptProfileSelector select, Dims const& dims) noexcept = 0;
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
    virtual ICudaEngine* getPImpl() noexcept = 0;
    virtual int32_t getNbLayers() const noexcept = 0;
    virtual IHostMemory* serialize() const noexcept = 0;
    virtual IExecutionContext* createExecutionContext(ExecutionContextAllocationStrategy strategy) noexcept = 0;
    virtual IExecutionContext* createExecutionContextWithoutDeviceMemory() noexcept = 0;
    virtual size_t getDeviceMemorySize() const noexcept = 0;
    virtual bool isRefittable() const noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual int32_t getNbOptimizationProfiles() const noexcept = 0;
    virtual int32_t const* getProfileTensorValues(
        char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept = 0;
    virtual EngineCapability getEngineCapability() const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual bool hasImplicitBatchDimension() const noexcept = 0;
    virtual TacticSources getTacticSources() const noexcept = 0;
    virtual ProfilingVerbosity getProfilingVerbosity() const noexcept = 0;
    virtual IEngineInspector* createEngineInspector() const noexcept = 0;
    virtual Dims getTensorShape(char const* tensorName) const noexcept = 0;
    virtual DataType getTensorDataType(char const* tensorName) const noexcept = 0;
    virtual TensorLocation getTensorLocation(char const* tensorName) const noexcept = 0;
    virtual bool isShapeInferenceIO(char const* tensorName) const noexcept = 0;
    virtual TensorIOMode getTensorIOMode(char const* tensorName) const noexcept = 0;
    virtual int32_t getTensorBytesPerComponent(char const* tensorName) const noexcept = 0;
    virtual int32_t getTensorComponentsPerElement(char const* tensorName) const noexcept = 0;
    virtual TensorFormat getTensorFormat(char const* tensorName) const noexcept = 0;
    virtual char const* getTensorFormatDesc(char const* tensorName) const noexcept = 0;
    virtual int32_t getTensorVectorizedDim(char const* tensorName) const noexcept = 0;
    virtual Dims getProfileShape(
        char const* tensorName, int32_t profileIndex, OptProfileSelector select) const noexcept = 0;
    virtual int32_t getNbIOTensors() const noexcept = 0;
    virtual char const* getIOTensorName(int32_t index) const noexcept = 0;
    virtual HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept = 0;
    virtual int32_t getNbAuxStreams() const noexcept = 0;

    virtual int32_t getTensorBytesPerComponentV2(char const* tensorName, int32_t profileIndex) const noexcept = 0;
    virtual int32_t getTensorComponentsPerElementV2(char const* tensorName, int32_t profileIndex) const noexcept = 0;
    virtual TensorFormat getTensorFormatV2(char const* tensorName, int32_t profileIndex) const noexcept = 0;
    virtual char const* getTensorFormatDescV2(char const* tensorName, int32_t profileIndex) const noexcept = 0;
    virtual int32_t getTensorVectorizedDimV2(char const* tensorName, int32_t profileIndex) const noexcept = 0;

    virtual ISerializationConfig* createSerializationConfig() noexcept = 0;
    virtual IHostMemory* serializeWithConfig(ISerializationConfig& config) const noexcept = 0;

    virtual size_t getDeviceMemorySizeForProfile(int32_t profileIndex) const noexcept = 0;
    virtual IRefitter* createRefitter(ILogger& logger) noexcept = 0;

    virtual bool setWeightStreamingBudget(int64_t gpuMemoryBudget) noexcept = 0;
    virtual int64_t getWeightStreamingBudget() const noexcept = 0;
    virtual int64_t getMinimumWeightStreamingBudget() const noexcept = 0;
    virtual int64_t getStreamableWeightsSize() const noexcept = 0;

    virtual bool isDebugTensor(char const* name) const noexcept = 0;

    // Added in TensorRT 10.1
    virtual bool setWeightStreamingBudgetV2(int64_t gpuMemoryBudget) noexcept = 0;
    virtual int64_t getWeightStreamingBudgetV2() const noexcept = 0;
    virtual int64_t getWeightStreamingAutomaticBudget() const noexcept = 0;
    virtual int64_t getWeightStreamingScratchMemorySize() const noexcept = 0;
    virtual int64_t getDeviceMemorySizeV2() const noexcept = 0;
    virtual int64_t getDeviceMemorySizeForProfileV2(int32_t profileIndex) const noexcept = 0;
};

class VExecutionContext : public VRoot
{
public:
    virtual IExecutionContext* getPImpl() noexcept = 0;
    virtual void setDebugSync(bool sync) noexcept = 0;
    virtual bool getDebugSync() const noexcept = 0;
    virtual void setProfiler(IProfiler*) noexcept = 0;
    virtual IProfiler* getProfiler() const noexcept = 0;
    virtual ICudaEngine const& getEngine() const noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual void setDeviceMemory(void* memory) noexcept = 0;
    virtual int32_t getOptimizationProfile() const noexcept = 0;
    virtual bool allInputDimensionsSpecified() const noexcept = 0;
    virtual bool allInputShapesSpecified() const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual bool executeV2(void* const* bindings) noexcept = 0;
    virtual bool setOptimizationProfileAsync(int32_t profileIndex, cudaStream_t stream) noexcept = 0;
    virtual void setEnqueueEmitsProfile(bool enqueueEmitsProfile) noexcept = 0;
    virtual bool getEnqueueEmitsProfile() const noexcept = 0;
    virtual bool reportToProfiler() const noexcept = 0;
    virtual bool setInputShape(char const* tensorName, Dims const& dims) noexcept = 0;
    virtual Dims getTensorShape(char const* tensorName) const noexcept = 0;
    virtual Dims getTensorStrides(char const* tensorName) const noexcept = 0;
    virtual bool setTensorAddress(char const* tensorName, void* data) noexcept = 0;
    virtual void const* getTensorAddress(char const* tensorName) const noexcept = 0;
    virtual bool setInputTensorAddress(char const* tensorName, void const* data) noexcept = 0;
    virtual bool setOutputTensorAddress(char const* tensorName, void* data) noexcept = 0;
    virtual int32_t inferShapes(int32_t nbMaxNames, char const** tensorNames) noexcept = 0;
    virtual bool setInputConsumedEvent(cudaEvent_t event) noexcept = 0;
    virtual cudaEvent_t getInputConsumedEvent() const noexcept = 0;
    virtual void* getOutputTensorAddress(char const* tensorName) const noexcept = 0;
    virtual bool setOutputAllocator(char const* tensorName, IOutputAllocator* outputAllocator) noexcept = 0;
    virtual IOutputAllocator* getOutputAllocator(char const* name) noexcept = 0;
    virtual int64_t getMaxOutputSize(char const* tensorName) const noexcept = 0;
    virtual bool setTemporaryStorageAllocator(IGpuAllocator* allocator) noexcept = 0;
    virtual IGpuAllocator* getTemporaryStorageAllocator() const noexcept = 0;
    virtual bool enqueueV3(cudaStream_t stream) noexcept = 0;
    virtual void setPersistentCacheLimit(size_t size) noexcept = 0;
    virtual size_t getPersistentCacheLimit() const noexcept = 0;
    virtual bool setNvtxVerbosity(ProfilingVerbosity verbosity) noexcept = 0;
    virtual ProfilingVerbosity getNvtxVerbosity() const noexcept = 0;
    virtual void setAuxStreams(cudaStream_t* auxStreams, int32_t nbStreams) noexcept = 0;
    virtual bool setDebugListener(IDebugListener* listener) noexcept = 0;
    virtual IDebugListener* getDebugListener() noexcept = 0;
    virtual bool setTensorDebugState(char const* name, bool flag) noexcept = 0;
    virtual bool getDebugState(char const* name) const noexcept = 0;
    virtual bool setAllTensorsDebugState(bool flag) noexcept = 0;
    virtual size_t updateDeviceMemorySizeForShapes() noexcept = 0;

    // Added in TensorRT 10.1
    virtual void setDeviceMemoryV2(void* memory, int64_t size) noexcept = 0;
};

class VEngineInspector : public VRoot
{
public:
    virtual IEngineInspector* getPImpl() noexcept = 0;
    virtual bool setExecutionContext(IExecutionContext const* context) noexcept = 0;
    virtual IExecutionContext const* getExecutionContext() const noexcept = 0;
    virtual char const* getLayerInformation(int32_t layerIndex, LayerInformationFormat format) const noexcept = 0;
    virtual char const* getEngineInformation(LayerInformationFormat format) const noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
};

class VTensor : public VRoot
{
public:
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual void setDimensions(Dims const& dimensions) noexcept = 0;
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
    virtual void setDimensionName(int32_t index, char const* name) noexcept = 0;
    virtual char const* getDimensionName(int32_t index) const noexcept = 0;
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
    virtual void setMetadata(char const* docString) noexcept = 0;
    virtual char const* getMetadata() const noexcept = 0;
};

class VConvolutionLayer : public VRoot
{
public:
    virtual void setNbOutputMaps(int64_t nbOutputMaps) noexcept = 0;
    virtual int64_t getNbOutputMaps() const noexcept = 0;
    virtual void setNbGroups(int64_t nbGroups) noexcept = 0;
    virtual int64_t getNbGroups() const noexcept = 0;
    virtual void setKernelWeights(Weights weights) noexcept = 0;
    virtual Weights getKernelWeights() const noexcept = 0;
    virtual void setBiasWeights(Weights weights) noexcept = 0;
    virtual Weights getBiasWeights() const noexcept = 0;
    virtual void setPrePadding(Dims const&  padding) noexcept = 0;
    virtual Dims getPrePadding() const noexcept = 0;
    virtual void setPostPadding(Dims const& padding) noexcept = 0;
    virtual Dims getPostPadding() const noexcept = 0;
    virtual void setPaddingMode(PaddingMode paddingMode) noexcept = 0;
    virtual PaddingMode getPaddingMode() const noexcept = 0;
    virtual void setKernelSizeNd(Dims const& kernelSize) noexcept = 0;
    virtual Dims getKernelSizeNd() const noexcept = 0;
    virtual void setStrideNd(Dims const& stride) noexcept = 0;
    virtual Dims getStrideNd() const noexcept = 0;
    virtual void setPaddingNd(Dims const& padding) noexcept = 0;
    virtual Dims getPaddingNd() const noexcept = 0;
    virtual void setDilationNd(Dims const& dilation) noexcept = 0;
    virtual Dims getDilationNd() const noexcept = 0;
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
    virtual void setBlendFactor(float blendFactor) noexcept = 0;
    virtual float getBlendFactor() const noexcept = 0;
    virtual void setAverageCountExcludesPadding(bool exclusive) noexcept = 0;
    virtual bool getAverageCountExcludesPadding() const noexcept = 0;
    virtual void setPrePadding(Dims const& padding) noexcept = 0;
    virtual Dims getPrePadding() const noexcept = 0;
    virtual void setPostPadding(Dims const& padding) noexcept = 0;
    virtual Dims getPostPadding() const noexcept = 0;
    virtual void setPaddingMode(PaddingMode paddingMode) noexcept = 0;
    virtual PaddingMode getPaddingMode() const noexcept = 0;
    virtual void setWindowSizeNd(Dims const& windowSize) noexcept = 0;
    virtual Dims getWindowSizeNd() const noexcept = 0;
    virtual void setStrideNd(Dims const& stride) noexcept = 0;
    virtual Dims getStrideNd() const noexcept = 0;
    virtual void setPaddingNd(Dims const& padding) noexcept = 0;
    virtual Dims getPaddingNd() const noexcept = 0;
};

class VLRNLayer : public VRoot
{
public:
    virtual void setWindowSize(int64_t windowSize) noexcept = 0;
    virtual int64_t getWindowSize() const noexcept = 0;
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
    virtual void setNbOutputMaps(int64_t nbOutputMaps) noexcept = 0;
    virtual int64_t getNbOutputMaps() const noexcept = 0;
    virtual void setNbGroups(int64_t nbGroups) noexcept = 0;
    virtual int64_t getNbGroups() const noexcept = 0;
    virtual void setKernelWeights(Weights weights) noexcept = 0;
    virtual Weights getKernelWeights() const noexcept = 0;
    virtual void setBiasWeights(Weights weights) noexcept = 0;
    virtual Weights getBiasWeights() const noexcept = 0;
    virtual void setPrePadding(Dims const& padding) noexcept = 0;
    virtual Dims getPrePadding() const noexcept = 0;
    virtual void setPostPadding(Dims const& padding) noexcept = 0;
    virtual Dims getPostPadding() const noexcept = 0;
    virtual void setPaddingMode(PaddingMode paddingMode) noexcept = 0;
    virtual PaddingMode getPaddingMode() const noexcept = 0;
    virtual void setKernelSizeNd(Dims const& kernelSize) noexcept = 0;
    virtual Dims getKernelSizeNd() const noexcept = 0;
    virtual void setStrideNd(Dims const& stride) noexcept = 0;
    virtual Dims getStrideNd() const noexcept = 0;
    virtual void setPaddingNd(Dims const& padding) noexcept = 0;
    virtual Dims getPaddingNd() const noexcept = 0;
    virtual void setDilationNd(Dims const& dilation) noexcept = 0;
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

class VPluginV3Layer : public VRoot
{
public:
    virtual IPluginV3& getPlugin() noexcept = 0;
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
    virtual void setPrePaddingNd(Dims const& padding) noexcept = 0;
    virtual Dims getPrePaddingNd() const noexcept = 0;
    virtual void setPostPaddingNd(Dims const& padding) noexcept = 0;
    virtual Dims getPostPaddingNd() const noexcept = 0;
};

class VShuffleLayer : public VRoot
{
public:
    virtual void setFirstTranspose(Permutation const& permutation) noexcept = 0;
    virtual Permutation const& getFirstTranspose() const noexcept = 0;
    virtual void setReshapeDimensions(Dims const& dimensions) noexcept = 0;
    virtual Dims getReshapeDimensions() const noexcept = 0;
    virtual void setSecondTranspose(Permutation const& permutation) noexcept = 0;
    virtual Permutation const& getSecondTranspose() const noexcept = 0;
    virtual void setZeroIsPlaceholder(bool zeroIsPlaceholder) noexcept = 0;
    virtual bool getZeroIsPlaceholder() const noexcept = 0;
};

class VSliceLayer : public VRoot
{
public:
    virtual void setStart(Dims const& start) noexcept = 0;
    virtual Dims getStart() const noexcept = 0;
    virtual void setSize(Dims const& size) noexcept = 0;
    virtual Dims getSize() const noexcept = 0;
    virtual void setStride(Dims const& stride) noexcept = 0;
    virtual Dims getStride() const noexcept = 0;
    virtual void setMode(SampleMode mode) noexcept = 0;
    virtual SampleMode getMode() const noexcept = 0;
    virtual void setAxes(Dims const& axes) noexcept = 0;
    virtual Dims getAxes() const noexcept = 0;
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

class VNonZeroLayer : public VRoot
{
public:
};

class VRaggedSoftMaxLayer : public VRoot
{
public:
};

class VIdentityLayer : public VRoot
{
public:
};

class VCastLayer : public VRoot
{
public:
    virtual void setToType(DataType toType) noexcept = 0;
    virtual DataType getToType() const noexcept = 0;
};

class VConstantLayer : public VRoot
{
public:
    virtual void setWeights(Weights weights) noexcept = 0;
    virtual Weights getWeights() const noexcept = 0;
    virtual void setDimensions(Dims const& dimensions) noexcept = 0;
    virtual Dims getDimensions() const noexcept = 0;
};

class VParametricReLULayer : public VRoot
{
public:
};

class VResizeLayer : public VRoot
{
public:
    virtual void setOutputDimensions(Dims const& dimensions) noexcept = 0;
    virtual Dims getOutputDimensions() const noexcept = 0;
    virtual void setScales(float const* scales, int32_t nbScales) noexcept = 0;
    virtual int32_t getScales(int32_t size, float* scales) const noexcept = 0;
    virtual void setResizeMode(InterpolationMode interpolationMode) noexcept = 0;
    virtual InterpolationMode getResizeMode() const noexcept = 0;
    virtual void setCoordinateTransformation(ResizeCoordinateTransformation coordTransform) noexcept = 0;
    virtual ResizeCoordinateTransformation getCoordinateTransformation() const noexcept = 0;
    virtual void setSelectorForSinglePixel(ResizeSelector selector) noexcept = 0;
    virtual ResizeSelector getSelectorForSinglePixel() const noexcept = 0;
    virtual void setNearestRounding(ResizeRoundMode value) noexcept = 0;
    virtual ResizeRoundMode getNearestRounding() const noexcept = 0;
    virtual void setCubicCoeff(float value) noexcept = 0;
    virtual float getCubicCoeff() const noexcept = 0;
    virtual void setExcludeOutside(bool value) noexcept = 0;
    virtual bool getExcludeOutside() const noexcept = 0;
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
    virtual void setDimensions(Dims const& dimensions) noexcept = 0;
    virtual Dims getDimensions() const noexcept = 0;
    virtual void setOperation(FillOperation op) noexcept = 0;
    virtual FillOperation getOperation() const noexcept = 0;
    virtual void setAlpha(double alpha) noexcept = 0;
    virtual double getAlpha() const noexcept = 0;
    virtual void setBeta(double beta) noexcept = 0;
    virtual double getBeta() const noexcept = 0;
    virtual void setAlphaInt64(int64_t alpha) noexcept = 0;
    virtual int64_t getAlphaInt64() const noexcept = 0;
    virtual void setBetaInt64(int64_t beta) noexcept = 0;
    virtual int64_t getBetaInt64() const noexcept = 0;
    virtual bool isAlphaBetaInt64() const noexcept = 0;
    virtual DataType getToType() const noexcept = 0;
    virtual void setToType(DataType toType) noexcept = 0;
};

class VQuantizeLayer : public VRoot
{
public:
    virtual int32_t getAxis() const noexcept = 0;
    virtual void setAxis(int32_t axis) noexcept = 0;
    virtual DataType getToType() const noexcept = 0;
    virtual void setToType(DataType toType) noexcept = 0;
};

class VDequantizeLayer : public VRoot
{
public:
    virtual int32_t getAxis() const noexcept = 0;
    virtual void setAxis(int32_t axis) noexcept = 0;
    virtual DataType getToType() const noexcept = 0;
    virtual void setToType(DataType toType) noexcept = 0;
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

class VOneHotLayer : public VRoot
{
public:
    virtual int32_t getAxis() const noexcept = 0;
    virtual void setAxis(int32_t axis) noexcept = 0;
}; // class VOneHotLayer

class VGridSampleLayer : public VRoot
{
public:
    virtual void setInterpolationMode(InterpolationMode mode) noexcept = 0;
    virtual InterpolationMode getInterpolationMode() const noexcept = 0;
    virtual void setAlignCorners(bool alignCorners) noexcept = 0;
    virtual bool getAlignCorners() const noexcept = 0;
    virtual bool setSampleMode(SampleMode mode) noexcept = 0;
    virtual SampleMode getSampleMode() const noexcept = 0;
}; // class VGridSampleLayer

class VNMSLayer : public VRoot
{
public:
    virtual void setBoundingBoxFormat(BoundingBoxFormat fmt) noexcept = 0;
    virtual BoundingBoxFormat getBoundingBoxFormat() const noexcept = 0;
    virtual void setTopKBoxLimit(int32_t limit) noexcept = 0;
    virtual int32_t getTopKBoxLimit() const noexcept = 0;
}; // class VNMSLayer

class VReverseSequenceLayer : public VRoot
{
public:
    virtual void setBatchAxis(int32_t batchAxis) noexcept = 0;
    virtual int32_t getBatchAxis() const noexcept = 0;

    virtual void setSequenceAxis(int32_t sequenceAxis) noexcept = 0;
    virtual int32_t getSequenceAxis() const noexcept = 0;
}; // class VReverseSequenceLayer

class VNormalizationLayer : public VRoot
{
public:
    virtual void setEpsilon(float eps) noexcept = 0;
    virtual float getEpsilon() const noexcept = 0;
    virtual void setAxes(uint32_t axesMask) noexcept = 0;
    virtual uint32_t getAxes() const noexcept = 0;
    virtual void setNbGroups(int64_t nbGroups) noexcept = 0;
    virtual int64_t getNbGroups() const noexcept = 0;
    virtual void setComputePrecision(DataType type) noexcept = 0;
    virtual DataType getComputePrecision() const noexcept = 0;
}; // class VNormalizationLayer

class VSqueezeLayer : public VRoot
{
};

class VUnsqueezeLayer : public VRoot
{
};

class VNetworkDefinition : public VRoot
{
public:
    virtual ITensor* addInput(char const* name, DataType type, Dims const& dimensions) noexcept = 0;
    virtual void markOutput(ITensor& tensor) noexcept = 0;
    virtual IActivationLayer* addActivation(ITensor& input, ActivationType type) noexcept = 0;
    virtual ILRNLayer* addLRN(ITensor& input, int64_t window, float alpha, float beta, float k) noexcept = 0;
    virtual IScaleLayer* addScale(
        ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) noexcept = 0;
    virtual ISoftMaxLayer* addSoftMax(ITensor& input) noexcept = 0;
    virtual IConcatenationLayer* addConcatenation(ITensor* const* inputs, int32_t nbInputs) noexcept = 0;
    virtual IElementWiseLayer* addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) noexcept = 0;
    virtual IUnaryLayer* addUnary(ITensor& input, UnaryOperation operation) noexcept = 0;
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
        ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1) noexcept = 0;
    virtual IConstantLayer* addConstant(Dims const& dimensions, Weights weights) noexcept = 0;
    virtual IIdentityLayer* addIdentity(ITensor& input) noexcept = 0;
    virtual void removeTensor(ITensor& tensor) noexcept = 0;
    virtual void unmarkOutput(ITensor& tensor) noexcept = 0;
    virtual IPluginV2Layer* addPluginV2(ITensor* const* inputs, int32_t nbInputs, IPluginV2& plugin) noexcept = 0;
    virtual IPluginV3Layer* addPluginV3(ITensor* const* inputs, int32_t nbInputs, ITensor* const* shapeInputs,
        int32_t nbShapeInputs, IPluginV3& plugin) noexcept = 0;
    virtual ISliceLayer* addSlice(ITensor& input, Dims const& start, Dims const& size, Dims const& stride) noexcept = 0;
    virtual void setName(char const* name) noexcept = 0;
    virtual char const* getName() const noexcept = 0;
    virtual IShapeLayer* addShape(ITensor& input) noexcept = 0;
    virtual bool hasImplicitBatchDimension() const noexcept = 0;
    virtual bool markOutputForShapes(ITensor& tensor) noexcept = 0;
    virtual bool unmarkOutputForShapes(ITensor& tensor) noexcept = 0;
    virtual IParametricReLULayer* addParametricReLU(ITensor& input, ITensor& slope) noexcept = 0;
    virtual IConvolutionLayer* addConvolutionNd(
        ITensor& input, int64_t nbOutputMaps, Dims const& kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
        = 0;
    virtual IPoolingLayer* addPoolingNd(ITensor& input, PoolingType type, Dims const& windowSize) noexcept = 0;
    virtual IDeconvolutionLayer* addDeconvolutionNd(
        ITensor& input, int64_t nbOutputMaps, Dims const& kernelSize, Weights kernelWeights, Weights biasWeights) noexcept
        = 0;
    virtual IScaleLayer* addScaleNd(
        ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power, int32_t channelAxis) noexcept = 0;
    virtual IResizeLayer* addResize(ITensor& input) noexcept = 0;
    virtual ILoop* addLoop() noexcept = 0;
    virtual ISelectLayer* addSelect(ITensor& condition, ITensor& thenInput, ITensor& elseInput) noexcept = 0;
    virtual IFillLayer* addFill(Dims const& dimensions, FillOperation op) noexcept = 0;
    virtual IPaddingLayer* addPaddingNd(ITensor& input, Dims const& prePadding, Dims const& postPadding) noexcept = 0;
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
    virtual IOneHotLayer* addOneHot(ITensor& indices, ITensor& values, ITensor& depth, int32_t axis) noexcept = 0;
    virtual INonZeroLayer* addNonZero(ITensor& input) noexcept = 0;
    virtual IGridSampleLayer* addGridSample(ITensor& input, ITensor& grid) noexcept = 0;
    virtual INMSLayer* addNMS(ITensor& boxes, ITensor& scores, ITensor& maxOutputBoxesPerClass) noexcept = 0;
    virtual IReverseSequenceLayer* addReverseSequence(ITensor& input, ITensor& sequenceLens) noexcept = 0;
    virtual INormalizationLayer* addNormalization(
        ITensor& input, ITensor& scale, ITensor& bias, uint32_t axesMask) noexcept = 0;
    virtual ICastLayer* addCast(ITensor& input, DataType toType) noexcept = 0;
    virtual IBuilder& getBuilder() const noexcept = 0;
    virtual NetworkDefinitionCreationFlags getFlags() const noexcept = 0;
    virtual bool getFlag(NetworkDefinitionCreationFlag networkDefinitionCreationFlag) const noexcept = 0;
    virtual IQuantizeLayer* addQuantizeV2(ITensor& input, ITensor& scale, DataType outputType) noexcept = 0;
    virtual IDequantizeLayer* addDequantizeV2(ITensor& input, ITensor& scale, DataType outputType) noexcept = 0;
    virtual IFillLayer* addFillV2(Dims const& dimensions, FillOperation op, DataType outputType) noexcept = 0;
    virtual bool markDebug(ITensor& tensor) noexcept = 0;
    virtual bool unmarkDebug(ITensor& tensor) noexcept = 0;
    virtual bool isDebugTensor(nvinfer1::ITensor const& tensor) const noexcept = 0;
    virtual bool markWeightsRefittable(char const* name) noexcept = 0;
    virtual bool unmarkWeightsRefittable(char const* name) noexcept = 0;
    virtual bool areWeightsMarkedRefittable(char const* name) const noexcept = 0;
    virtual ISqueezeLayer* addSqueeze(ITensor& input, ITensor& axes) noexcept = 0;
    virtual IUnsqueezeLayer* addUnsqueeze(ITensor& input, ITensor& axes) noexcept = 0;
};

class VAlgorithmIOInfo : public VRoot
{
public:
    virtual DataType getDataType() const noexcept = 0;
    virtual Dims getStrides() const noexcept = 0;
    virtual int64_t getVectorizedDim() const noexcept = 0;
    virtual int64_t getComponentsPerElement() const noexcept = 0;
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
    virtual void setAvgTimingIterations(int32_t avgTiming) noexcept = 0;
    virtual int32_t getAvgTimingIterations() const noexcept = 0;
    virtual void setEngineCapability(EngineCapability capability) noexcept = 0;
    virtual EngineCapability getEngineCapability() const noexcept = 0;
    virtual void setInt8Calibrator(IInt8Calibrator* calibrator) noexcept = 0;
    virtual IInt8Calibrator* getInt8Calibrator() const noexcept = 0;
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
    virtual void setPreviewFeature(PreviewFeature feature, bool enable) noexcept = 0;
    virtual bool getPreviewFeature(PreviewFeature feature) const noexcept = 0;
    virtual void setBuilderOptimizationLevel(int32_t level) noexcept = 0;
    virtual int32_t getBuilderOptimizationLevel() const noexcept = 0;
    virtual void setHardwareCompatibilityLevel(HardwareCompatibilityLevel hardwareCompatibilityLevel) noexcept = 0;
    virtual HardwareCompatibilityLevel getHardwareCompatibilityLevel() const noexcept = 0;
    virtual void setPluginsToSerialize(char const* const* paths, int32_t nbPaths) noexcept = 0;
    virtual char const* getPluginToSerialize(int32_t index) const noexcept = 0;
    virtual int32_t getNbPluginsToSerialize() const noexcept = 0;
    virtual void setMaxAuxStreams(int32_t nbStreams) noexcept = 0;
    virtual int32_t getMaxAuxStreams() const noexcept = 0;
    virtual void setProgressMonitor(IProgressMonitor* monitor) noexcept = 0;
    virtual IProgressMonitor* getProgressMonitor() const noexcept = 0;
    virtual void setRuntimePlatform(RuntimePlatform runtimePlatform) noexcept = 0;
    virtual RuntimePlatform getRuntimePlatform() const noexcept = 0;
    virtual void setMaxNbTactics(int32_t maxTactics) noexcept = 0;
    virtual int32_t getMaxNbTactics() const noexcept = 0;
};

class VSerializationConfig : public VRoot
{
public:
    virtual bool setFlags(SerializationFlags serializationFlags) noexcept = 0;
    virtual SerializationFlags getFlags() const noexcept = 0;
    virtual bool clearFlag(SerializationFlag serializationFlag) noexcept = 0;
    virtual bool setFlag(SerializationFlag serializationFlag) noexcept = 0;
    virtual bool getFlag(SerializationFlag serializationFlag) const noexcept = 0;
};

class VBuilder : public VRoot
{
public:
    virtual bool platformHasFastFp16() const noexcept = 0;
    virtual bool platformHasFastInt8() const noexcept = 0;
    virtual int32_t getMaxDLABatchSize() const noexcept = 0;
    virtual int32_t getNbDLACores() const noexcept = 0;
    virtual void setGpuAllocator(IGpuAllocator* allocator) noexcept = 0;
    virtual nvinfer1::IBuilderConfig* createBuilderConfig() noexcept = 0;
    virtual nvinfer1::INetworkDefinition* createNetworkV2(NetworkDefinitionCreationFlags flags) noexcept = 0;
    virtual nvinfer1::IOptimizationProfile* createOptimizationProfile() noexcept = 0;
    virtual void setErrorRecorder(IErrorRecorder* recorder) noexcept = 0;
    virtual IErrorRecorder* getErrorRecorder() const noexcept = 0;
    virtual void reset() noexcept = 0;
    virtual bool platformHasTf32() const noexcept = 0;
    virtual nvinfer1::IHostMemory* buildSerializedNetwork(
        INetworkDefinition& network, IBuilderConfig& config) noexcept = 0;
    virtual bool isNetworkSupported(INetworkDefinition const& network, IBuilderConfig const& config) const noexcept = 0;
    virtual ILogger* getLogger() const noexcept = 0;
    virtual bool setMaxThreads(int32_t maxThreads) noexcept = 0;
    virtual int32_t getMaxThreads() const noexcept = 0;
    virtual IPluginRegistry& getPluginRegistry() noexcept = 0;
    virtual ICudaEngine* buildEngineWithConfig(INetworkDefinition& network, IBuilderConfig& config) noexcept = 0;
};

} // namespace apiv
} // namespace nvinfer1

// @endcond

#endif // NV_INFER_RUNTIME_IMPL_H
