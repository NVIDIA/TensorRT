/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This contains the core elements of the API, i.e. builder, logger, engine, runtime, context.
#include "ForwardDeclarations.h"
#include "utils.h"
#include <chrono>
#include <iomanip>
#include <pybind11/stl.h>

#include "infer/pyCoreDoc.h"
#include <cuda_runtime_api.h>

namespace tensorrt
{
using namespace nvinfer1;
// Long lambda functions should go here rather than being inlined into the bindings (1 liners are OK).
namespace lambdas
{
// For IOptimizationProfile
static const auto opt_profile_set_shape
    = [](IOptimizationProfile& self, std::string const& inputName, Dims const& min, Dims const& opt, Dims const& max) {
          PY_ASSERT_RUNTIME_ERROR(self.setDimensions(inputName.c_str(), OptProfileSelector::kMIN, min),
              "Shape provided for min is inconsistent with other shapes.");
          PY_ASSERT_RUNTIME_ERROR(self.setDimensions(inputName.c_str(), OptProfileSelector::kOPT, opt),
              "Shape provided for opt is inconsistent with other shapes.");
          PY_ASSERT_RUNTIME_ERROR(self.setDimensions(inputName.c_str(), OptProfileSelector::kMAX, max),
              "Shape provided for max is inconsistent with other shapes.");
      };

static const auto opt_profile_get_shape
    = [](IOptimizationProfile& self, std::string const& inputName) -> std::vector<Dims> {
    std::vector<Dims> shapes{};
    Dims minShape = self.getDimensions(inputName.c_str(), OptProfileSelector::kMIN);
    if (minShape.nbDims != -1)
    {
        shapes.emplace_back(minShape);
        shapes.emplace_back(self.getDimensions(inputName.c_str(), OptProfileSelector::kOPT));
        shapes.emplace_back(self.getDimensions(inputName.c_str(), OptProfileSelector::kMAX));
    }
    return shapes;
};

static const auto opt_profile_set_shape_input
    = [](IOptimizationProfile& self, std::string const& inputName, std::vector<int32_t> const& min,
          std::vector<int32_t> const& opt, std::vector<int32_t> const& max) {
          PY_ASSERT_RUNTIME_ERROR(self.setShapeValues(inputName.c_str(), OptProfileSelector::kMIN, min.data(), min.size()),
              "min input provided for shape tensor is inconsistent with other inputs.");
          PY_ASSERT_RUNTIME_ERROR(self.setShapeValues(inputName.c_str(), OptProfileSelector::kOPT, opt.data(), opt.size()),
              "opt input provided for shape tensor is inconsistent with other inputs.");
          PY_ASSERT_RUNTIME_ERROR(self.setShapeValues(inputName.c_str(), OptProfileSelector::kMAX, max.data(), max.size()),
              "max input provided for shape tensor is inconsistent with other inputs.");
      };

static const auto opt_profile_get_shape_input
    = [](IOptimizationProfile& self, std::string const& inputName) -> std::vector<std::vector<int32_t>> {
    std::vector<std::vector<int32_t>> shapes{};
    int32_t const shapeSize = self.getNbShapeValues(inputName.c_str());
    int32_t const* shapePtr = self.getShapeValues(inputName.c_str(), OptProfileSelector::kMIN);
    // In the Python bindings, it is impossible to set only one shape in an optimization profile.
    if (shapePtr && shapeSize >= 0)
    {
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
        shapePtr = self.getShapeValues(inputName.c_str(), OptProfileSelector::kOPT);
        PY_ASSERT_RUNTIME_ERROR(shapePtr != nullptr, "Invalid shape for OPT.");
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
        shapePtr = self.getShapeValues(inputName.c_str(), OptProfileSelector::kMAX);
        PY_ASSERT_RUNTIME_ERROR(shapePtr != nullptr, "Invalid shape for MAX.");
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
    }
    return shapes;
};

// For IExecutionContext

static const auto execute_v2 = [](IExecutionContext& self, std::vector<size_t>& bindings) {
    return self.executeV2(reinterpret_cast<void**>(bindings.data()));
};

std::vector<char const*> infer_shapes(IExecutionContext& self)
{
    int32_t const size{self.getEngine().getNbIOTensors()};
    std::vector<char const*> names(size);
    int32_t const nbNames = self.inferShapes(names.size(), names.data());

    if (nbNames < 0)
    {
        std::stringstream msg;
        msg << "infer_shapes error code: " << nbNames;
        utils::throwPyError(PyExc_RuntimeError, msg.str().c_str());
    }

    names.resize(nbNames);
    return names;
}

bool execute_async_v3(IExecutionContext& self, size_t streamHandle)
{
    return self.enqueueV3(reinterpret_cast<cudaStream_t>(streamHandle));
}

bool set_tensor_address(IExecutionContext& self, char const* tensor_name, size_t memory)
{
    return self.setTensorAddress(tensor_name, reinterpret_cast<void*>(memory));
}

size_t get_tensor_address(IExecutionContext& self, char const* tensor_name)
{
    return reinterpret_cast<size_t>(self.getTensorAddress(tensor_name));
}

bool set_input_consumed_event(IExecutionContext& self, size_t inputConsumed)
{
    return self.setInputConsumedEvent(reinterpret_cast<cudaEvent_t>(inputConsumed));
}

size_t get_input_consumed_event(IExecutionContext& self)
{
    return reinterpret_cast<size_t>(self.getInputConsumedEvent());
}

void set_aux_streams(IExecutionContext& self, std::vector<size_t> streamHandle)
{
    self.setAuxStreams(reinterpret_cast<cudaStream_t*>(streamHandle.data()), static_cast<int32_t>(streamHandle.size()));
}

template <typename PyIterable>
Dims castDimsFromPyIterable(PyIterable& in)
{
    int32_t const maxDims{static_cast<int32_t>(Dims::MAX_DIMS)};
    Dims dims{};
    dims.nbDims = py::len(in);
    PY_ASSERT_RUNTIME_ERROR(dims.nbDims <= maxDims, "The number of input dims exceeds the maximum allowed number of dimensions");
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        dims.d[i] = in[i].template cast<int32_t>();
    }
    return dims;
}

template <typename PyIterable>
bool setInputShape(IExecutionContext& self, char const* tensorName, PyIterable& in)
{
    return self.setInputShape(tensorName, castDimsFromPyIterable<PyIterable>(in));
}

// For IRuntime
static const auto runtime_deserialize_cuda_engine = [](IRuntime& self, py::buffer& serializedEngine) {
    py::buffer_info info = serializedEngine.request();
    return self.deserializeCudaEngine(info.ptr, info.size * info.itemsize);
};

// For ICudaEngine
// TODO: Add slicing support?
static const auto engine_getitem = [](ICudaEngine& self, int32_t pyIndex) {
    // Support python's negative indexing
    int32_t const index = (pyIndex < 0) ? static_cast<int32_t>(self.getNbIOTensors()) + pyIndex : pyIndex;
    PY_ASSERT_INDEX_ERROR(index < self.getNbIOTensors());
    return self.getIOTensorName(index);
};

std::vector<Dims> get_tensor_profile_shape(ICudaEngine& self, std::string const& tensorName, int32_t profileIndex)
{
    std::vector<Dims> shapes{};
    shapes.emplace_back(self.getProfileShape(tensorName.c_str(), profileIndex, OptProfileSelector::kMIN));
    shapes.emplace_back(self.getProfileShape(tensorName.c_str(), profileIndex, OptProfileSelector::kOPT));
    shapes.emplace_back(self.getProfileShape(tensorName.c_str(), profileIndex, OptProfileSelector::kMAX));
    return shapes;
};

std::vector<Dims> engine_get_profile_shape(ICudaEngine& self, int32_t profileIndex, int32_t bindingIndex)
{
    std::vector<Dims> shapes{};
    auto const tensorName = self.getIOTensorName(bindingIndex);
    shapes.emplace_back(self.getProfileShape(tensorName, profileIndex, OptProfileSelector::kMIN));
    shapes.emplace_back(self.getProfileShape(tensorName, profileIndex, OptProfileSelector::kOPT));
    shapes.emplace_back(self.getProfileShape(tensorName, profileIndex, OptProfileSelector::kMAX));
    return shapes;
};
// Overload to allow using binding names instead of indices.
std::vector<Dims> engine_get_profile_shape_str(ICudaEngine& self, int32_t profileIndex, std::string const& bindingName)
{
    return get_tensor_profile_shape(self, bindingName, profileIndex);
};

std::vector<std::vector<int32_t>> get_tensor_profile_values(
    ICudaEngine& self, int32_t profileIndex, std::string const& tensorName)
{
    char const* const name = tensorName.c_str();
    bool const isShapeInput{self.isShapeInferenceIO(name) && self.getTensorIOMode(name) == TensorIOMode::kINPUT};
    PY_ASSERT_RUNTIME_ERROR(isShapeInput, "Binding index does not correspond to an input shape tensor.");
    Dims const shape = self.getTensorShape(name);
    PY_ASSERT_RUNTIME_ERROR(shape.nbDims >= 0, "Missing shape for input shape tensor");
    auto const shapeSize{utils::volume(shape)};
    PY_ASSERT_RUNTIME_ERROR(shapeSize >= 0, "Negative volume for input shape tensor");
    std::vector<std::vector<int32_t>> shapes{};

    // In the Python bindings, it is impossible to set only one shape in an optimization profile.
    int32_t const* shapePtr{self.getProfileTensorValues(name, profileIndex, OptProfileSelector::kMIN)};
    if (shapePtr)
    {
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
        shapePtr = self.getProfileTensorValues(name, profileIndex, OptProfileSelector::kOPT);
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
        shapePtr = self.getProfileTensorValues(name, profileIndex, OptProfileSelector::kMAX);
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
    }
    return shapes;
};

// For IGpuAllocator
void* allocate_async(
    IGpuAllocator& self, uint64_t const size, uint64_t const alignment, AllocatorFlags const flags, size_t streamHandle)
{
    return self.allocateAsync(size, alignment, flags, reinterpret_cast<cudaStream_t>(streamHandle));
}

bool deallocate_async(IGpuAllocator& self, void* const memory, size_t streamHandle)
{
    return self.deallocateAsync(memory, reinterpret_cast<cudaStream_t>(streamHandle));
}

// For IOutputAllocator
void* reallocate_output_async(IOutputAllocator& self, char const* tensorName, void* currentMemory, uint64_t size,
    uint64_t alignment, size_t streamHandle)
{
    return self.reallocateOutputAsync(
        tensorName, currentMemory, size, alignment, reinterpret_cast<cudaStream_t>(streamHandle));
}

// For IBuilderConfig
static const auto netconfig_get_profile_stream
    = [](IBuilderConfig& self) -> size_t { return reinterpret_cast<size_t>(self.getProfileStream()); };

static const auto netconfig_set_profile_stream = [](IBuilderConfig& self, size_t streamHandle) {
    self.setProfileStream(reinterpret_cast<cudaStream_t>(streamHandle));
};

static const auto netconfig_create_timing_cache = [](IBuilderConfig& self, py::buffer& serializedTimingCache) {
    py::buffer_info info = serializedTimingCache.request();
    return self.createTimingCache(info.ptr, info.size * info.itemsize);
};

static const auto get_plugins_to_serialize = [](IBuilderConfig& self) {
    std::vector<std::string> paths;
    int64_t const nbPlugins = self.getNbPluginsToSerialize();
    if (nbPlugins < 0)
    {
        utils::throwPyError(PyExc_RuntimeError, "Internal error");
    }

    paths.reserve(nbPlugins);
    for (int64_t i = 0; i < nbPlugins; ++i)
    {
        paths.emplace_back(std::string{self.getPluginToSerialize(i)});
    }
    return paths;
};

static const auto set_plugins_to_serialize = [](IBuilderConfig& self, std::vector<std::string> const& paths) {
    std::vector<char const*> cStrings;
    cStrings.reserve(paths.size());
    for (auto const& path : paths)
    {
        cStrings.push_back(path.c_str());
    }
    self.setPluginsToSerialize(reinterpret_cast<char const* const*>(cStrings.data()), cStrings.size());
};

// For IRefitter
static const auto refitter_get_missing = [](IRefitter& self) {
    // First get the number of missing weights.
    int32_t const size{self.getMissing(0, nullptr, nullptr)};
    // Now that we know how many weights are missing, we can create the buffers appropriately.
    std::vector<const char*> layerNames(size);
    std::vector<WeightsRole> roles(size);
    self.getMissing(size, layerNames.data(), roles.data());
    return std::pair<std::vector<const char*>, std::vector<WeightsRole>>{layerNames, roles};
};

static const auto refitter_get_missing_weights = [](IRefitter& self) {
    // First get the number of missing weights.
    int32_t const size{self.getMissingWeights(0, nullptr)};
    // Now that we know how many weights are missing, we can create the buffers appropriately.
    std::vector<char const*> names(size);
    self.getMissingWeights(size, names.data());
    return names;
};

static const auto refitter_get_all = [](IRefitter& self) {
    int32_t const size{self.getAll(0, nullptr, nullptr)};
    std::vector<char const*> layerNames(size);
    std::vector<WeightsRole> roles(size);
    self.getAll(size, layerNames.data(), roles.data());
    return std::pair<std::vector<const char*>, std::vector<WeightsRole>>{layerNames, roles};
};

static const auto refitter_get_all_weights = [](IRefitter& self) {
    int32_t const size{self.getAllWeights(0, nullptr)};
    std::vector<char const*> names(size);
    self.getAllWeights(size, names.data());
    return names;
};

static const auto refitter_get_dynamic_range = [](IRefitter& self, std::string const& tensorName) {
    return py::make_tuple(self.getDynamicRangeMin(tensorName.c_str()), self.getDynamicRangeMax(tensorName.c_str()));
};

static const auto refitter_set_dynamic_range
    = [](IRefitter& self, std::string const& tensorName, std::vector<float> const& range) -> bool {
    PY_ASSERT_VALUE_ERROR(range.size() == 2, "Dynamic range must contain exactly 2 elements");
    return self.setDynamicRange(tensorName.c_str(), range[0], range[1]);
};

static const auto refitter_get_tensors_with_dynamic_range = [](IRefitter& self) {
    int32_t const size = self.getTensorsWithDynamicRange(0, nullptr);
    std::vector<char const*> tensorNames(size);
    self.getTensorsWithDynamicRange(size, tensorNames.data());
    return tensorNames;
};

static const auto refitter_refit_cuda_engine_async = [](IRefitter& self, size_t streamHandle) {
    return self.refitCudaEngineAsync(reinterpret_cast<cudaStream_t>(streamHandle));
};

static const auto context_set_optimization_profile_async
    = [](IExecutionContext& self, int32_t const profileIndex, size_t streamHandle) {
          PY_ASSERT_RUNTIME_ERROR(
              self.setOptimizationProfileAsync(profileIndex, reinterpret_cast<cudaStream_t>(streamHandle)),
              "Error in set optimization profile async.");
          return true;
      };

void context_set_device_memory(IExecutionContext& self, size_t memory)
{
    self.setDeviceMemory(reinterpret_cast<void*>(memory));
}

void serialization_config_set_flags(ISerializationConfig& self, uint32_t flags)
{
    if (!self.setFlags(flags))
    {
        utils::throwPyError(PyExc_RuntimeError, "Provided serialization flags is incorrect");
    }
}

// For IDebugListener, this function is intended to be override by client.
// The bindings here will never be called and is for documentation purpose only.
void docProcessDebugTensor(IDebugListener& self, void const* addr, TensorLocation location, DataType type,
    Dims const& shape, char const* name, size_t stream)
{
    return;
}

} // namespace lambdas

namespace PyGpuAllocatorHelper
{
template <typename TAllocator, typename... Args>
void* allocHelper(TAllocator* allocator, const char* pyFuncName, bool showWarning, Args&&... args) noexcept
{
    try
    {
        py::gil_scoped_acquire gil{};
        py::function pyAllocFunc = utils::getOverride(static_cast<TAllocator*>(allocator), pyFuncName, showWarning);

        if (!pyAllocFunc)
        {
            return nullptr;
        }

        py::object ptr = pyAllocFunc(std::forward<Args>(args)...);
        try
        {
            return reinterpret_cast<void*>(ptr.cast<size_t>());
        }
        catch (const py::cast_error& e)
        {
            std::cerr << "[ERROR] Return value of allocate() could not be interpreted as an int" << std::endl;
        }
    }
    catch (std::exception const& e)
    {
        std::cerr << "[ERROR] Exception caught in allocate(): " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[ERROR] Exception caught in allocate()" << std::endl;
        return nullptr;
    }

    return nullptr;
}

} // namespace PyGpuAllocatorHelper

class PyGpuAllocator : public IGpuAllocator
{
public:
    using IGpuAllocator::IGpuAllocator;

    void* allocate(uint64_t size, uint64_t alignment, AllocatorFlags flags) noexcept override
    {
        return PyGpuAllocatorHelper::allocHelper<IGpuAllocator>(this, "allocate", true, size, alignment, flags);
    }

    void* reallocate(void* baseAddr, uint64_t alignment, uint64_t newSize) noexcept override
    {
        return PyGpuAllocatorHelper::allocHelper<IGpuAllocator>(
            this, "reallocate", true, reinterpret_cast<size_t>(baseAddr), alignment, newSize);
    }

    bool deallocate(void* memory) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            py::function pyDeallocate = utils::getOverride(static_cast<IGpuAllocator*>(this), "deallocate");
            if (!pyDeallocate)
            {
                return false;
            }

            py::object status{};
            status = pyDeallocate(reinterpret_cast<size_t>(memory));
            return status.cast<bool>();
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in deallocate(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in deallocate()" << std::endl;
        }
        return false;
    }

}; // PyGpuAllocator

///////////

class PyGpuAsyncAllocator : public IGpuAsyncAllocator
{
public:
    using IGpuAsyncAllocator::IGpuAsyncAllocator;

    void* allocateAsync(uint64_t size, uint64_t alignment, AllocatorFlags flags, cudaStream_t stream) noexcept override
    {
        intptr_t cudaStreamPtr = reinterpret_cast<intptr_t>(stream);
        return PyGpuAllocatorHelper::allocHelper<IGpuAsyncAllocator>(
            this, "allocate_async", true, size, alignment, cudaStreamPtr, flags);
    }

    void* reallocate(void* baseAddr, uint64_t alignment, uint64_t newSize) noexcept override
    {
        return PyGpuAllocatorHelper::allocHelper<IGpuAsyncAllocator>(
            this, "reallocate", true, reinterpret_cast<size_t>(baseAddr), alignment, newSize);
    }

    bool deallocateAsync(void* memory, cudaStream_t stream) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            py::function pyDeallocateAsync
                = utils::getOverride(static_cast<IGpuAsyncAllocator*>(this), "deallocate_async");
            if (!pyDeallocateAsync)
            {
                return false;
            }

            py::object status{};
            intptr_t cudaStreamPtr = reinterpret_cast<intptr_t>(stream);
            status = pyDeallocateAsync(reinterpret_cast<size_t>(memory), cudaStreamPtr);
            return status.cast<bool>();
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in deallocate(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in deallocate()" << std::endl;
        }
        return false;
    }

}; // PyGpuAsyncAllocator

/////////////////////////////
class PyOutputAllocator : public IOutputAllocator
{
public:
    void* reallocateOutput(
        char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            py::function pyFunc = utils::getOverride(static_cast<IOutputAllocator*>(this), "reallocate_output");

            if (!pyFunc)
            {
                return nullptr;
            }

            py::object ptr = pyFunc(tensorName, reinterpret_cast<size_t>(currentMemory), size, alignment);
            try
            {
                return reinterpret_cast<void*>(ptr.cast<size_t>());
            }
            catch (const py::cast_error& e)
            {
                std::cerr << "[ERROR] Return value of reallocateOutput() could not be interpreted as an int"
                          << std::endl;
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in reallocateOutput(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in reallocateOutput()" << std::endl;
            return nullptr;
        }

        return nullptr;
    }

    void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment,
        cudaStream_t stream) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            py::function pyFunc
                = utils::getOverride(static_cast<IOutputAllocator*>(this), "reallocate_output_async", false);

            if (!pyFunc)
            {
                //! For legacy implementation, the user might not have implemented this method, so we go for the default
                //! method.
                return reallocateOutput(tensorName, currentMemory, size, alignment);
            }

            intptr_t cudaStreamPtr = reinterpret_cast<intptr_t>(stream);
            py::object ptr
                = pyFunc(tensorName, reinterpret_cast<size_t>(currentMemory), size, alignment, cudaStreamPtr);
            try
            {
                return reinterpret_cast<void*>(ptr.cast<size_t>());
            }
            catch (const py::cast_error& e)
            {
                std::cerr << "[ERROR] Return value of reallocateOutputAsync() could not be interpreted as an int"
                          << std::endl;
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in reallocateOutputAsync(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in reallocateOutputAsync()" << std::endl;
            return nullptr;
        }

        return nullptr;
    }

    void notifyShape(char const* tensorName, Dims const& dims) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            PYBIND11_OVERLOAD_PURE_NAME(void, IOutputAllocator, "notify_shape", notifyShape, tensorName, dims);
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in notifyShape(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in notifyShape()" << std::endl;
        }
    }
};

class PyStreamReader : public IStreamReader
{
public:
    int64_t read(void* destination, int64_t size) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            py::function pyFunc = utils::getOverride(static_cast<IStreamReader*>(this), "read");

            if (!pyFunc)
            {
                return 0;
            }

            py::object bytesRead = pyFunc(reinterpret_cast<size_t>(destination), size);
            return bytesRead.cast<int64_t>();
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in read(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in read()" << std::endl;
        }

        return 0;
    }
};

class PyDebugListener : public IDebugListener
{
public:
    bool processDebugTensor(void const* addr, TensorLocation location, DataType type, Dims const& shape,
        char const* name, cudaStream_t stream) override
    {
        try
        {
            py::gil_scoped_acquire gil{};
            py::function pyFunc = utils::getOverride(static_cast<IDebugListener*>(this), "process_debug_tensor");

            if (!pyFunc)
            {
                return false;
            }

            pyFunc(reinterpret_cast<size_t>(addr), location, type, shape, name, reinterpret_cast<size_t>(stream));
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in processDebugTensor(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in processDebugTensor()" << std::endl;
        }
        return true;
    }
};

void bindCore(py::module& m)
{
    class PyLogger : public ILogger
    {
    public:
        virtual void log(Severity severity, const char* msg) noexcept override
        {
            try
            {
                py::gil_scoped_acquire gil{};
                PYBIND11_OVERLOAD_PURE_NAME(void, ILogger, "log", log, severity, msg);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in log(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in log()" << std::endl;
            }
        }
    };

    py::class_<ILogger, PyLogger> baseLoggerBinding{m, "ILogger", ILoggerDoc::descr, py::module_local()};

    py::enum_<ILogger::Severity>(
        baseLoggerBinding, "Severity", py::arithmetic(), SeverityDoc::descr, py::module_local())
        .value("INTERNAL_ERROR", ILogger::Severity::kINTERNAL_ERROR, SeverityDoc::internal_error)
        .value("ERROR", ILogger::Severity::kERROR, SeverityDoc::error)
        .value("WARNING", ILogger::Severity::kWARNING, SeverityDoc::warning)
        .value("INFO", ILogger::Severity::kINFO, SeverityDoc::info)
        .value("VERBOSE", ILogger::Severity::kVERBOSE, SeverityDoc::verbose)
        // We export into the outer scope, so we can access with trt.ILogger.X.
        .export_values();

    baseLoggerBinding.def(py::init<>()).def("log", &ILogger::log, "severity"_a, "msg"_a, ILoggerDoc::log);

    class DefaultLogger : public ILogger
    {
    public:
        DefaultLogger(Severity minSeverity = Severity::kWARNING)
            : mMinSeverity(minSeverity)
        {
        }

        virtual void log(Severity severity, const char* msg) noexcept override
        {
            //  INFO is the largest value, so this comparison is inverted.
            if (severity > mMinSeverity)
                return;

            // prepend timestamp
            std::time_t timestamp = std::time(nullptr);
            tm* tm_local = std::localtime(&timestamp);
            std::cout << "[";
            std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
            std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
            std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
            std::string loggingPrefix = "[TRT] ";
            switch (severity)
            {
            case Severity::kINTERNAL_ERROR:
            {
                loggingPrefix += "[F] ";
                break;
            }
            case Severity::kERROR:
            {
                loggingPrefix += "[E] ";
                break;
            }
            case Severity::kWARNING:
            {
                loggingPrefix += "[W] ";
                break;
            }
            case Severity::kINFO:
            {
                loggingPrefix += "[I] ";
                break;
            }
            case Severity::kVERBOSE:
            {
                loggingPrefix += "[V] ";
                break;
            }
            }
            std::cout << loggingPrefix << msg << std::endl;
        }

        Severity mMinSeverity;
    };

    py::class_<DefaultLogger, ILogger>(m, "Logger", LoggerDoc::descr, py::module_local())
        .def(py::init<ILogger::Severity>(), "min_severity"_a = ILogger::Severity::kWARNING)
        .def_readwrite("min_severity", &DefaultLogger::mMinSeverity)
        .def("log", &DefaultLogger::log, "severity"_a, "msg"_a, LoggerDoc::log);

    class PyProfiler : public IProfiler
    {
    public:
        void reportLayerTime(const char* layerName, float ms) noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(void, IProfiler, "report_layer_time", reportLayerTime, layerName, ms);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in report_layer_time(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in report_layer_time()" << std::endl;
            }
        }
    };

    py::class_<IProfiler, PyProfiler>(m, "IProfiler", IProfilerDoc::descr, py::module_local())
        .def(py::init<>())
        .def("report_layer_time", &IProfiler::reportLayerTime, "layer_name"_a, "ms"_a, IProfilerDoc::report_layer_time);

    class DefaultProfiler : public IProfiler
    {
    public:
        void reportLayerTime(const char* layerName, float ms) noexcept override
        {
            std::cout << layerName << ": " << ms << "ms" << std::endl;
        }
    };

    py::class_<DefaultProfiler, IProfiler>(m, "Profiler", ProfilerDoc::descr, py::module_local())
        .def(py::init<>())
        .def("report_layer_time", &IProfiler::reportLayerTime, "layer_name"_a, "ms"_a, ProfilerDoc::report_layer_time);

    py::class_<IOptimizationProfile, std::unique_ptr<IOptimizationProfile, py::nodelete>>(
        m, "IOptimizationProfile", IOptimizationProfileDoc::descr, py::module_local())
        .def("set_shape", lambdas::opt_profile_set_shape, "input"_a, "min"_a, "opt"_a, "max"_a,
            IOptimizationProfileDoc::set_shape)
        .def("get_shape", lambdas::opt_profile_get_shape, "input"_a, IOptimizationProfileDoc::get_shape)
        .def("set_shape_input", lambdas::opt_profile_set_shape_input, "input"_a, "min"_a, "opt"_a, "max"_a,
            IOptimizationProfileDoc::set_shape_input)
        .def("get_shape_input", lambdas::opt_profile_get_shape_input, "input"_a,
            IOptimizationProfileDoc::get_shape_input)
        .def_property("extra_memory_target", &IOptimizationProfile::getExtraMemoryTarget,
            &IOptimizationProfile::setExtraMemoryTarget)
        .def("__nonzero__", &IOptimizationProfile::isValid)
        .def("__bool__", &IOptimizationProfile::isValid);

    py::enum_<ErrorCode>(m, "ErrorCodeTRT", py::arithmetic{}, ErrorCodeDoc::descr, py::module_local())
        .value("SUCCESS", ErrorCode::kSUCCESS, ErrorCodeDoc::SUCCESS)
        .value("UNSPECIFIED_ERROR", ErrorCode::kUNSPECIFIED_ERROR, ErrorCodeDoc::UNSPECIFIED_ERROR)
        .value("INTERNAL_ERROR", ErrorCode::kINTERNAL_ERROR, ErrorCodeDoc::INTERNAL_ERROR)
        .value("INVALID_ARGUMENT", ErrorCode::kINVALID_ARGUMENT, ErrorCodeDoc::INVALID_ARGUMENT)
        .value("INVALID_CONFIG", ErrorCode::kINVALID_CONFIG, ErrorCodeDoc::INVALID_CONFIG)
        .value("FAILED_ALLOCATION", ErrorCode::kFAILED_ALLOCATION, ErrorCodeDoc::FAILED_ALLOCATION)
        .value("FAILED_INITIALIZATION", ErrorCode::kFAILED_INITIALIZATION, ErrorCodeDoc::FAILED_INITIALIZATION)
        .value("FAILED_EXECUTION", ErrorCode::kFAILED_EXECUTION, ErrorCodeDoc::FAILED_EXECUTION)
        .value("FAILED_COMPUTATION", ErrorCode::kFAILED_COMPUTATION, ErrorCodeDoc::FAILED_COMPUTATION)
        .value("INVALID_STATE", ErrorCode::kINVALID_STATE, ErrorCodeDoc::INVALID_STATE)
        .value("UNSUPPORTED_STATE", ErrorCode::kUNSUPPORTED_STATE, ErrorCodeDoc::UNSUPPORTED_STATE);

    // Provide a base implementation of Error recorder.
    // Trampoline class is required as this class needs to be implemented by user.
    class PyErrorRecorder : public IErrorRecorder
    {
    public:
        virtual ErrorCode getErrorCode(int32_t errorIdx) const noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(ErrorCode, IErrorRecorder, "get_error_code", getErrorCode, errorIdx);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in get_error_code(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in get_error_code()" << std::endl;
            }
            return {};
        }

        virtual ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(ErrorDesc, IErrorRecorder, "get_error_desc", getErrorDesc, errorIdx);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in get_error_desc(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in get_error_desc()" << std::endl;
            }
            return {};
        }

        virtual void clear() noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(void, IErrorRecorder, "clear", clear);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in clear(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in clear()" << std::endl;
            }
        }

        virtual bool reportError(ErrorCode val, ErrorDesc desc) noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(bool, IErrorRecorder, "report_error", reportError, val, desc);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in report_error(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in report_error()" << std::endl;
            }
            return false;
        }

        virtual int32_t getNbErrors() const noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(int32_t, IErrorRecorder, "get_num_errors", getNbErrors);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in get_num_errors(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in get_num_errors()" << std::endl;
            }
            return -1;
        }

        virtual bool hasOverflowed() const noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(bool, IErrorRecorder, "has_overflowed", hasOverflowed);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in has_overflowed(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in has_overflowed()" << std::endl;
            }
            return false;
        }

        virtual RefCount incRefCount() noexcept override
        {
            return ++mRefCount;
        }

        virtual RefCount decRefCount() noexcept override
        {
            return --mRefCount;
        }

    private:
        int32_t mRefCount{0};
    };

    py::class_<IErrorRecorder, PyErrorRecorder>(m, "IErrorRecorder", IErrorRecorderDoc::descr, py::module_local())
        .def(py::init<>())
        .def_property_readonly("MAX_DESC_LENGTH", []() { return IErrorRecorder::kMAX_DESC_LENGTH; })
        .def("num_errors", &IErrorRecorder::getNbErrors, IErrorRecorderDoc::get_num_errors)
        .def("get_error_code", &IErrorRecorder::getErrorCode, IErrorRecorderDoc::get_error_code)
        .def("get_error_desc", &IErrorRecorder::getErrorDesc, IErrorRecorderDoc::get_error_desc)
        .def("has_overflowed", &IErrorRecorder::hasOverflowed, IErrorRecorderDoc::has_overflowed)
        .def("clear", &IErrorRecorder::clear, IErrorRecorderDoc::clear)
        .def("report_error", &IErrorRecorder::reportError, IErrorRecorderDoc::report_error);

    // Provide a base implementation of IProgressMonitor.
    // Trampoline class is required as this class needs to be implemented by the user.
    class PyProgressMonitor : public IProgressMonitor
    {
    public:
        void phaseStart(char const* phaseName, char const* parentPhase, int32_t nbSteps) noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(
                    void, IProgressMonitor, "phase_start", phaseStart, phaseName, parentPhase, nbSteps);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in phase_start(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in phase_start()" << std::endl;
            }
        }

        bool stepComplete(char const* phaseName, int32_t step) noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(bool, IProgressMonitor, "step_complete", stepComplete, phaseName, step);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in step_complete(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in step_complete()" << std::endl;
            }
            // Use the exception as an indicator that we should cancel the build
            return false;
        }

        void phaseFinish(char const* phaseName) noexcept override
        {
            try
            {
                PYBIND11_OVERLOAD_PURE_NAME(void, IProgressMonitor, "phase_finish", phaseFinish, phaseName);
            }
            catch (std::exception const& e)
            {
                std::cerr << "[ERROR] Exception caught in phase_finish(): " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[ERROR] Exception caught in phase_finish()" << std::endl;
            }
        }
    };

    py::class_<IDebugListener, PyDebugListener>(m, "IDebugListener", IDebugListenerDoc::descr, py::module_local())
        .def(py::init<>())
        .def("process_debug_tensor", lambdas::docProcessDebugTensor, "addr"_a, "location"_a, "type"_a, "shape"_a,
            "name"_a, "stream"_a, IDebugListenerDoc::process_debug_tensor);

    py::class_<IOutputAllocator, PyOutputAllocator>(
        m, "IOutputAllocator", OutputAllocatorDoc::descr, py::module_local())
        .def(py::init<>());

    py::class_<IProgressMonitor, PyProgressMonitor>(
        m, "IProgressMonitor", IProgressMonitorDoc::descr, py::module_local())
        .def(py::init<>())
        .def("phase_start", &IProgressMonitor::phaseStart, IProgressMonitorDoc::phase_start, "phase_name"_a,
            "parent_phase"_a, "num_steps"_a)
        .def("step_complete", &IProgressMonitor::stepComplete, IProgressMonitorDoc::step_complete, "phase_name"_a,
            "step"_a)
        .def("phase_finish", &IProgressMonitor::phaseFinish, IProgressMonitorDoc::phase_finish, "phase_name"_a);

    py::class_<IExecutionContext>(m, "IExecutionContext", IExecutionContextDoc::descr, py::module_local())
        .def("execute_v2", lambdas::execute_v2, "bindings"_a, IExecutionContextDoc::execute_v2,
            py::call_guard<py::gil_scoped_release>{})
        .def_property("debug_sync", &IExecutionContext::getDebugSync, &IExecutionContext::setDebugSync)
        .def_property("profiler", &IExecutionContext::getProfiler,
            py::cpp_function(&IExecutionContext::setProfiler, py::keep_alive<1, 2>{}))
        .def_property_readonly("engine", &IExecutionContext::getEngine)
        .def_property(
            "name", &IExecutionContext::getName, py::cpp_function(&IExecutionContext::setName, py::keep_alive<1, 2>{}))
        // For writeonly properties, we use a nullptr getter.
        .def_property("device_memory", nullptr, &lambdas::context_set_device_memory)
        .def("update_device_memory_size_for_shapes", &IExecutionContext::updateDeviceMemorySizeForShapes,
            IExecutionContextDoc::update_device_memory_size_for_shapes)
        .def_property_readonly("active_optimization_profile", &IExecutionContext::getOptimizationProfile)
        // Start of enqueueV3 related APIs.
        .def("get_tensor_strides", &IExecutionContext::getTensorStrides, "name"_a,
            IExecutionContextDoc::get_tensor_strides)
        .def("set_input_shape", lambdas::setInputShape<py::tuple>, "name"_a, "shape"_a,
            IExecutionContextDoc::set_input_shape)
        .def("set_input_shape", lambdas::setInputShape<py::list>, "name"_a, "shape"_a,
            IExecutionContextDoc::set_input_shape)
        .def("set_input_shape", &IExecutionContext::setInputShape, "name"_a, "shape"_a,
            IExecutionContextDoc::set_input_shape)
        .def("get_tensor_shape", &IExecutionContext::getTensorShape, "name"_a, IExecutionContextDoc::get_tensor_shape)
        .def("set_tensor_address", lambdas::set_tensor_address, "name"_a, "memory"_a,
            IExecutionContextDoc::set_tensor_address)
        .def("get_tensor_address", lambdas::get_tensor_address, "name"_a, IExecutionContextDoc::get_tensor_address)
        .def("set_input_consumed_event", lambdas::set_input_consumed_event, "event"_a,
            IExecutionContextDoc::set_input_consumed_event)
        .def("get_input_consumed_event", lambdas::get_input_consumed_event,
            IExecutionContextDoc::get_input_consumed_event)
        .def("set_output_allocator", &IExecutionContext::setOutputAllocator, "name"_a, "output_allocator"_a,
            IExecutionContextDoc::set_output_allocator, py::keep_alive<1, 3>{})
        .def("get_output_allocator", &IExecutionContext::getOutputAllocator, "name"_a,
            IExecutionContextDoc::get_output_allocator)
        .def("get_max_output_size", &IExecutionContext::getMaxOutputSize, "name"_a,
            IExecutionContextDoc::get_max_output_size)
        .def_property("temporary_allocator", &IExecutionContext::getTemporaryStorageAllocator,
            py::cpp_function(&IExecutionContext::setTemporaryStorageAllocator, py::keep_alive<1, 2>{}))
        .def("infer_shapes", lambdas::infer_shapes, IExecutionContextDoc::infer_shapes,
            py::call_guard<py::gil_scoped_release>{})
        .def("execute_async_v3", lambdas::execute_async_v3, "stream_handle"_a, IExecutionContextDoc::execute_async_v3,
            py::call_guard<py::gil_scoped_release>{})
        // End of enqueueV3 related APIs.
        .def_property_readonly("all_binding_shapes_specified", &IExecutionContext::allInputDimensionsSpecified)
        .def_property_readonly("all_shape_inputs_specified", &IExecutionContext::allInputShapesSpecified)
        .def("set_optimization_profile_async", lambdas::context_set_optimization_profile_async, "profile_index"_a,
            "stream_handle"_a, IExecutionContextDoc::set_optimization_profile_async,
            py::call_guard<py::gil_scoped_release>{})
        .def_property("error_recorder", &IExecutionContext::getErrorRecorder,
            py::cpp_function(&IExecutionContext::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def_property("enqueue_emits_profile", &IExecutionContext::getEnqueueEmitsProfile,
            py::cpp_function(&IExecutionContext::setEnqueueEmitsProfile, py::keep_alive<1, 2>{}))
        .def("report_to_profiler", &IExecutionContext::reportToProfiler, IExecutionContextDoc::report_to_profiler)
        .def_property("persistent_cache_limit", &IExecutionContext::getPersistentCacheLimit,
            &IExecutionContext::setPersistentCacheLimit)
        .def_property("nvtx_verbosity", &IExecutionContext::getNvtxVerbosity, &IExecutionContext::setNvtxVerbosity)
        .def("set_aux_streams", lambdas::set_aux_streams, "aux_streams"_a, IExecutionContextDoc::set_aux_streams)
        .def("__del__", &utils::doNothingDel<IExecutionContext>)
        .def("set_debug_listener", &IExecutionContext::setDebugListener, "listener"_a,
            IExecutionContextDoc::set_debug_listener)
        .def("get_debug_listener", &IExecutionContext::getDebugListener, IExecutionContextDoc::get_debug_listener)
        .def("set_tensor_debug_state", &IExecutionContext::setTensorDebugState, "name"_a, "flag"_a,
            IExecutionContextDoc::set_tensor_debug_state)
        .def("get_debug_state", &IExecutionContext::getDebugState, "name"_a, IExecutionContextDoc::get_debug_state)
        .def("set_all_tensors_debug_state", &IExecutionContext::setAllTensorsDebugState, "flag"_a,
            IExecutionContextDoc::set_all_tensors_debug_state)
               ;

    py::enum_<ExecutionContextAllocationStrategy>(m, "ExecutionContextAllocationStrategy", py::arithmetic{},
        ExecutionContextAllocationStrategyDoc::descr, py::module_local())
        .value("STATIC", ExecutionContextAllocationStrategy::kSTATIC, ExecutionContextAllocationStrategyDoc::STATIC)
        .value("ON_PROFILE_CHANGE", ExecutionContextAllocationStrategy::kON_PROFILE_CHANGE,
            ExecutionContextAllocationStrategyDoc::ON_PROFILE_CHANGE)
        .value("USER_MANAGED", ExecutionContextAllocationStrategy::kUSER_MANAGED,
            ExecutionContextAllocationStrategyDoc::USER_MANAGED);

    py::enum_<SerializationFlag>(
        m, "SerializationFlag", py::arithmetic{}, SerializationFlagDoc::descr, py::module_local())
        .value("EXCLUDE_WEIGHTS", SerializationFlag::kEXCLUDE_WEIGHTS, SerializationFlagDoc::EXCLUDE_WEIGHTS)
        .value("EXCLUDE_LEAN_RUNTIME", SerializationFlag::kEXCLUDE_LEAN_RUNTIME,
            SerializationFlagDoc::EXCLUDE_LEAN_RUNTIME);

    py::class_<ISerializationConfig>(m, "ISerializationConfig", ISerializationConfigDoc::descr, py::module_local())
        .def_property("flags", &ISerializationConfig::getFlags, &lambdas::serialization_config_set_flags)
        .def("clear_flag", &ISerializationConfig::clearFlag, "flag"_a, ISerializationConfigDoc::clear_flag)
        .def("set_flag", &ISerializationConfig::setFlag, "flag"_a, ISerializationConfigDoc::set_flag)
        .def("get_flag", &ISerializationConfig::getFlag, "flag"_a, ISerializationConfigDoc::get_flag);

    py::enum_<LayerInformationFormat>(m, "LayerInformationFormat", LayerInformationFormatDoc::descr, py::module_local())
        .value("ONELINE", LayerInformationFormat::kONELINE, LayerInformationFormatDoc::ONELINE)
        .value("JSON", LayerInformationFormat::kJSON, LayerInformationFormatDoc::JSON);

#if EXPORT_ALL_BINDINGS
    // EngineInspector
    py::class_<IEngineInspector>(m, "EngineInspector", RuntimeInspectorDoc::descr, py::module_local())
        .def_property(
            "execution_context", &IEngineInspector::getExecutionContext, &IEngineInspector::setExecutionContext)
        .def("get_layer_information", &IEngineInspector::getLayerInformation, "layer_index"_a, "format"_a,
            RuntimeInspectorDoc::get_layer_information)
        .def("get_engine_information", &IEngineInspector::getEngineInformation, "format"_a,
            RuntimeInspectorDoc::get_engine_information)
        .def_property("error_recorder", &IEngineInspector::getErrorRecorder,
            py::cpp_function(&IEngineInspector::setErrorRecorder, py::keep_alive<1, 2>{}));
#endif // EXPORT_ALL_BINDINGS

    py::enum_<EngineCapability>(m, "EngineCapability", py::arithmetic{}, EngineCapabilityDoc::descr, py::module_local())
        .value("STANDARD", EngineCapability::kSTANDARD, EngineCapabilityDoc::STANDARD)
        .value("SAFETY", EngineCapability::kSAFETY, EngineCapabilityDoc::SAFETY)
        .value("DLA_STANDALONE", EngineCapability::kDLA_STANDALONE, EngineCapabilityDoc::DLA_STANDALONE);

    // Bind to a Python enum called TensorLocation.
    py::enum_<TensorLocation>(m, "TensorLocation", TensorLocationDoc::descr, py::module_local())
        .value("DEVICE", TensorLocation::kDEVICE, TensorLocationDoc::DEVICE)
        .value("HOST", TensorLocation::kHOST, TensorLocationDoc::HOST); // TensorLocation

    py::enum_<TensorIOMode>(m, "TensorIOMode", TensorIOModeDoc::descr, py::module_local())
        .value("NONE", TensorIOMode::kNONE, TensorIOModeDoc::NONE)
        .value("INPUT", TensorIOMode::kINPUT, TensorIOModeDoc::INPUT)
        .value("OUTPUT", TensorIOMode::kOUTPUT, TensorIOModeDoc::OUTPUT);

    py::class_<ICudaEngine>(m, "ICudaEngine", ICudaEngineDoc::descr, py::module_local())
        .def("__getitem__", lambdas::engine_getitem)
        .def_property_readonly("has_implicit_batch_dimension",
            utils::deprecateMember(
                &ICudaEngine::hasImplicitBatchDimension, "Implicit batch dimensions support has been removed"))
        .def_property_readonly("num_layers", &ICudaEngine::getNbLayers)
        .def("serialize", &ICudaEngine::serialize, ICudaEngineDoc::serialize, py::call_guard<py::gil_scoped_release>{})
        .def("create_serialization_config", &ICudaEngine::createSerializationConfig,
            ICudaEngineDoc::create_serialization_config, py::keep_alive<0, 1>{})
        .def("serialize_with_config", &ICudaEngine::serializeWithConfig, ICudaEngineDoc::serialize_with_config,
            py::call_guard<py::gil_scoped_release>{})
        .def("create_execution_context", &ICudaEngine::createExecutionContext, ICudaEngineDoc::create_execution_context,
            py::arg("strategy") = ExecutionContextAllocationStrategy::kSTATIC, py::keep_alive<0, 1>{},
            py::call_guard<py::gil_scoped_release>{})
        .def("create_execution_context_without_device_memory",
            utils::deprecateMember(&ICudaEngine::createExecutionContextWithoutDeviceMemory, "create_execution_context"),
            ICudaEngineDoc::create_execution_context_without_device_memory, py::keep_alive<0, 1>{},
            py::call_guard<py::gil_scoped_release>{})
        .def("get_device_memory_size_for_profile", &ICudaEngine::getDeviceMemorySizeForProfile, "profile_index"_a,
            ICudaEngineDoc::get_device_memory_size_for_profile)
        .def_property_readonly("device_memory_size", &ICudaEngine::getDeviceMemorySize)
        .def_property_readonly("refittable", &ICudaEngine::isRefittable)
        .def_property_readonly("name", &ICudaEngine::getName)
        .def_property_readonly("num_optimization_profiles", &ICudaEngine::getNbOptimizationProfiles)
        .def_property_readonly("engine_capability", &ICudaEngine::getEngineCapability)
        .def("get_profile_shape", utils::deprecate(lambdas::engine_get_profile_shape, "get_tensor_profile_shape"),
            "profile_index"_a, "binding"_a, ICudaEngineDoc::get_profile_shape)
        .def("get_profile_shape", utils::deprecate(lambdas::engine_get_profile_shape_str, "get_tensor_profile_shape"),
            "profile_index"_a, "binding"_a, ICudaEngineDoc::get_profile_shape)
        // Start of enqueueV3 related APIs.
        .def_property_readonly("num_io_tensors", &ICudaEngine::getNbIOTensors)
        .def("get_tensor_name", &ICudaEngine::getIOTensorName, "index"_a, ICudaEngineDoc::get_tensor_name)
        .def("get_tensor_mode", &ICudaEngine::getTensorIOMode, "name"_a, ICudaEngineDoc::get_tensor_mode)
        .def("is_shape_inference_io", &ICudaEngine::isShapeInferenceIO, "name"_a, ICudaEngineDoc::is_shape_inference_io)
        .def("get_tensor_shape", &ICudaEngine::getTensorShape, "name"_a, ICudaEngineDoc::get_tensor_shape)
        .def("get_tensor_dtype", &ICudaEngine::getTensorDataType, "name"_a, ICudaEngineDoc::get_tensor_dtype)
        .def("get_tensor_location", &ICudaEngine::getTensorLocation, "name"_a, ICudaEngineDoc::get_tensor_location)

        .def(
            "get_tensor_bytes_per_component",
            [](ICudaEngine& self, std::string const& name) -> int32_t {
                return self.getTensorBytesPerComponent(name.c_str());
            },
            "name"_a, ICudaEngineDoc::get_tensor_bytes_per_component)
        .def(
            "get_tensor_bytes_per_component",
            [](ICudaEngine& self, std::string const& name, int32_t profileIndex) -> int32_t {
                return self.getTensorBytesPerComponent(name.c_str(), profileIndex);
            },
            "name"_a, "profile_index"_a, ICudaEngineDoc::get_tensor_bytes_per_component)

        .def(
            "get_tensor_components_per_element",
            [](ICudaEngine& self, std::string const& name) -> int32_t {
                return self.getTensorComponentsPerElement(name.c_str());
            },
            "name"_a, ICudaEngineDoc::get_tensor_components_per_element)
        .def(
            "get_tensor_components_per_element",
            [](ICudaEngine& self, std::string const& name, int32_t profileIndex) -> int32_t {
                return self.getTensorComponentsPerElement(name.c_str(), profileIndex);
            },
            "name"_a, "profile_index"_a, ICudaEngineDoc::get_tensor_components_per_element)

        .def(
            "get_tensor_format",
            [](ICudaEngine& self, std::string const& name) -> TensorFormat {
                return self.getTensorFormat(name.c_str());
            },
            "name"_a, ICudaEngineDoc::get_tensor_format)

        .def(
            "get_tensor_format",
            [](ICudaEngine& self, std::string const& name, int32_t profileIndex) -> TensorFormat {
                return self.getTensorFormat(name.c_str(), profileIndex);
            },
            "name"_a, "profile_index"_a, ICudaEngineDoc::get_tensor_format)

        .def(
            "get_tensor_format_desc",
            [](ICudaEngine& self, std::string const& name) -> const char* {
                return self.getTensorFormatDesc(name.c_str());
            },
            "name"_a, ICudaEngineDoc::get_tensor_format_desc)
        .def(
            "get_tensor_format_desc",
            [](ICudaEngine& self, std::string const& name, int32_t profileIndex) -> const char* {
                return self.getTensorFormatDesc(name.c_str(), profileIndex);
            },
            "name"_a, "profile_index"_a, ICudaEngineDoc::get_tensor_format_desc)

        .def(
            "get_tensor_vectorized_dim",
            [](ICudaEngine& self, std::string const& name) -> int32_t {
                return self.getTensorVectorizedDim(name.c_str());
            },
            "name"_a, ICudaEngineDoc::get_tensor_vectorized_dim)
        .def(
            "get_tensor_vectorized_dim",
            [](ICudaEngine& self, std::string const& name, int32_t profileIndex) -> int32_t {
                return self.getTensorVectorizedDim(name.c_str(), profileIndex);
            },
            "name"_a, "profile_index"_a, ICudaEngineDoc::get_tensor_vectorized_dim)

        .def("get_tensor_profile_shape", lambdas::get_tensor_profile_shape, "name"_a, "profile_index"_a,
            ICudaEngineDoc::get_tensor_profile_shape)
        .def("get_tensor_profile_values", lambdas::get_tensor_profile_values, "name"_a, "profile_index"_a,
            ICudaEngineDoc::get_tensor_profile_values)
        // End of enqueueV3 related APIs.
        .def_property("error_recorder", &ICudaEngine::getErrorRecorder,
            py::cpp_function(&ICudaEngine::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def_property_readonly("tactic_sources", &ICudaEngine::getTacticSources)
        .def_property_readonly("profiling_verbosity", &ICudaEngine::getProfilingVerbosity)
        .def("create_engine_inspector", &ICudaEngine::createEngineInspector, ICudaEngineDoc::create_engine_inspector,
            py::keep_alive<0, 1>{})
        .def_property_readonly("hardware_compatibility_level", &ICudaEngine::getHardwareCompatibilityLevel)
        .def_property_readonly("num_aux_streams", &ICudaEngine::getNbAuxStreams)
        // Weight streaming APIs
        .def_property(
            "weight_streaming_budget", &ICudaEngine::getWeightStreamingBudget, &ICudaEngine::setWeightStreamingBudget)
        .def_property_readonly("minimum_weight_streaming_budget", &ICudaEngine::getMinimumWeightStreamingBudget)
        .def_property_readonly("streamable_weights_size", &ICudaEngine::getStreamableWeightsSize)
        .def("is_debug_tensor", &ICudaEngine::isDebugTensor, "name"_a, ICudaEngineDoc::is_debug_tensor)
                .def("__del__", &utils::doNothingDel<ICudaEngine>);

    py::enum_<AllocatorFlag>(m, "AllocatorFlag", py::arithmetic{}, AllocatorFlagDoc::descr, py::module_local())
        .value("RESIZABLE", AllocatorFlag::kRESIZABLE, AllocatorFlagDoc::RESIZABLE);

    py::class_<IGpuAllocator, PyGpuAllocator, IVersionedInterface>(
        m, "IGpuAllocator", GpuAllocatorDoc::descr, py::module_local())
        .def(py::init<>())
        .def("allocate",
            utils::deprecateMember(&IGpuAllocator::allocate, "Deprecated in TensorRT 10.0. Superseded by allocate"),
            "size"_a, "alignment"_a, "flags"_a, GpuAllocatorDoc::allocate)
        .def("reallocate", &IGpuAllocator::reallocate, "address"_a, "alignment"_a, "new_size"_a,
            GpuAllocatorDoc::reallocate)
        .def("deallocate",
            utils::deprecateMember(
                &IGpuAllocator::deallocate, "Deprecated in TensorRT 10.0. Superseded by deallocate_async"),
            "memory"_a, GpuAllocatorDoc::deallocate)
        .def("allocate_async", &lambdas::allocate_async, "size"_a, "alignment"_a, "flags"_a, "stream"_a,
            GpuAllocatorDoc::allocate_async)
        .def("deallocate_async", &lambdas::deallocate_async, "memory"_a, "stream"_a, GpuAllocatorDoc::deallocate_async);

    py::class_<IGpuAsyncAllocator, PyGpuAsyncAllocator, IGpuAllocator>(
        m, "IGpuAsyncAllocator", GpuAsyncAllocatorDoc::descr, py::module_local())
        .def(py::init<>())
        .def("allocate",
            utils::deprecateMember(
                &IGpuAsyncAllocator::allocate, "Deprecated in TensorRT 10.0. Superseded by allocate"),
            "size"_a, "alignment"_a, "flags"_a, GpuAsyncAllocatorDoc::allocate)
        .def("deallocate",
            utils::deprecateMember(
                &IGpuAsyncAllocator::deallocate, "Deprecated in TensorRT 10.0. Superseded by deallocate_async"),
            "memory"_a, GpuAsyncAllocatorDoc::deallocate)
        .def("allocate_async", &lambdas::allocate_async, "size"_a, "alignment"_a, "flags"_a, "stream"_a,
            GpuAsyncAllocatorDoc::allocate_async)
        .def("deallocate_async", &lambdas::deallocate_async, "memory"_a, "stream"_a,
            GpuAsyncAllocatorDoc::deallocate_async);

    py::class_<IStreamReader, PyStreamReader>(m, "IStreamReader", StreamReaderDoc::descr, py::module_local())
        .def(py::init<>())
        .def("read", &IStreamReader::read, "destination"_a, "size"_a, StreamReaderDoc::read);

    py::enum_<BuilderFlag>(m, "BuilderFlag", py::arithmetic{}, BuilderFlagDoc::descr, py::module_local())
        .value("FP16", BuilderFlag::kFP16, BuilderFlagDoc::FP16)
        .value("BF16", BuilderFlag::kBF16, BuilderFlagDoc::BF16)
        .value("INT8", BuilderFlag::kINT8, BuilderFlagDoc::INT8)
        .value("DEBUG", BuilderFlag::kDEBUG, BuilderFlagDoc::DEBUG)
        .value("GPU_FALLBACK", BuilderFlag::kGPU_FALLBACK, BuilderFlagDoc::GPU_FALLBACK)
        .value("REFIT", BuilderFlag::kREFIT, BuilderFlagDoc::REFIT)
        .value("DISABLE_TIMING_CACHE", BuilderFlag::kDISABLE_TIMING_CACHE, BuilderFlagDoc::DISABLE_TIMING_CACHE)
        .value("TF32", BuilderFlag::kTF32, BuilderFlagDoc::TF32)
        .value("SPARSE_WEIGHTS", BuilderFlag::kSPARSE_WEIGHTS, BuilderFlagDoc::SPARSE_WEIGHTS)
        .value("SAFETY_SCOPE", BuilderFlag::kSAFETY_SCOPE, BuilderFlagDoc::SAFETY_SCOPE)
        .value("OBEY_PRECISION_CONSTRAINTS", BuilderFlag::kOBEY_PRECISION_CONSTRAINTS,
            BuilderFlagDoc::OBEY_PRECISION_CONSTRAINTS)
        .value("PREFER_PRECISION_CONSTRAINTS", BuilderFlag::kPREFER_PRECISION_CONSTRAINTS,
            BuilderFlagDoc::PREFER_PRECISION_CONSTRAINTS)
        .value("DIRECT_IO", BuilderFlag::kDIRECT_IO, BuilderFlagDoc::DIRECT_IO)
        .value(
            "REJECT_EMPTY_ALGORITHMS", BuilderFlag::kREJECT_EMPTY_ALGORITHMS, BuilderFlagDoc::REJECT_EMPTY_ALGORITHMS)
        .value("VERSION_COMPATIBLE", BuilderFlag::kVERSION_COMPATIBLE, BuilderFlagDoc::VERSION_COMPATIBLE)
        .value("EXCLUDE_LEAN_RUNTIME", BuilderFlag::kEXCLUDE_LEAN_RUNTIME, BuilderFlagDoc::EXCLUDE_LEAN_RUNTIME)
        .value("FP8", BuilderFlag::kFP8, BuilderFlagDoc::FP8)
        .value("ERROR_ON_TIMING_CACHE_MISS", BuilderFlag::kERROR_ON_TIMING_CACHE_MISS,
            BuilderFlagDoc::ERROR_ON_TIMING_CACHE_MISS)
        .value("DISABLE_COMPILATION_CACHE", BuilderFlag::kDISABLE_COMPILATION_CACHE,
            BuilderFlagDoc::DISABLE_COMPILATION_CACHE)
        .value("WEIGHTLESS", BuilderFlag::kWEIGHTLESS, BuilderFlagDoc::WEIGHTLESS)
        .value("STRIP_PLAN", BuilderFlag::kSTRIP_PLAN, BuilderFlagDoc::STRIP_PLAN)
        .value("REFIT_IDENTICAL", BuilderFlag::kREFIT_IDENTICAL, BuilderFlagDoc::REFIT_IDENTICAL)
        .value("WEIGHT_STREAMING", BuilderFlag::kWEIGHT_STREAMING, BuilderFlagDoc::WEIGHT_STREAMING);

    py::enum_<MemoryPoolType>(m, "MemoryPoolType", MemoryPoolTypeDoc::descr, py::module_local())
        .value("WORKSPACE", MemoryPoolType::kWORKSPACE, MemoryPoolTypeDoc::WORKSPACE)
        .value("DLA_MANAGED_SRAM", MemoryPoolType::kDLA_MANAGED_SRAM, MemoryPoolTypeDoc::DLA_MANAGED_SRAM)
        .value("DLA_LOCAL_DRAM", MemoryPoolType::kDLA_LOCAL_DRAM, MemoryPoolTypeDoc::DLA_LOCAL_DRAM)
        .value("DLA_GLOBAL_DRAM", MemoryPoolType::kDLA_GLOBAL_DRAM, MemoryPoolTypeDoc::DLA_GLOBAL_DRAM)
        .value("TACTIC_DRAM", MemoryPoolType::kTACTIC_DRAM, MemoryPoolTypeDoc::TACTIC_DRAM)
        .value("TACTIC_SHARED_MEMORY", MemoryPoolType::kTACTIC_SHARED_MEMORY, MemoryPoolTypeDoc::TACTIC_SHARED_MEMORY);

    py::enum_<QuantizationFlag>(m, "QuantizationFlag", py::arithmetic{}, QuantizationFlagDoc::descr, py::module_local())
        .value("CALIBRATE_BEFORE_FUSION", QuantizationFlag::kCALIBRATE_BEFORE_FUSION,
            QuantizationFlagDoc::CALIBRATE_BEFORE_FUSION);

    py::enum_<PreviewFeature>(m, "PreviewFeature", PreviewFeatureDoc::descr, py::module_local())
        .value("PROFILE_SHARING_0806", PreviewFeature::kPROFILE_SHARING_0806, PreviewFeatureDoc::PROFILE_SHARING_0806);

    py::enum_<HardwareCompatibilityLevel>(
        m, "HardwareCompatibilityLevel", HardwareCompatibilityLevelDoc::descr, py::module_local())
        .value("NONE", HardwareCompatibilityLevel::kNONE, HardwareCompatibilityLevelDoc::NONE)
        .value("AMPERE_PLUS", HardwareCompatibilityLevel::kAMPERE_PLUS, HardwareCompatibilityLevelDoc::AMPERE_PLUS);

    py::enum_<DeviceType>(m, "DeviceType", DeviceTypeDoc::descr, py::module_local())
        .value("GPU", DeviceType::kGPU, DeviceTypeDoc::GPU)
        .value("DLA", DeviceType::kDLA, DeviceTypeDoc::DLA);

    py::enum_<TempfileControlFlag>(m, "TempfileControlFlag", TempfileControlFlagDoc::descr, py::module_local())
        .value("ALLOW_IN_MEMORY_FILES", TempfileControlFlag::kALLOW_IN_MEMORY_FILES,
            TempfileControlFlagDoc::ALLOW_IN_MEMORY_FILES)
        .value("ALLOW_TEMPORARY_FILES", TempfileControlFlag::kALLOW_TEMPORARY_FILES,
            TempfileControlFlagDoc::ALLOW_TEMPORARY_FILES);

    // Bind to a Python enum called ProfilingVerbosity.
    py::enum_<ProfilingVerbosity>(m, "ProfilingVerbosity", ProfilingVerbosityDoc::descr, py::module_local())
        .value("LAYER_NAMES_ONLY", ProfilingVerbosity::kLAYER_NAMES_ONLY, ProfilingVerbosityDoc::LAYER_NAMES_ONLY)
        .value("DETAILED", ProfilingVerbosity::kDETAILED, ProfilingVerbosityDoc::DETAILED)
        .value("NONE", ProfilingVerbosity::kNONE, ProfilingVerbosityDoc::NONE);

    py::enum_<TacticSource>(m, "TacticSource", py::arithmetic{}, TacticSourceDoc::descr, py::module_local())
        .value("CUBLAS", TacticSource::kCUBLAS, TacticSourceDoc::CUBLAS)
        .value("CUBLAS_LT", TacticSource::kCUBLAS_LT, TacticSourceDoc::CUBLAS_LT)
        .value("CUDNN", TacticSource::kCUDNN, TacticSourceDoc::CUDNN)
        .value("EDGE_MASK_CONVOLUTIONS", TacticSource::kEDGE_MASK_CONVOLUTIONS, TacticSourceDoc::EDGE_MASK_CONVOLUTIONS)
        .value("JIT_CONVOLUTIONS", TacticSource::kJIT_CONVOLUTIONS, TacticSourceDoc::JIT_CONVOLUTIONS);

    py::class_<ITimingCache>(m, "ITimingCache", ITimingCacheDoc::descr, py::module_local())
        .def("serialize", &ITimingCache::serialize, ITimingCacheDoc::serialize)
        .def("combine", &ITimingCache::combine, "input_cache"_a, "ignore_mismatch"_a, ITimingCacheDoc::combine)
        .def("reset", &ITimingCache::reset, ITimingCacheDoc::reset);

#if EXPORT_ALL_BINDINGS
    py::class_<IBuilderConfig>(m, "IBuilderConfig", IBuilderConfigDoc::descr, py::module_local())
        .def_property(
            "avg_timing_iterations", &IBuilderConfig::getAvgTimingIterations, &IBuilderConfig::setAvgTimingIterations)
        .def_property("int8_calibrator", &IBuilderConfig::getInt8Calibrator,
            py::cpp_function(&IBuilderConfig::setInt8Calibrator, py::keep_alive<1, 2>{}))
        .def_property("engine_capability", &IBuilderConfig::getEngineCapability, &IBuilderConfig::setEngineCapability)
        .def("set_memory_pool_limit", &IBuilderConfig::setMemoryPoolLimit, "pool"_a, "pool_size"_a,
            IBuilderConfigDoc::set_memory_pool_limit)
        .def("get_memory_pool_limit", &IBuilderConfig::getMemoryPoolLimit, "pool"_a,
            IBuilderConfigDoc::get_memory_pool_limit)
        .def_property("flags", &IBuilderConfig::getFlags, &IBuilderConfig::setFlags)
        .def_property(
            "default_device_type", &IBuilderConfig::getDefaultDeviceType, &IBuilderConfig::setDefaultDeviceType)
        .def_property("DLA_core", &IBuilderConfig::getDLACore, &IBuilderConfig::setDLACore)
        .def("clear_flag", &IBuilderConfig::clearFlag, "flag"_a, IBuilderConfigDoc::clear_flag)
        .def("set_flag", &IBuilderConfig::setFlag, "flag"_a, IBuilderConfigDoc::set_flag)
        .def("get_flag", &IBuilderConfig::getFlag, "flag"_a, IBuilderConfigDoc::get_flag)
        .def_property(
            "quantization_flags", &IBuilderConfig::getQuantizationFlags, &IBuilderConfig::setQuantizationFlags)
        .def("clear_quantization_flag", &IBuilderConfig::clearQuantizationFlag, "flag"_a,
            IBuilderConfigDoc::clear_quantization_flag)
        .def("set_quantization_flag", &IBuilderConfig::setQuantizationFlag, "flag"_a,
            IBuilderConfigDoc::set_quantization_flag)
        .def("get_quantization_flag", &IBuilderConfig::getQuantizationFlag, "flag"_a,
            IBuilderConfigDoc::get_quantization_flag)
        .def("reset", &IBuilderConfig::reset, IBuilderConfigDoc::reset)
        .def_property("profile_stream", lambdas::netconfig_get_profile_stream, lambdas::netconfig_set_profile_stream)
        .def("add_optimization_profile", &IBuilderConfig::addOptimizationProfile, "profile"_a,
            IBuilderConfigDoc::add_optimization_profile)
        .def("set_calibration_profile", &IBuilderConfig::setCalibrationProfile, "profile"_a,
            IBuilderConfigDoc::set_calibration_profile)
        .def("get_calibration_profile", &IBuilderConfig::getCalibrationProfile,
            IBuilderConfigDoc::get_calibration_profile)
        .def_property_readonly("num_optimization_profiles", &IBuilderConfig::getNbOptimizationProfiles)
        .def("set_device_type", &IBuilderConfig::setDeviceType, "layer"_a, "device_type"_a,
            IBuilderConfigDoc::set_device_type)
        .def("get_device_type", &IBuilderConfig::getDeviceType, "layer"_a, IBuilderConfigDoc::get_device_type)
        .def("is_device_type_set", &IBuilderConfig::isDeviceTypeSet, "layer"_a, IBuilderConfigDoc::is_device_type_set)
        .def("reset_device_type", &IBuilderConfig::resetDeviceType, "layer"_a, IBuilderConfigDoc::reset_device_type)
        .def("can_run_on_DLA", &IBuilderConfig::canRunOnDLA, "layer"_a, IBuilderConfigDoc::can_run_on_DLA)
        .def_property(
            "profiling_verbosity", &IBuilderConfig::getProfilingVerbosity, &IBuilderConfig::setProfilingVerbosity)
        .def_property("algorithm_selector", &IBuilderConfig::getAlgorithmSelector,
            py::cpp_function(&IBuilderConfig::setAlgorithmSelector, py::keep_alive<1, 2>{}))
        .def("set_tactic_sources", &IBuilderConfig::setTacticSources, "tactic_sources"_a,
            IBuilderConfigDoc::set_tactic_sources)
        .def("get_tactic_sources", &IBuilderConfig::getTacticSources, IBuilderConfigDoc::get_tactic_sources)
        .def("create_timing_cache", lambdas::netconfig_create_timing_cache, "serialized_timing_cache"_a,
            IBuilderConfigDoc::create_timing_cache, py::call_guard<py::gil_scoped_release>{})
        .def("set_timing_cache", &IBuilderConfig::setTimingCache, "cache"_a, "ignore_mismatch"_a,
            IBuilderConfigDoc::set_timing_cache, py::keep_alive<1, 2>{})
        .def("get_timing_cache", &IBuilderConfig::getTimingCache, IBuilderConfigDoc::get_timing_cache)
        .def("set_preview_feature", &IBuilderConfig::setPreviewFeature, "feature"_a, "enable"_a,
            IBuilderConfigDoc::set_preview_feature)
        .def("get_preview_feature", &IBuilderConfig::getPreviewFeature, "feature"_a,
            IBuilderConfigDoc::get_preview_feature)
        .def_property("builder_optimization_level", &IBuilderConfig::getBuilderOptimizationLevel,
            &IBuilderConfig::setBuilderOptimizationLevel)
        .def_property("hardware_compatibility_level", &IBuilderConfig::getHardwareCompatibilityLevel,
            &IBuilderConfig::setHardwareCompatibilityLevel)
        .def_property("plugins_to_serialize", lambdas::get_plugins_to_serialize, lambdas::set_plugins_to_serialize)
        .def_property("max_aux_streams", &IBuilderConfig::getMaxAuxStreams, &IBuilderConfig::setMaxAuxStreams)
        .def_property("progress_monitor", &IBuilderConfig::getProgressMonitor,
            py::cpp_function(&IBuilderConfig::setProgressMonitor, py::keep_alive<1, 2>{}))
               .def("__del__", &utils::doNothingDel<IBuilderConfig>);

    py::enum_<NetworkDefinitionCreationFlag>(m, "NetworkDefinitionCreationFlag", py::arithmetic{},
        NetworkDefinitionCreationFlagDoc::descr, py::module_local())
        .value("EXPLICIT_BATCH", NetworkDefinitionCreationFlag::kEXPLICIT_BATCH,
            NetworkDefinitionCreationFlagDoc::EXPLICIT_BATCH)
        .value("STRONGLY_TYPED", NetworkDefinitionCreationFlag::kSTRONGLY_TYPED,
            NetworkDefinitionCreationFlagDoc::STRONGLY_TYPED);

    // Builder
    py::class_<IBuilder>(m, "Builder", BuilderDoc::descr, py::module_local())
        .def(py::init(&nvinfer1::createInferBuilder), "logger"_a, BuilderDoc::init, py::keep_alive<1, 2>{})
        .def("create_network", &IBuilder::createNetworkV2, "flags"_a = 0U, BuilderDoc::create_network,
            py::keep_alive<0, 1>{})
        .def_property_readonly("platform_has_tf32", &IBuilder::platformHasTf32)
        .def_property_readonly("platform_has_fast_fp16", &IBuilder::platformHasFastFp16)
        .def_property_readonly("platform_has_fast_int8", &IBuilder::platformHasFastInt8)
        .def_property_readonly("max_DLA_batch_size", &IBuilder::getMaxDLABatchSize)
        .def_property_readonly("num_DLA_cores", &IBuilder::getNbDLACores)
        .def_property("gpu_allocator", nullptr, py::cpp_function(&IBuilder::setGpuAllocator, py::keep_alive<1, 2>{}))
        .def("create_optimization_profile", &IBuilder::createOptimizationProfile,
            BuilderDoc::create_optimization_profile, py::return_value_policy::reference_internal)
        .def_property("error_recorder", &IBuilder::getErrorRecorder,
            py::cpp_function(&IBuilder::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def("create_builder_config", &IBuilder::createBuilderConfig, BuilderDoc::create_builder_config,
            py::keep_alive<0, 1>{})
        .def("build_serialized_network", &IBuilder::buildSerializedNetwork, "network"_a, "config"_a,
            BuilderDoc::build_serialized_network, py::call_guard<py::gil_scoped_release>{})
        .def("is_network_supported", &IBuilder::isNetworkSupported, "network"_a, "config"_a,
            BuilderDoc::is_network_supported, py::call_guard<py::gil_scoped_release>{})
        .def_property_readonly("logger", &IBuilder::getLogger)
        .def_property("max_threads", &IBuilder::getMaxThreads, &IBuilder::setMaxThreads)
        .def("reset", &IBuilder::reset, BuilderDoc::reset)
        .def("get_plugin_registry", &IBuilder::getPluginRegistry, py::return_value_policy::reference_internal,
            BuilderDoc::get_plugin_registry)
        .def("__del__", &utils::doNothingDel<IBuilder>);
#endif // EXPORT_ALL_BINDINGS

    // Runtime
    py::class_<IRuntime>(m, "Runtime", RuntimeDoc::descr, py::module_local())
        .def(py::init(&nvinfer1::createInferRuntime), "logger"_a, RuntimeDoc::init, py::keep_alive<1, 2>{})
        .def("deserialize_cuda_engine", lambdas::runtime_deserialize_cuda_engine, "serialized_engine"_a,
            RuntimeDoc::deserialize_cuda_engine, py::call_guard<py::gil_scoped_release>{}, py::keep_alive<0, 1>{})
        .def("deserialize_cuda_engine", py::overload_cast<IStreamReader&>(&IRuntime::deserializeCudaEngine),
            "stream_reader"_a, RuntimeDoc::deserialize_cuda_engine_reader, py::call_guard<py::gil_scoped_release>{},
            py::keep_alive<0, 1>{})
               .def_property("DLA_core", &IRuntime::getDLACore, &IRuntime::setDLACore)
        .def_property_readonly("num_DLA_cores", &IRuntime::getNbDLACores)
        .def_property("gpu_allocator", nullptr, py::cpp_function(&IRuntime::setGpuAllocator, py::keep_alive<1, 2>{}))
        .def_property("error_recorder", &IRuntime::getErrorRecorder,
            py::cpp_function(&IRuntime::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def_property_readonly("logger", &IRuntime::getLogger)
        .def_property("max_threads", &IRuntime::getMaxThreads, &IRuntime::setMaxThreads)
        .def_property("temporary_directory", &IRuntime::getTemporaryDirectory, &IRuntime::setTemporaryDirectory)
        .def_property("tempfile_control_flags", &IRuntime::getTempfileControlFlags, &IRuntime::setTempfileControlFlags)
        .def("get_plugin_registry", &IRuntime::getPluginRegistry, py::return_value_policy::reference_internal,
            RuntimeDoc::get_plugin_registry)
        .def("load_runtime", &IRuntime::loadRuntime, "path"_a, RuntimeDoc::load_runtime)
        .def_property(
            "engine_host_code_allowed", &IRuntime::getEngineHostCodeAllowed, &IRuntime::setEngineHostCodeAllowed)
        .def("__del__", &utils::doNothingDel<IRuntime>);

    // Refitter
    py::class_<IRefitter>(m, "Refitter", RefitterDoc::descr, py::module_local())
        .def(py::init(&nvinfer1::createInferRefitter), "engine"_a, "logger"_a, py::keep_alive<1, 2>{},
            py::keep_alive<1, 3>{}, RefitterDoc::init)
        .def("set_weights", &IRefitter::setWeights, "layer_name"_a, "role"_a, "weights"_a, py::keep_alive<1, 4>{},
            RefitterDoc::set_weights)
        .def("set_named_weights", py::overload_cast<char const*, Weights>(&IRefitter::setNamedWeights), "name"_a,
            "weights"_a, py::keep_alive<1, 3>{}, RefitterDoc::set_named_weights)
        .def("refit_cuda_engine", &IRefitter::refitCudaEngine, RefitterDoc::refit_cuda_engine,
            py::call_guard<py::gil_scoped_release>{})
        .def("get_missing", lambdas::refitter_get_missing, RefitterDoc::get_missing)
        .def("get_missing_weights", lambdas::refitter_get_missing_weights, RefitterDoc::get_missing_weights)
        .def("get_all", lambdas::refitter_get_all, RefitterDoc::get_all)
        .def("get_all_weights", lambdas::refitter_get_all_weights, RefitterDoc::get_all_weights)
        .def("get_dynamic_range", lambdas::refitter_get_dynamic_range, "tensor_name"_a, RefitterDoc::get_dynamic_range)
        .def("set_dynamic_range", lambdas::refitter_set_dynamic_range, "tensor_name"_a, "range"_a,
            RefitterDoc::set_dynamic_range)
        .def("get_tensors_with_dynamic_range", lambdas::refitter_get_tensors_with_dynamic_range,
            RefitterDoc::get_tensors_with_dynamic_range)
        .def_property("error_recorder", &IRefitter::getErrorRecorder,
            py::cpp_function(&IRefitter::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def_property_readonly("logger", &IRefitter::getLogger)
        .def_property("max_threads", &IRefitter::getMaxThreads, &IRefitter::setMaxThreads)
        .def("set_named_weights", py::overload_cast<char const*, Weights, TensorLocation>(&IRefitter::setNamedWeights),
            "name"_a, "weights"_a, "location"_a, py::keep_alive<1, 3>{}, RefitterDoc::set_named_weights_with_location)
        .def("get_named_weights", &IRefitter::getNamedWeights, "weights_name"_a, RefitterDoc::get_named_weights)
        .def(
            "get_weights_location", &IRefitter::getWeightsLocation, "weights_name"_a, RefitterDoc::get_weights_location)
        .def("unset_named_weights", &IRefitter::unsetNamedWeights, "weights_name"_a, RefitterDoc::unset_named_weights)
        .def_property("weights_validation", &IRefitter::getWeightsValidation, &IRefitter::setWeightsValidation)
        .def("refit_cuda_engine_async", lambdas::refitter_refit_cuda_engine_async, "stream_handle"_a,
            RefitterDoc::refit_cuda_engine_async, py::call_guard<py::gil_scoped_release>{})
        .def("get_weights_prototype", &IRefitter::getWeightsPrototype, "weights_name"_a,
            RefitterDoc::get_weights_prototype)
        .def("__del__", &utils::doNothingDel<IRefitter>);
}

} // namespace tensorrt
