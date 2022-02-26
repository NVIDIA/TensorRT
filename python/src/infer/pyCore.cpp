/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
    = [](IOptimizationProfile& self, const std::string& inputName, const Dims& min, const Dims& opt, const Dims& max) {
          if (!self.setDimensions(inputName.c_str(), OptProfileSelector::kMIN, min))
          {
              throw std::runtime_error{"Shape provided for min is inconsistent with other shapes."};
          }
          if (!self.setDimensions(inputName.c_str(), OptProfileSelector::kOPT, opt))
          {
              throw std::runtime_error{"Shape provided for opt is inconsistent with other shapes."};
          }
          if (!self.setDimensions(inputName.c_str(), OptProfileSelector::kMAX, max))
          {
              throw std::runtime_error{"Shape provided for max is inconsistent with other shapes."};
          }
      };

static const auto opt_profile_get_shape
    = [](IOptimizationProfile& self, const std::string& inputName) -> std::vector<Dims> {
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
    = [](IOptimizationProfile& self, const std::string& inputName, const std::vector<int32_t>& min,
          const std::vector<int32_t>& opt, const std::vector<int32_t>& max) {
          if (!self.setShapeValues(inputName.c_str(), OptProfileSelector::kMIN, min.data(), min.size()))
          {
              throw std::runtime_error{"min input provided for shape tensor is inconsistent with other inputs."};
          }
          if (!self.setShapeValues(inputName.c_str(), OptProfileSelector::kOPT, opt.data(), opt.size()))
          {
              throw std::runtime_error{"opt input provided for shape tensor is inconsistent with other inputs."};
          }
          if (!self.setShapeValues(inputName.c_str(), OptProfileSelector::kMAX, max.data(), max.size()))
          {
              throw std::runtime_error{"max input provided for shape tensor is inconsistent with other inputs."};
          }
      };

static const auto opt_profile_get_shape_input
    = [](IOptimizationProfile& self, const std::string& inputName) -> std::vector<std::vector<int32_t>> {
    std::vector<std::vector<int32_t>> shapes{};
    int shapeSize = self.getNbShapeValues(inputName.c_str());
    const int32_t* shapePtr = self.getShapeValues(inputName.c_str(), OptProfileSelector::kMIN);
    // In the Python bindings, it is impossible to set only one shape in an optimization profile.
    if (shapePtr && shapeSize >= 0)
    {
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
        if (!(shapePtr = self.getShapeValues(inputName.c_str(), OptProfileSelector::kOPT)))
        {
            throw std::runtime_error{"Invalid shape for OPT."};
        }
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
        if (!(shapePtr = self.getShapeValues(inputName.c_str(), OptProfileSelector::kMAX)))
        {
            throw std::runtime_error{"Invalid shape for MAX."};
        }
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
    }
    return shapes;
};

// For IExecutionContext
static const auto execute = [](IExecutionContext& self, int batchSize, std::vector<size_t>& bindings) {
    return self.execute(batchSize, reinterpret_cast<void**>(bindings.data()));
};

static const auto execute_async = [](IExecutionContext& self, int batchSize, std::vector<size_t>& bindings,
                                      size_t streamHandle, void* inputConsumed) {
    return self.enqueue(batchSize, reinterpret_cast<void**>(bindings.data()),
        reinterpret_cast<cudaStream_t>(streamHandle), reinterpret_cast<cudaEvent_t*>(inputConsumed));
};

static const auto execute_v2 = [](IExecutionContext& self, std::vector<size_t>& bindings) {
    return self.executeV2(reinterpret_cast<void**>(bindings.data()));
};

static const auto execute_async_v2
    = [](IExecutionContext& self, std::vector<size_t>& bindings, size_t streamHandle, void* inputConsumed) {
          return self.enqueueV2(reinterpret_cast<void**>(bindings.data()), reinterpret_cast<cudaStream_t>(streamHandle),
              reinterpret_cast<cudaEvent_t*>(inputConsumed));
      };

void context_set_optimization_profile(IExecutionContext& self, int32_t profileIndex)
{
    if (!self.setOptimizationProfile(profileIndex))
    {
        throw std::runtime_error{"Error in set optimization profile."};
    }
};

static const auto context_set_shape_input
    = [](IExecutionContext& self, int binding, const std::vector<int32_t>& shape) {
          return self.setInputShapeBinding(binding, shape.data());
      };

static const auto context_get_shape = [](IExecutionContext& self, int binding) {
    Dims shapeOfShape = self.getBindingDimensions(binding);
    int numVals = std::accumulate(shapeOfShape.d, shapeOfShape.d + shapeOfShape.nbDims, 1, std::multiplies<int>{});
    std::vector<int32_t> shape(numVals);
    if (!self.getShapeBinding(binding, shape.data()))
    {
        throw std::runtime_error{"Error in get shape bindings."};
    }
    return shape;
};

// For IRuntime
static const auto runtime_deserialize_cuda_engine = [](IRuntime& self, py::buffer& serializedEngine) {
    py::buffer_info info = serializedEngine.request();
    return self.deserializeCudaEngine(info.ptr, info.size * info.itemsize);
};

// For ICudaEngine
static const auto engine_binding_is_input = [](ICudaEngine& self, const std::string& name) {
    return self.bindingIsInput(self.getBindingIndex(name.c_str()));
};

static const auto engine_get_binding_shape = [](ICudaEngine& self, const std::string& name) {
    return self.getBindingDimensions(self.getBindingIndex(name.c_str()));
};

static const auto engine_get_binding_dtype = [](ICudaEngine& self, const std::string& name) {
    return self.getBindingDataType(self.getBindingIndex(name.c_str()));
};

static const auto engine_get_location
    = [](ICudaEngine& self, const std::string& name) { return self.getLocation(self.getBindingIndex(name.c_str())); };

// TODO: Add slicing support?
static const auto engine_getitem = [](ICudaEngine& self, int pyIndex) {
    // Support python's negative indexing
    size_t index = (pyIndex < 0) ? static_cast<int>(self.getNbBindings()) + pyIndex : pyIndex;
    if (index >= self.getNbBindings())
        throw py::index_error();
    return self.getBindingName(index);
};

static const auto engine_get_profile_shape
    = [](ICudaEngine& self, int profileIndex, int bindingIndex) -> std::vector<Dims> {
    std::vector<Dims> shapes{};
    shapes.emplace_back(self.getProfileDimensions(bindingIndex, profileIndex, OptProfileSelector::kMIN));
    shapes.emplace_back(self.getProfileDimensions(bindingIndex, profileIndex, OptProfileSelector::kOPT));
    shapes.emplace_back(self.getProfileDimensions(bindingIndex, profileIndex, OptProfileSelector::kMAX));
    return shapes;
};
// Overload to allow using binding names instead of indices.
static const auto engine_get_profile_shape_str
    = [](ICudaEngine& self, int profileIndex, const std::string& bindingName) -> std::vector<Dims> {
    return engine_get_profile_shape(self, profileIndex, self.getBindingIndex(bindingName.c_str()));
};

static const auto engine_get_profile_shape_input
    = [](ICudaEngine& self, int profileIndex, int bindingIndex) -> std::vector<std::vector<int32_t>> {
    if (!self.isShapeBinding(bindingIndex) || !self.bindingIsInput(bindingIndex))
    {
        throw std::runtime_error{
            "Binding index " + std::to_string(bindingIndex) + " does not correspond to an input shape tensor."};
    }
    std::vector<std::vector<int32_t>> shapes{};
    int shapeSize = self.getBindingDimensions(bindingIndex).nbDims;
    // In the Python bindings, it is impossible to set only one shape in an optimization profile.
    const int32_t* shapePtr = self.getProfileShapeValues(bindingIndex, profileIndex, OptProfileSelector::kMIN);
    if (shapePtr)
    {
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
        shapePtr = self.getProfileShapeValues(bindingIndex, profileIndex, OptProfileSelector::kOPT);
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
        shapePtr = self.getProfileShapeValues(bindingIndex, profileIndex, OptProfileSelector::kMAX);
        shapes.emplace_back(shapePtr, shapePtr + shapeSize);
    }
    return shapes;
};

// Overload to allow using binding names instead of indices.
static const auto engine_get_profile_shape_input_str
    = [](ICudaEngine& self, int profileIndex, const std::string& bindingName) -> std::vector<std::vector<int32_t>> {
    return engine_get_profile_shape_input(self, profileIndex, self.getBindingIndex(bindingName.c_str()));
};

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

// For IRefitter
static const auto refitter_get_missing = [](IRefitter& self) {
    // First get the number of missing weights.
    int size = self.getMissing(0, nullptr, nullptr);
    // Now that we know how many weights are missing, we can create the buffers appropriately.
    std::vector<const char*> layerNames(size);
    std::vector<WeightsRole> roles(size);
    self.getMissing(size, layerNames.data(), roles.data());
    return std::pair<std::vector<const char*>, std::vector<WeightsRole>>{layerNames, roles};
};

static const auto refitter_get_missing_weights = [](IRefitter& self) {
    // First get the number of missing weights.
    int size = self.getMissingWeights(0, nullptr);
    // Now that we know how many weights are missing, we can create the buffers appropriately.
    std::vector<const char*> names(size);
    self.getMissingWeights(size, names.data());
    return names;
};

static const auto refitter_get_all = [](IRefitter& self) {
    int size = self.getAll(0, nullptr, nullptr);
    std::vector<const char*> layerNames(size);
    std::vector<WeightsRole> roles(size);
    self.getAll(size, layerNames.data(), roles.data());
    return std::pair<std::vector<const char*>, std::vector<WeightsRole>>{layerNames, roles};
};

static const auto refitter_get_all_weights = [](IRefitter& self) {
    int size = self.getAllWeights(0, nullptr);
    std::vector<const char*> names(size);
    self.getAllWeights(size, names.data());
    return names;
};

static const auto refitter_get_dynamic_range = [](IRefitter& self, const std::string& tensorName) {
    return py::make_tuple(self.getDynamicRangeMin(tensorName.c_str()), self.getDynamicRangeMax(tensorName.c_str()));
};

static const auto refitter_set_dynamic_range
    = [](IRefitter& self, const std::string& tensorName, const std::vector<float>& range) -> bool {
    if (range.size() == 2)
    {
        return self.setDynamicRange(tensorName.c_str(), range[0], range[1]);
    }
    else
    {
        throw py::value_error{"Dynamic range must contain exactly 2 elements"};
    }
};

static const auto refitter_get_tensors_with_dynamic_range = [](IRefitter& self) {
    int size = self.getTensorsWithDynamicRange(0, nullptr);
    std::vector<const char*> tensorNames(size);
    self.getTensorsWithDynamicRange(size, tensorNames.data());
    return tensorNames;
};

static const auto context_set_optimization_profile_async
    = [](IExecutionContext& self, int profileIndex, size_t streamHandle) {
          if (!self.setOptimizationProfileAsync(profileIndex, reinterpret_cast<cudaStream_t>(streamHandle)))
          {
              throw std::runtime_error{"Error in set optimization profile async."};
          };
          return true;
      };

} // namespace lambdas

class PyGpuAllocator : public IGpuAllocator
{
public:
    using IGpuAllocator::IGpuAllocator;

    template <typename... Args>
    void* allocHelper(const char* pyFuncName, bool showWarning, Args&&... args) noexcept
    {
        py::gil_scoped_acquire gil{};
        py::function pyAllocFunc = utils::getOverride(static_cast<IGpuAllocator*>(this), pyFuncName, showWarning);

        if (!pyAllocFunc)
        {
            return nullptr;
        }

        py::object ptr{};
        try
        {
            ptr = pyAllocFunc(std::forward<Args>(args)...);
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in allocate()" << std::endl;
            return nullptr;
        }

        try
        {
            return reinterpret_cast<void*>(ptr.cast<size_t>());
        }
        catch (const py::cast_error& e)
        {
            std::cerr << "[ERROR] Return value of allocate() could not be interpreted as an int" << std::endl;
        }

        return nullptr;
    }

    void* allocate(uint64_t size, uint64_t alignment, AllocatorFlags flags) noexcept override
    {
        return allocHelper("allocate", true, size, alignment, flags);
    }

    void* reallocate(void* baseAddr, uint64_t alignment, uint64_t newSize) noexcept override
    {
        return allocHelper("reallocate", false, reinterpret_cast<size_t>(baseAddr), alignment, newSize);
    }

    void free(void* memory) noexcept override
    {
        py::gil_scoped_acquire gil{};
        py::function pyFree = utils::getOverride(static_cast<IGpuAllocator*>(this), "free");
        if (!pyFree)
        {
            return;
        }

        try
        {
            pyFree(reinterpret_cast<size_t>(memory));
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in free()" << std::endl;
        }
    }

    bool deallocate(void* memory) noexcept override
    {
        py::gil_scoped_acquire gil{};
        py::function pyDeallocate = utils::getOverride(static_cast<IGpuAllocator*>(this), "deallocate");
        if (!pyDeallocate)
        {
            return false;
        }

        py::object status{};
        try
        {
            status = pyDeallocate(reinterpret_cast<size_t>(memory));
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in deallocate()" << std::endl;
            return false;
        }
        return status.cast<bool>();
    }
};

void bindCore(py::module& m)
{
    class PyLogger : public ILogger
    {
    public:
        virtual void log(Severity severity, const char* msg) noexcept override
        {
            PYBIND11_OVERLOAD_PURE_NAME(void, ILogger, "log", log, severity, msg);
        }
    };

    py::class_<ILogger, PyLogger> baseLoggerBinding{m, "ILogger", ILoggerDoc::descr};
    baseLoggerBinding.def(py::init<>()).def("log", &ILogger::log, "severity"_a, "msg"_a, ILoggerDoc::log);

    py::enum_<ILogger::Severity>(baseLoggerBinding, "Severity", py::arithmetic(), SeverityDoc::descr)
        .value("INTERNAL_ERROR", ILogger::Severity::kINTERNAL_ERROR, SeverityDoc::internal_error)
        .value("ERROR", ILogger::Severity::kERROR, SeverityDoc::error)
        .value("WARNING", ILogger::Severity::kWARNING, SeverityDoc::warning)
        .value("INFO", ILogger::Severity::kINFO, SeverityDoc::info)
        .value("VERBOSE", ILogger::Severity::kVERBOSE, SeverityDoc::verbose)
        // We export into the outer scope, so we can access with trt.ILogger.X.
        .export_values();

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

    py::class_<DefaultLogger, ILogger>(m, "Logger", LoggerDoc::descr)
        .def(py::init<ILogger::Severity>(), "min_severity"_a = ILogger::Severity::kWARNING)
        .def_readwrite("min_severity", &DefaultLogger::mMinSeverity)
        .def("log", &DefaultLogger::log, "severity"_a, "msg"_a, LoggerDoc::log);

    class PyProfiler : public IProfiler
    {
    public:
        void reportLayerTime(const char* layerName, float ms) noexcept override
        {
            PYBIND11_OVERLOAD_PURE_NAME(void, IProfiler, "report_layer_time", reportLayerTime, layerName, ms);
        }
    };

    py::class_<IProfiler, PyProfiler>(m, "IProfiler", IProfilerDoc::descr)
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

    py::class_<DefaultProfiler, IProfiler>(m, "Profiler", ProfilerDoc::descr)
        .def(py::init<>())
        .def("report_layer_time", &IProfiler::reportLayerTime, "layer_name"_a, "ms"_a, ProfilerDoc::report_layer_time);

    py::class_<IOptimizationProfile, std::unique_ptr<IOptimizationProfile, py::nodelete>>(
        m, "IOptimizationProfile", IOptimizationProfileDoc::descr)
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

    py::enum_<ErrorCode>(m, "ErrorCodeTRT", py::arithmetic{}, ErrorCodeDoc::descr)
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
            PYBIND11_OVERLOAD_PURE_NAME(ErrorCode, IErrorRecorder, "get_error_code", getErrorCode, errorIdx);
        }

        virtual ErrorDesc getErrorDesc(int32_t errorIdx) const noexcept override
        {
            PYBIND11_OVERLOAD_PURE_NAME(ErrorDesc, IErrorRecorder, "get_error_desc", getErrorDesc, errorIdx);
        }

        virtual void clear() noexcept override
        {
            PYBIND11_OVERLOAD_PURE_NAME(void, IErrorRecorder, "clear", clear);
        }

        virtual bool reportError(ErrorCode val, ErrorDesc desc) noexcept override
        {
            PYBIND11_OVERLOAD_PURE_NAME(bool, IErrorRecorder, "report_error", reportError, val, desc);
        }

        virtual int32_t getNbErrors() const noexcept override
        {
            PYBIND11_OVERLOAD_PURE_NAME(int32_t, IErrorRecorder, "get_num_errors", getNbErrors);
        }

        virtual bool hasOverflowed() const noexcept override
        {
            PYBIND11_OVERLOAD_PURE_NAME(bool, IErrorRecorder, "has_overflowed", hasOverflowed);
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

    py::class_<IErrorRecorder, PyErrorRecorder>(m, "IErrorRecorder", IErrorRecorderDoc::descr)
        .def(py::init<>())
        .def_property_readonly("MAX_DESC_LENGTH", []() { return IErrorRecorder::kMAX_DESC_LENGTH; })
        .def("num_errors", &IErrorRecorder::getNbErrors, IErrorRecorderDoc::get_num_errors)
        .def("get_error_code", &IErrorRecorder::getErrorCode, IErrorRecorderDoc::get_error_code)
        .def("get_error_desc", &IErrorRecorder::getErrorDesc, IErrorRecorderDoc::get_error_desc)
        .def("has_overflowed", &IErrorRecorder::hasOverflowed, IErrorRecorderDoc::has_overflowed)
        .def("clear", &IErrorRecorder::clear, IErrorRecorderDoc::clear)
        .def("report_error", &IErrorRecorder::reportError, IErrorRecorderDoc::report_error);

    py::class_<IExecutionContext>(m, "IExecutionContext", IExecutionContextDoc::descr)
        .def("execute", lambdas::execute, "batch_size"_a = 1, "bindings"_a, IExecutionContextDoc::execute,
            py::call_guard<py::gil_scoped_release>{})
        .def("execute_async", lambdas::execute_async, "batch_size"_a = 1, "bindings"_a, "stream_handle"_a,
            "input_consumed"_a = nullptr, IExecutionContextDoc::execute_async, py::call_guard<py::gil_scoped_release>{})
        .def("execute_v2", lambdas::execute_v2, "bindings"_a, IExecutionContextDoc::execute_v2,
            py::call_guard<py::gil_scoped_release>{})
        .def("execute_async_v2", lambdas::execute_async_v2, "bindings"_a, "stream_handle"_a,
            "input_consumed"_a = nullptr, IExecutionContextDoc::execute_async_v2,
            py::call_guard<py::gil_scoped_release>{})
        .def_property("debug_sync", &IExecutionContext::getDebugSync, &IExecutionContext::setDebugSync)
        .def_property("profiler", &IExecutionContext::getProfiler,
            py::cpp_function(&IExecutionContext::setProfiler, py::keep_alive<1, 2>{}))
        .def_property_readonly("engine", &IExecutionContext::getEngine)
        .def_property(
            "name", &IExecutionContext::getName, py::cpp_function(&IExecutionContext::setName, py::keep_alive<1, 2>{}))
        // For writeonly properties, we use a nullptr getter.
        // TODO: Does this make sense in Python?
        .def_property("device_memory", nullptr, &IExecutionContext::setDeviceMemory)
        .def_property("active_optimization_profile", &IExecutionContext::getOptimizationProfile,
            utils::deprecate(lambdas::context_set_optimization_profile, "set_optimization_profile_async"))
        .def("get_strides", &IExecutionContext::getStrides, "binding"_a, IExecutionContextDoc::get_strides)
        .def("set_binding_shape", &IExecutionContext::setBindingDimensions, "binding"_a, "shape"_a,
            IExecutionContextDoc::set_binding_shape)
        .def("get_binding_shape", &IExecutionContext::getBindingDimensions, "binding"_a,
            IExecutionContextDoc::get_binding_shape)
        .def("set_shape_input", lambdas::context_set_shape_input, "binding"_a, "shape"_a,
            IExecutionContextDoc::set_shape_input)
        .def("get_shape", lambdas::context_get_shape, "binding"_a, IExecutionContextDoc::get_shape)
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
        .def("__del__", &utils::doNothingDel<IExecutionContext>);

    py::class_<ICudaEngine>(m, "ICudaEngine", ICudaEngineDoc::descr)
        .def_property_readonly("num_bindings", &ICudaEngine::getNbBindings)
        .def("__len__", &ICudaEngine::getNbBindings)
        .def("__getitem__",
            [](ICudaEngine& self, const std::string& name) { return self.getBindingIndex(name.c_str()); })
        .def("__getitem__", lambdas::engine_getitem)
        .def("get_binding_name", &ICudaEngine::getBindingName, "index"_a, ICudaEngineDoc::get_binding_name)
        .def("get_binding_index", &ICudaEngine::getBindingIndex, "name"_a, ICudaEngineDoc::get_binding_index)
        .def("binding_is_input", &ICudaEngine::bindingIsInput, "index"_a, ICudaEngineDoc::binding_is_input)
        .def("binding_is_input", lambdas::engine_binding_is_input, "name"_a, ICudaEngineDoc::binding_is_input_str)
        .def("get_binding_shape", &ICudaEngine::getBindingDimensions, "index"_a, ICudaEngineDoc::get_binding_shape)
        // Overload so that we can get shape based on tensor names.
        .def("get_binding_shape", lambdas::engine_get_binding_shape, "name"_a, ICudaEngineDoc::get_binding_shape_str)
        .def("get_binding_dtype", &ICudaEngine::getBindingDataType, "index"_a, ICudaEngineDoc::get_binding_dtype)
        // Overload so that we can get type based on tensor names.
        .def("get_binding_dtype", lambdas::engine_get_binding_dtype, "name"_a, ICudaEngineDoc::get_binding_dtype_str)
        .def_property_readonly("has_implicit_batch_dimension", &ICudaEngine::hasImplicitBatchDimension)
        .def_property_readonly("max_batch_size", &ICudaEngine::getMaxBatchSize)
        .def_property_readonly("num_layers", &ICudaEngine::getNbLayers)
        .def("serialize", &ICudaEngine::serialize, ICudaEngineDoc::serialize)
        .def("create_execution_context", &ICudaEngine::createExecutionContext, ICudaEngineDoc::create_execution_context,
            py::keep_alive<0, 1>{})
        .def("get_location", &ICudaEngine::getLocation, "index"_a, ICudaEngineDoc::get_location)
        .def("get_location", lambdas::engine_get_location, "name"_a, ICudaEngineDoc::get_location_str)
        .def("create_execution_context_without_device_memory", &ICudaEngine::createExecutionContextWithoutDeviceMemory,
            ICudaEngineDoc::create_execution_context_without_device_memory, py::keep_alive<0, 1>{})
        .def_property_readonly("device_memory_size", &ICudaEngine::getDeviceMemorySize)
        .def_property_readonly("refittable", &ICudaEngine::isRefittable)
        .def_property_readonly("name", &ICudaEngine::getName)
        .def_property_readonly("num_optimization_profiles", &ICudaEngine::getNbOptimizationProfiles)
        .def_property_readonly("engine_capability", &ICudaEngine::getEngineCapability)
        .def("get_profile_shape", lambdas::engine_get_profile_shape, "profile_index"_a, "binding"_a,
            ICudaEngineDoc::get_profile_shape)
        .def("get_profile_shape", lambdas::engine_get_profile_shape_str, "profile_index"_a, "binding"_a,
            ICudaEngineDoc::get_profile_shape)
        .def("get_profile_shape_input", lambdas::engine_get_profile_shape_input, "profile_index"_a, "binding"_a,
            ICudaEngineDoc::get_profile_shape_input)
        .def("get_profile_shape_input", lambdas::engine_get_profile_shape_input_str, "profile_index"_a, "binding"_a,
            ICudaEngineDoc::get_profile_shape_input)
        .def("is_shape_binding", &ICudaEngine::isShapeBinding, "binding"_a, ICudaEngineDoc::is_shape_binding)
        .def(
            "is_execution_binding", &ICudaEngine::isExecutionBinding, "binding"_a, ICudaEngineDoc::is_execution_binding)
        .def("get_binding_bytes_per_component", &ICudaEngine::getBindingBytesPerComponent, "index"_a,
            ICudaEngineDoc::get_binding_bytes_per_component)
        .def("get_binding_components_per_element", &ICudaEngine::getBindingComponentsPerElement, "index"_a,
            ICudaEngineDoc::get_binding_components_per_element)
        .def("get_binding_format", &ICudaEngine::getBindingFormat, "index"_a, ICudaEngineDoc::get_binding_format)
        .def("get_binding_format_desc", &ICudaEngine::getBindingFormatDesc, "index"_a,
            ICudaEngineDoc::get_binding_format_desc)
        .def("get_binding_vectorized_dim", &ICudaEngine::getBindingVectorizedDim, "index"_a,
            ICudaEngineDoc::get_binding_vectorized_dim)
        .def_property("error_recorder", &ICudaEngine::getErrorRecorder,
            py::cpp_function(&ICudaEngine::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def_property_readonly("tactic_sources", &ICudaEngine::getTacticSources)
        .def_property_readonly("profiling_verbosity", &ICudaEngine::getProfilingVerbosity)
        .def("__del__", &utils::doNothingDel<ICudaEngine>)
        .def("create_engine_inspector", &ICudaEngine::createEngineInspector, ICudaEngineDoc::create_engine_inspector);

    py::enum_<AllocatorFlag>(m, "AllocatorFlag", py::arithmetic{}, AllocatorFlagDoc::descr)
        .value("RESIZABLE", AllocatorFlag::kRESIZABLE, AllocatorFlagDoc::RESIZABLE);

    py::class_<IGpuAllocator, PyGpuAllocator>(m, "IGpuAllocator", GpuAllocatorDoc::descr)
        .def(py::init<>())
        .def("allocate", &IGpuAllocator::allocate, "size"_a, "alignment"_a, "flags"_a, GpuAllocatorDoc::allocate)
        .def("reallocate", &IGpuAllocator::reallocate, "address"_a, "alignment"_a, "new_size"_a,
            GpuAllocatorDoc::reallocate)
        .def("free", &IGpuAllocator::free, "memory"_a, GpuAllocatorDoc::free)
        .def("deallocate", &IGpuAllocator::deallocate, "memory"_a, GpuAllocatorDoc::deallocate);

    py::enum_<BuilderFlag>(m, "BuilderFlag", py::arithmetic{}, BuilderFlagDoc::descr)
        .value("FP16", BuilderFlag::kFP16, BuilderFlagDoc::FP16)
        .value("INT8", BuilderFlag::kINT8, BuilderFlagDoc::INT8)
        .value("DEBUG", BuilderFlag::kDEBUG, BuilderFlagDoc::DEBUG)
        .value("GPU_FALLBACK", BuilderFlag::kGPU_FALLBACK, BuilderFlagDoc::GPU_FALLBACK)
        .value("STRICT_TYPES", BuilderFlag::kSTRICT_TYPES, BuilderFlagDoc::STRICT_TYPES)
        .value("REFIT", BuilderFlag::kREFIT, BuilderFlagDoc::REFIT)
        .value("DISABLE_TIMING_CACHE", BuilderFlag::kDISABLE_TIMING_CACHE, BuilderFlagDoc::DISABLE_TIMING_CACHE)
        .value("TF32", BuilderFlag::kTF32, BuilderFlagDoc::TF32)
        .value("SPARSE_WEIGHTS", BuilderFlag::kSPARSE_WEIGHTS, BuilderFlagDoc::SPARSE_WEIGHTS)
        .value("SAFETY_SCOPE", BuilderFlag::kSAFETY_SCOPE, BuilderFlagDoc::SAFETY_SCOPE);

    py::enum_<QuantizationFlag>(m, "QuantizationFlag", py::arithmetic{}, QuantizationFlagDoc::descr)
        .value("CALIBRATE_BEFORE_FUSION", QuantizationFlag::kCALIBRATE_BEFORE_FUSION,
            QuantizationFlagDoc::CALIBRATE_BEFORE_FUSION);

    py::enum_<DeviceType>(m, "DeviceType", DeviceTypeDoc::descr)
        .value("GPU", DeviceType::kGPU, DeviceTypeDoc::GPU)
        .value("DLA", DeviceType::kDLA, DeviceTypeDoc::DLA);

    // Bind to a Python enum called ProfilingVerbosity.
    py::enum_<ProfilingVerbosity>(m, "ProfilingVerbosity", ProfilingVerbosityDoc::descr)
        .value("LAYER_NAMES_ONLY", ProfilingVerbosity::kLAYER_NAMES_ONLY, ProfilingVerbosityDoc::LAYER_NAMES_ONLY)
        .value("DETAILED", ProfilingVerbosity::kDETAILED, ProfilingVerbosityDoc::DETAILED)
        .value("NONE", ProfilingVerbosity::kNONE, ProfilingVerbosityDoc::NONE)
        .value("DEFAULT", ProfilingVerbosity::kDEFAULT, ProfilingVerbosityDoc::DEFAULT)
        .value("VERBOSE", ProfilingVerbosity::kVERBOSE, ProfilingVerbosityDoc::VERBOSE);

    py::enum_<TacticSource>(m, "TacticSource", py::arithmetic{}, TacticSourceDoc::descr)
        .value("CUBLAS", TacticSource::kCUBLAS, TacticSourceDoc::CUBLAS)
        .value("CUBLAS_LT", TacticSource::kCUBLAS_LT, TacticSourceDoc::CUBLAS_LT)
        .value("CUDNN", TacticSource::kCUDNN, TacticSourceDoc::CUDNN);

    py::enum_<EngineCapability>(m, "EngineCapability", py::arithmetic{}, EngineCapabilityDoc::descr)
        .value("DEFAULT", EngineCapability::kDEFAULT, EngineCapabilityDoc::DEFAULT)
        .value("SAFE_GPU", EngineCapability::kSAFE_GPU, EngineCapabilityDoc::SAFE_GPU)
        .value("SAFE_DLA", EngineCapability::kSAFE_DLA, EngineCapabilityDoc::SAFE_DLA)
        .value("STANDARD", EngineCapability::kSTANDARD, EngineCapabilityDoc::STANDARD)
        .value("SAFETY", EngineCapability::kSAFETY, EngineCapabilityDoc::SAFETY)
        .value("DLA_STANDALONE", EngineCapability::kDLA_STANDALONE, EngineCapabilityDoc::DLA_STANDALONE);

    py::enum_<LayerInformationFormat>(m, "LayerInformationFormat", LayerInformationFormatDoc::descr)
        .value("ONELINE", LayerInformationFormat::kONELINE, LayerInformationFormatDoc::ONELINE)
        .value("JSON", LayerInformationFormat::kJSON, LayerInformationFormatDoc::JSON);

    py::class_<ITimingCache>(m, "ITimingCache", ITimingCacheDoc::descr)
        .def("serialize", &ITimingCache::serialize, ITimingCacheDoc::serialize)
        .def("combine", &ITimingCache::combine, "input_cache"_a, "ignore_mismatch"_a, ITimingCacheDoc::combine)
        .def("reset", &ITimingCache::reset, ITimingCacheDoc::reset);

    py::class_<IBuilderConfig>(m, "IBuilderConfig", IBuilderConfigDoc::descr)
        .def_property(
            "min_timing_iterations", &IBuilderConfig::getMinTimingIterations, &IBuilderConfig::setMinTimingIterations)
        .def_property(
            "avg_timing_iterations", &IBuilderConfig::getAvgTimingIterations, &IBuilderConfig::setAvgTimingIterations)
        .def_property("int8_calibrator", &IBuilderConfig::getInt8Calibrator,
            py::cpp_function(&IBuilderConfig::setInt8Calibrator, py::keep_alive<1, 2>{}))
        .def_property("engine_capability", &IBuilderConfig::getEngineCapability, &IBuilderConfig::setEngineCapability)
        .def_property("max_workspace_size", &IBuilderConfig::getMaxWorkspaceSize, &IBuilderConfig::setMaxWorkspaceSize)
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
        .def("__del__", &utils::doNothingDel<IBuilderConfig>);

    py::enum_<NetworkDefinitionCreationFlag>(
        m, "NetworkDefinitionCreationFlag", py::arithmetic{}, NetworkDefinitionCreationFlagDoc::descr)
        .value("EXPLICIT_BATCH", NetworkDefinitionCreationFlag::kEXPLICIT_BATCH,
            NetworkDefinitionCreationFlagDoc::EXPLICIT_BATCH)
        .value("EXPLICIT_PRECISION", NetworkDefinitionCreationFlag::kEXPLICIT_PRECISION,
            NetworkDefinitionCreationFlagDoc::EXPLICIT_PRECISION);

    // Builder
    py::class_<IBuilder>(m, "Builder", BuilderDoc::descr)
        .def(py::init(&nvinfer1::createInferBuilder), "logger"_a, BuilderDoc::init, py::keep_alive<1, 2>{})
        .def("create_network", &IBuilder::createNetworkV2, "flags"_a = 0U, BuilderDoc::create_network,
            py::keep_alive<0, 1>{})
        .def_property("max_batch_size", &IBuilder::getMaxBatchSize, &IBuilder::setMaxBatchSize)
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
        .def("build_engine", utils::deprecateMember(&IBuilder::buildEngineWithConfig, "build_serialized_network"),
            "network"_a, "config"_a, BuilderDoc::build_engine, py::call_guard<py::gil_scoped_release>{},
            py::keep_alive<0, 1>{})
        .def("build_serialized_network", &IBuilder::buildSerializedNetwork, "network"_a, "config"_a,
            BuilderDoc::build_serialized_network, py::call_guard<py::gil_scoped_release>{})
        .def("is_network_supported", &IBuilder::isNetworkSupported, "network"_a, "config"_a,
            BuilderDoc::is_network_supported, py::call_guard<py::gil_scoped_release>{})
        .def_property_readonly("logger", &IBuilder::getLogger)
        .def("reset", &IBuilder::reset, BuilderDoc::reset)
        .def("__del__", &utils::doNothingDel<IBuilder>);

    // Runtime
    py::class_<IRuntime>(m, "Runtime", RuntimeDoc::descr)
        .def(py::init(&nvinfer1::createInferRuntime), "logger"_a, RuntimeDoc::init, py::keep_alive<1, 2>{})
        .def("deserialize_cuda_engine", lambdas::runtime_deserialize_cuda_engine, "serialized_engine"_a,
            RuntimeDoc::deserialize_cuda_engine, py::call_guard<py::gil_scoped_release>{}, py::keep_alive<0, 1>{})
        .def_property("DLA_core", &IRuntime::getDLACore, &IRuntime::setDLACore)
        .def_property_readonly("num_DLA_cores", &IRuntime::getNbDLACores)
        .def_property("gpu_allocator", nullptr, py::cpp_function(&IRuntime::setGpuAllocator, py::keep_alive<1, 2>{}))
        .def_property("error_recorder", &IRuntime::getErrorRecorder,
            py::cpp_function(&IRuntime::setErrorRecorder, py::keep_alive<1, 2>{}))
        .def_property_readonly("logger", &IRuntime::getLogger)
        .def("__del__", &utils::doNothingDel<IRuntime>);

    // EngineInspector
    py::class_<IEngineInspector>(m, "EngineInspector", RuntimeInspectorDoc::descr)
        .def_property(
            "execution_context", &IEngineInspector::getExecutionContext, &IEngineInspector::setExecutionContext)
        .def("get_layer_information", &IEngineInspector::getLayerInformation, "layer_index"_a, "format"_a,
            RuntimeInspectorDoc::get_layer_information)
        .def("get_engine_information", &IEngineInspector::getEngineInformation, "format"_a,
            RuntimeInspectorDoc::get_engine_information)
        .def_property("error_recorder", &IEngineInspector::getErrorRecorder,
            py::cpp_function(&IEngineInspector::setErrorRecorder, py::keep_alive<1, 2>{}));

    // Refitter
    py::class_<IRefitter>(m, "Refitter", RefitterDoc::descr)
        .def(py::init(&nvinfer1::createInferRefitter), "engine"_a, "logger"_a, py::keep_alive<1, 2>{},
            py::keep_alive<1, 3>{}, RefitterDoc::init)
        .def("set_weights", &IRefitter::setWeights, "layer_name"_a, "role"_a, "weights"_a, py::keep_alive<1, 4>{},
            RefitterDoc::set_weights)
        .def("set_named_weights", &IRefitter::setNamedWeights, "name"_a, "weights"_a, py::keep_alive<1, 3>{},
            RefitterDoc::set_named_weights)
        .def("refit_cuda_engine", &IRefitter::refitCudaEngine, RefitterDoc::refit_cuda_engine)
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
        .def("__del__", &utils::doNothingDel<IRefitter>);
}

} // namespace tensorrt
