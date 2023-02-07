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

// This contains int8 calibration related things.
#include "ForwardDeclarations.h"
#include "infer/pyInt8Doc.h"
#include "utils.h"
#include <pybind11/stl.h>

using namespace nvinfer1;

namespace tensorrt
{
// Use CRTP to share code among several different classes.
template <typename Derived>
class pyCalibratorTrampoline : public Derived
{
public:
    using Derived::Derived; // Inherit constructors

    int getBatchSize() const noexcept override
    {
        try
        {
            PYBIND11_OVERLOAD_PURE_NAME(int, Derived, "get_batch_size", getBatchSize);
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in get_batch_size(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in get_batch_size()" << std::endl;
        }
        return -1;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyGetBatch = utils::getOverride(static_cast<Derived*>(this), "get_batch");
            std::vector<const char*> namesVec(names, names + nbBindings);
            py::object result = pyGetBatch(namesVec);
            // Copy over into the other data structure.
            if (!result.is_none() && result.cast<std::vector<size_t>>().size() != 0)
            {
                std::memcpy(bindings, result.cast<std::vector<size_t>>().data(), nbBindings * sizeof(void*));
                return true;
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in get_batch(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in get_batch()" << std::endl;
        }
        return false;
    }

    const void* readCalibrationCache(std::size_t& length) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyReadCalibrationCache = utils::getOverride(static_cast<Derived*>(this), "read_calibration_cache");

            // Cannot cast `None` to py::buffer.
            auto cacheRaw = pyReadCalibrationCache();
            if (cacheRaw.is_none())
            {
                return nullptr;
            }

            mCache = py::buffer{cacheRaw};
            {
                py::buffer_info info = mCache.request();
                length = info.size * info.itemsize;
                return info.ptr;
            }
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in read_calibration_cache(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in read_calibration_cache()" << std::endl;
        }
        return nullptr;
    }

    void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override
    {
        try
        {
            py::gil_scoped_acquire gil{};

            py::function pyWriteCalibrationCache
                = utils::getOverride(static_cast<Derived*>(this), "write_calibration_cache");

    #if PYBIND11_VERSION_MAJOR < 2 || PYBIND11_VERSION_MAJOR == 2 && PYBIND11_VERSION_MINOR < 6
            py::buffer_info info{
                const_cast<void*>(ptr),                   /* Pointer to buffer */
                sizeof(uint8_t),                          /* Size of one scalar */
                py::format_descriptor<uint8_t>::format(), /* Python struct-style format descriptor */
                1,                                        /* Number of dimensions */
                {length},                                 /* Buffer dimensions */
                { sizeof(uint8_t) }                       /* Strides (in bytes) for each index */
            };
            py::memoryview cache{info};
    #else
            py::memoryview cache{
                py::memoryview::from_buffer(static_cast<const uint8_t*>(ptr), {length}, {sizeof(uint8_t)})};
    #endif
            pyWriteCalibrationCache(cache);
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in write_calibration_cache(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in write_calibration_cache()" << std::endl;
        }
    }

private:
    py::buffer mCache{};
};

class pyIInt8Calibrator : public pyCalibratorTrampoline<IInt8Calibrator>
{
public:
    using Derived = pyCalibratorTrampoline<IInt8Calibrator>;
    using Derived::Derived;

    CalibrationAlgoType getAlgorithm() noexcept override
    {
        try
        {
            PYBIND11_OVERLOAD_PURE_NAME(CalibrationAlgoType, IInt8Calibrator, "get_algorithm", getAlgorithm);
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in get_algorithm(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in get_algorithm()" << std::endl;
        }
        return {};
    }
};

class pyIInt8LegacyCalibrator : public pyCalibratorTrampoline<IInt8LegacyCalibrator>
{
public:
    using Derived = pyCalibratorTrampoline<IInt8LegacyCalibrator>;
    using Derived::Derived;

    double getQuantile() const noexcept override
    {
        try
        {
            PYBIND11_OVERLOAD_PURE_NAME(double, IInt8LegacyCalibrator, "get_quantile", getQuantile);
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in get_quantile(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in get_quantile()" << std::endl;
        }
        return {};
    }

    double getRegressionCutoff() const noexcept override
    {
        try
        {
            PYBIND11_OVERLOAD_PURE_NAME(double, IInt8LegacyCalibrator, "get_regression_cutoff", getRegressionCutoff);
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in get_regression_cutoff(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in get_regression_cutoff()" << std::endl;
        }
        return {};
    }

    const void* readHistogramCache(std::size_t& length) noexcept override
    {
        try
        {
            PYBIND11_OVERLOAD_PURE_NAME(
                const void*, IInt8LegacyCalibrator, "read_histogram_cache", readHistogramCache, length);
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in read_histogram_cache(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in read_histogram_cache()" << std::endl;
        }
        return {};
    }

    void writeHistogramCache(const void* ptr, std::size_t length) noexcept override
    {
        try
        {
            PYBIND11_OVERLOAD_PURE_NAME(
                void, IInt8LegacyCalibrator, "write_histogram_cache", writeHistogramCache, ptr, length);
        }
        catch (std::exception const& e)
        {
            std::cerr << "[ERROR] Exception caught in write_histogram_cache(): " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "[ERROR] Exception caught in write_histogram_cache()" << std::endl;
        }
    }
};

// NOTE: Fake bindings are provided for some of the application-implemented functions here.
// These are solely for documentation purposes. The user is meant to override these functions
// in their own code, and the bindings here will never be called.

template <typename T>
std::vector<size_t> docGetBatch(T&, const std::vector<std::string>&)
{
    return {};
}

template <typename T>
py::buffer docReadCalibrationCache(T&)
{
    return {};
}

template <typename T>
void docWriteCalibrationCache(T&, py::buffer)
{
}

void bindInt8(py::module& m)
{
    py::enum_<CalibrationAlgoType>(m, "CalibrationAlgoType", CalibrationAlgoTypeDoc::descr)
        .value("LEGACY_CALIBRATION", CalibrationAlgoType::kLEGACY_CALIBRATION)
        .value("ENTROPY_CALIBRATION", CalibrationAlgoType::kENTROPY_CALIBRATION)
        .value("ENTROPY_CALIBRATION_2", CalibrationAlgoType::kENTROPY_CALIBRATION_2)
        .value("MINMAX_CALIBRATION", CalibrationAlgoType::kMINMAX_CALIBRATION);

    py::class_<IInt8Calibrator, pyIInt8Calibrator>(m, "IInt8Calibrator", IInt8CalibratorDoc::descr)
        .def(py::init<>())
        .def("get_batch_size", &IInt8Calibrator::getBatchSize, IInt8CalibratorDoc::get_batch_size)
        .def("get_algorithm", &IInt8Calibrator::getAlgorithm, IInt8CalibratorDoc::get_algorithm)
        // For documentation purposes only
        .def("get_batch", docGetBatch<IInt8Calibrator>, "names"_a, IInt8CalibratorDoc::get_batch)
        .def("read_calibration_cache", docReadCalibrationCache<IInt8Calibrator>,
            IInt8CalibratorDoc::read_calibration_cache)
        .def("write_calibration_cache", docWriteCalibrationCache<IInt8Calibrator>, "cache"_a,
            IInt8CalibratorDoc::write_calibration_cache);

    py::class_<IInt8LegacyCalibrator, IInt8Calibrator, pyIInt8LegacyCalibrator>(
        m, "IInt8LegacyCalibrator", IInt8LegacyCalibratorDoc::descr)
        .def(py::init<>())
        .def("get_batch_size", &IInt8LegacyCalibrator::getBatchSize, IInt8CalibratorDoc::get_batch_size)
        .def("get_algorithm", &IInt8LegacyCalibrator::getAlgorithm, IInt8LegacyCalibratorDoc::get_algorithm)
        // For documentation purposes only
        .def("get_batch", docGetBatch<IInt8LegacyCalibrator>, "names"_a, IInt8CalibratorDoc::get_batch)
        .def("read_calibration_cache", docReadCalibrationCache<IInt8LegacyCalibrator>,
            IInt8CalibratorDoc::read_calibration_cache)
        .def("write_calibration_cache", docWriteCalibrationCache<IInt8LegacyCalibrator>, "cache"_a,
            IInt8CalibratorDoc::write_calibration_cache);

    py::class_<IInt8EntropyCalibrator, IInt8Calibrator, pyCalibratorTrampoline<IInt8EntropyCalibrator>>(
        m, "IInt8EntropyCalibrator", IInt8EntropyCalibratorDoc::descr)
        .def(py::init<>())
        .def("get_batch_size", &IInt8EntropyCalibrator::getBatchSize, IInt8CalibratorDoc::get_batch_size)
        .def("get_algorithm", &IInt8EntropyCalibrator::getAlgorithm, IInt8EntropyCalibratorDoc::get_algorithm)
        // For documentation purposes only
        .def("get_batch", docGetBatch<IInt8EntropyCalibrator>, "names"_a, IInt8CalibratorDoc::get_batch)
        .def("read_calibration_cache", docReadCalibrationCache<IInt8EntropyCalibrator>,
            IInt8CalibratorDoc::read_calibration_cache)
        .def("write_calibration_cache", docWriteCalibrationCache<IInt8EntropyCalibrator>, "cache"_a,
            IInt8CalibratorDoc::write_calibration_cache);

    py::class_<IInt8EntropyCalibrator2, IInt8Calibrator, pyCalibratorTrampoline<IInt8EntropyCalibrator2>>(
        m, "IInt8EntropyCalibrator2", IInt8EntropyCalibrator2Doc::descr)
        .def(py::init<>())
        .def("get_batch_size", &IInt8EntropyCalibrator2::getBatchSize, IInt8CalibratorDoc::get_batch_size)
        .def("get_algorithm", &IInt8EntropyCalibrator2::getAlgorithm, IInt8EntropyCalibrator2Doc::get_algorithm)
        // For documentation purposes only
        .def("get_batch", docGetBatch<IInt8EntropyCalibrator2>, "names"_a, IInt8CalibratorDoc::get_batch)
        .def("read_calibration_cache", docReadCalibrationCache<IInt8EntropyCalibrator2>,
            IInt8CalibratorDoc::read_calibration_cache)
        .def("write_calibration_cache", docWriteCalibrationCache<IInt8EntropyCalibrator2>, "cache"_a,
            IInt8CalibratorDoc::write_calibration_cache);

    py::class_<IInt8MinMaxCalibrator, IInt8Calibrator, pyCalibratorTrampoline<IInt8MinMaxCalibrator>>(
        m, "IInt8MinMaxCalibrator", IInt8MinMaxCalibratorDoc::descr)
        .def(py::init<>())
        .def("get_batch_size", &IInt8MinMaxCalibrator::getBatchSize, IInt8CalibratorDoc::get_batch_size)
        .def("get_algorithm", &IInt8MinMaxCalibrator::getAlgorithm, IInt8MinMaxCalibratorDoc::get_algorithm)
        // For documentation purposes only
        .def("get_batch", docGetBatch<IInt8MinMaxCalibrator>, "names"_a, IInt8CalibratorDoc::get_batch)
        .def("read_calibration_cache", docReadCalibrationCache<IInt8MinMaxCalibrator>,
            IInt8CalibratorDoc::read_calibration_cache)
        .def("write_calibration_cache", docWriteCalibrationCache<IInt8MinMaxCalibrator>, "cache"_a,
            IInt8CalibratorDoc::write_calibration_cache);

} // Int8
} // namespace tensorrt
