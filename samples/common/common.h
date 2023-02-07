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

#ifndef TENSORRT_COMMON_H
#define TENSORRT_COMMON_H

// For loadLibrary
#ifdef _MSC_VER
// Needed so that the max/min definitions in windows.h do not conflict with std::max/min.
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <dlfcn.h>
#endif

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <ratio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#else
#include <stdio.h>  // fileno
#include <unistd.h> // lockf
#endif

#include "safeCommon.h"

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if defined(__aarch64__) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            sample::gLogError << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line " << __LINE__     \
                      << std::endl;                                                                                    \
            return val;                                                                                                \
        }                                                                                                              \
    } while (0)

#undef ASSERT
#define ASSERT(condition)                                                   \
    do                                                                      \
    {                                                                       \
        if (!(condition))                                                   \
        {                                                                   \
            sample::gLogError << "Assertion failure: " << #condition << std::endl;  \
            abort();                                                        \
        }                                                                   \
    } while (0)


#define CHECK_RETURN(status, val) CHECK_RETURN_W_MSG(status, val, "")

#define OBJ_GUARD(A) std::unique_ptr<A, void (*)(A * t)>

template <typename T, typename T_>
OBJ_GUARD(T)
makeObjGuard(T_* t)
{
    CHECK(!(std::is_base_of<T, T_>::value || std::is_same<T, T_>::value));
    auto deleter = [](T* t) { t->destroy(); };
    return std::unique_ptr<T, decltype(deleter)>{static_cast<T*>(t), deleter};
}

constexpr long double operator"" _GiB(long double val)
{
    return val * (1 << 30);
}
constexpr long double operator"" _MiB(long double val)
{
    return val * (1 << 20);
}
constexpr long double operator"" _KiB(long double val)
{
    return val * (1 << 10);
}

struct SimpleProfiler : public nvinfer1::IProfiler
{
    struct Record
    {
        float time{0};
        int count{0};
    };

    virtual void reportLayerTime(const char* layerName, float ms) noexcept
    {
        mProfile[layerName].count++;
        mProfile[layerName].time += ms;
        if (std::find(mLayerNames.begin(), mLayerNames.end(), layerName) == mLayerNames.end())
        {
            mLayerNames.push_back(layerName);
        }
    }

    SimpleProfiler(const char* name, const std::vector<SimpleProfiler>& srcProfilers = std::vector<SimpleProfiler>())
        : mName(name)
    {
        for (const auto& srcProfiler : srcProfilers)
        {
            for (const auto& rec : srcProfiler.mProfile)
            {
                auto it = mProfile.find(rec.first);
                if (it == mProfile.end())
                {
                    mProfile.insert(rec);
                }
                else
                {
                    it->second.time += rec.second.time;
                    it->second.count += rec.second.count;
                }
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const SimpleProfiler& value)
    {
        out << "========== " << value.mName << " profile ==========" << std::endl;
        float totalTime = 0;
        std::string layerNameStr = "TensorRT layer name";
        int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
        for (const auto& elem : value.mProfile)
        {
            totalTime += elem.second.time;
            maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
        }

        auto old_settings = out.flags();
        auto old_precision = out.precision();
        // Output header
        {
            out << std::setfill(' ') << std::setw(maxLayerNameLength) << layerNameStr << " ";
            out << std::setw(12) << "Runtime, "
                << "%"
                << " ";
            out << std::setw(12) << "Invocations"
                << " ";
            out << std::setw(12) << "Runtime, ms" << std::endl;
        }
        for (size_t i = 0; i < value.mLayerNames.size(); i++)
        {
            const std::string layerName = value.mLayerNames[i];
            auto elem = value.mProfile.at(layerName);
            out << std::setw(maxLayerNameLength) << layerName << " ";
            out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.time * 100.0F / totalTime) << "%"
                << " ";
            out << std::setw(12) << elem.count << " ";
            out << std::setw(12) << std::fixed << std::setprecision(2) << elem.time << std::endl;
        }
        out.flags(old_settings);
        out.precision(old_precision);
        out << "========== " << value.mName << " total runtime = " << totalTime << " ms ==========" << std::endl;

        return out;
    }

private:
    std::string mName;
    std::vector<std::string> mLayerNames;
    std::map<std::string, Record> mProfile;
};

//! Locate path to file, given its filename or filepath suffix and possible dirs it might lie in.
//! Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path.
inline std::string locateFile(
    const std::string& filepathSuffix, const std::vector<std::string>& directories, bool reportError = true)
{
    const int MAX_DEPTH{10};
    bool found{false};
    std::string filepath;

    for (auto& dir : directories)
    {
        if (!dir.empty() && dir.back() != '/')
        {
#ifdef _MSC_VER
            filepath = dir + "\\" + filepathSuffix;
#else
            filepath = dir + "/" + filepathSuffix;
#endif
        }
        else
        {
            filepath = dir + filepathSuffix;
        }

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            const std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found)
            {
                break;
            }

            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    // Could not find the file
    if (filepath.empty())
    {
        const std::string dirList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
            [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << dirList << std::endl;

        if (reportError)
        {
            std::cout << "&&&& FAILED" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return filepath;
}

inline void readPGMFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

namespace samplesCommon
{

// Swaps endianness of an integral type.
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T swapEndianness(const T& value)
{
    uint8_t bytes[sizeof(T)];
    for (int i = 0; i < static_cast<int>(sizeof(T)); ++i)
    {
        bytes[sizeof(T) - 1 - i] = *(reinterpret_cast<const uint8_t*>(&value) + i);
    }
    return *reinterpret_cast<T*>(bytes);
}

class HostMemory
{
public:
    HostMemory() = delete;
    virtual void* data() const noexcept
    {
        return mData;
    }
    virtual std::size_t size() const noexcept
    {
        return mSize;
    }
    virtual nvinfer1::DataType type() const noexcept
    {
        return mType;
    }
    virtual ~HostMemory() {}

protected:
    HostMemory(std::size_t size, nvinfer1::DataType type)
        : mData{nullptr}
        , mSize(size)
        , mType(type)
    {
    }
    void* mData;
    std::size_t mSize;
    nvinfer1::DataType mType;
};

template <typename ElemType, nvinfer1::DataType dataType>
class TypedHostMemory : public HostMemory
{
public:
    explicit TypedHostMemory(std::size_t size)
        : HostMemory(size, dataType)
    {
        mData = new ElemType[size];
    };
    ~TypedHostMemory() noexcept
    {
        delete[](ElemType*) mData;
    }
    ElemType* raw() noexcept
    {
        return static_cast<ElemType*>(data());
    }
};

using FloatMemory = TypedHostMemory<float, nvinfer1::DataType::kFLOAT>;
using HalfMemory = TypedHostMemory<uint16_t, nvinfer1::DataType::kHALF>;
using ByteMemory = TypedHostMemory<uint8_t, nvinfer1::DataType::kINT8>;

inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

inline bool isDebug()
{
    return (std::getenv("TENSORRT_DEBUG") ? true : false);
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

static auto StreamDeleter = [](cudaStream_t* pStream)
    {
        if (pStream)
        {
            cudaStreamDestroy(*pStream);
            delete pStream;
        }
    };

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        pStream.reset(nullptr);
    }

    return pStream;
}

//! Return vector of indices that puts magnitudes of sequence in descending order.
template <class Iter>
std::vector<size_t> argMagnitudeSort(Iter begin, Iter end)
{
    std::vector<size_t> indices(end - begin);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&begin](size_t i, size_t j) { return std::abs(begin[j]) < std::abs(begin[i]); });
    return indices;
}

inline bool readReferenceFile(const std::string& fileName, std::vector<std::string>& refVector)
{
    std::ifstream infile(fileName);
    if (!infile.is_open())
    {
        std::cout << "ERROR: readReferenceFile: Attempting to read from a file that is not open." << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line))
    {
        if (line.empty())
            continue;
        refVector.push_back(line);
    }
    infile.close();
    return true;
}

template <typename T>
std::vector<std::string> classify(
    const std::vector<std::string>& refVector, const std::vector<T>& output, const size_t topK)
{
    const auto inds = samplesCommon::argMagnitudeSort(output.cbegin(), output.cend());
    std::vector<std::string> result;
    result.reserve(topK);
    for (size_t k = 0; k < topK; ++k)
    {
        result.push_back(refVector[inds[k]]);
    }
    return result;
}

// Returns indices of highest K magnitudes in v.
template <typename T>
std::vector<size_t> topKMagnitudes(const std::vector<T>& v, const size_t k)
{
    std::vector<size_t> indices = samplesCommon::argMagnitudeSort(v.cbegin(), v.cend());
    indices.resize(k);
    return indices;
}

template <typename T>
bool readASCIIFile(const std::string& fileName, const size_t size, std::vector<T>& out)
{
    std::ifstream infile(fileName);
    if (!infile.is_open())
    {
        std::cout << "ERROR readASCIIFile: Attempting to read from a file that is not open." << std::endl;
        return false;
    }
    out.clear();
    out.reserve(size);
    out.assign(std::istream_iterator<T>(infile), std::istream_iterator<T>());
    infile.close();
    return true;
}

template <typename T>
bool writeASCIIFile(const std::string& fileName, const std::vector<T>& in)
{
    std::ofstream outfile(fileName);
    if (!outfile.is_open())
    {
        std::cout << "ERROR: writeASCIIFile: Attempting to write to a file that is not open." << std::endl;
        return false;
    }
    for (auto fn : in)
    {
        outfile << fn << "\n";
    }
    outfile.close();
    return true;
}

inline void print_version()
{
    std::cout << "  TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH
              << "." << NV_TENSORRT_BUILD << std::endl;
}

inline std::string getFileType(const std::string& filepath)
{
    return filepath.substr(filepath.find_last_of(".") + 1);
}

inline std::string toLower(const std::string& inp)
{
    std::string out = inp;
    std::transform(out.begin(), out.end(), out.begin(), ::tolower);
    return out;
}

inline float getMaxValue(const float* buffer, int64_t size)
{
    assert(buffer != nullptr);
    assert(size > 0);
    return *std::max_element(buffer, buffer + size);
}

inline int32_t calculateSoftmax(float* const prob, int32_t const numDigits)
{
    ASSERT(prob != nullptr);
    ASSERT(numDigits == 10);
    float sum{0.0F};
    std::transform(prob, prob + numDigits, prob, [&sum](float v) -> float {
        sum += exp(v);
        return exp(v);
    });

    ASSERT(sum != 0.0F);
    std::transform(prob, prob + numDigits, prob, [sum](float v) -> float { return v / sum; });
    int32_t idx = std::max_element(prob, prob + numDigits) - prob;
    return idx;
}

// Ensures that every tensor used by a network has a dynamic range set.
//
// All tensors in a network must have a dynamic range specified if a calibrator is not used.
// This function is just a utility to globally fill in missing scales and zero-points for the entire network.
//
// If a tensor does not have a dyanamic range set, it is assigned inRange or outRange as follows:
//
// * If the tensor is the input to a layer or output of a pooling node, its dynamic range is derived from inRange.
// * Otherwise its dynamic range is derived from outRange.
//
// The default parameter values are intended to demonstrate, for final layers in the network,
// cases where dynamic ranges are asymmetric.
//
// The default parameter values choosen arbitrarily. Range values should be choosen such that
// we avoid underflow or overflow. Also range value should be non zero to avoid uniform zero scale tensor.
inline void setAllDynamicRanges(nvinfer1::INetworkDefinition* network, float inRange = 2.0f, float outRange = 4.0f)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                ASSERT(input->setDynamicRange(-inRange, inRange));
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    ASSERT(output->setDynamicRange(-inRange, inRange));
                }
                else
                {
                    ASSERT(output->setDynamicRange(-outRange, outRange));
                }
            }
        }
    }
}

inline void setDummyInt8DynamicRanges(const nvinfer1::IBuilderConfig* c, nvinfer1::INetworkDefinition* n)
{
    // Set dummy per-tensor dynamic range if Int8 mode is requested.
    if (c->getFlag(nvinfer1::BuilderFlag::kINT8))
    {
        sample::gLogWarning << "Int8 calibrator not provided. Generating dummy per-tensor dynamic range. Int8 accuracy "
                               "is not guaranteed."
                            << std::endl;
        setAllDynamicRanges(n);
    }
}

inline void enableDLA(
    nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
    }
}

inline int32_t parseDLA(int32_t argc, char** argv)
{
    for (int32_t i = 1; i < argc; i++)
    {
        if (strncmp(argv[i], "--useDLACore=", 13) == 0)
        {
            return std::stoi(argv[i] + 13);
        }
    }
    return -1;
}

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

inline int64_t volume(nvinfer1::Dims const& dims, int32_t start, int32_t stop)
{
    ASSERT(start >= 0);
    ASSERT(start <= stop);
    ASSERT(stop <= dims.nbDims);
    ASSERT(std::all_of(dims.d + start, dims.d + stop, [](int32_t x) { return x >= 0; }));
    return std::accumulate(dims.d + start, dims.d + stop, int64_t{1}, std::multiplies<int64_t>{});
}

template <int C, int H, int W>
struct PPM
{
    std::string magic, fileName;
    int h, w, max;
    uint8_t buffer[C * H * W];
};

// New vPPM(variable sized PPM) class with variable dimensions.
struct vPPM
{
    std::string magic, fileName;
    int h, w, max;
    std::vector<uint8_t> buffer;
};

struct BBox
{
    float x1, y1, x2, y2;
};

template <int C, int H, int W>
void readPPMFile(const std::string& filename, samplesCommon::PPM<C, H, W>& ppm)
{
    ppm.fileName = filename;
    std::ifstream infile(filename, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

inline void readPPMFile(const std::string& filename, vPPM& ppm, std::vector<std::string>& input_dir)
{
    ppm.fileName = filename;
    std::ifstream infile(locateFile(filename, input_dir), std::ifstream::binary);
    infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    infile.seekg(1, infile.cur);

    for (int i = 0; i < ppm.w * ppm.h * 3; ++i)
    {
        ppm.buffer.push_back(0);
    }

    infile.read(reinterpret_cast<char*>(&ppm.buffer[0]), ppm.w * ppm.h * 3);
}

template <int C, int H, int W>
void writePPMFileWithBBox(const std::string& filename, PPM<C, H, W>& ppm, const BBox& bbox)
{
    std::ofstream outfile("./" + filename, std::ofstream::binary);
    assert(!outfile.fail());
    outfile << "P6"
            << "\n"
            << ppm.w << " " << ppm.h << "\n"
            << ppm.max << "\n";

    auto round = [](float x) -> int { return int(std::floor(x + 0.5f)); };
    const int x1 = std::min(std::max(0, round(int(bbox.x1))), W - 1);
    const int x2 = std::min(std::max(0, round(int(bbox.x2))), W - 1);
    const int y1 = std::min(std::max(0, round(int(bbox.y1))), H - 1);
    const int y2 = std::min(std::max(0, round(int(bbox.y2))), H - 1);

    for (int x = x1; x <= x2; ++x)
    {
        // bbox top border
        ppm.buffer[(y1 * ppm.w + x) * 3] = 255;
        ppm.buffer[(y1 * ppm.w + x) * 3 + 1] = 0;
        ppm.buffer[(y1 * ppm.w + x) * 3 + 2] = 0;
        // bbox bottom border
        ppm.buffer[(y2 * ppm.w + x) * 3] = 255;
        ppm.buffer[(y2 * ppm.w + x) * 3 + 1] = 0;
        ppm.buffer[(y2 * ppm.w + x) * 3 + 2] = 0;
    }

    for (int y = y1; y <= y2; ++y)
    {
        // bbox left border
        ppm.buffer[(y * ppm.w + x1) * 3] = 255;
        ppm.buffer[(y * ppm.w + x1) * 3 + 1] = 0;
        ppm.buffer[(y * ppm.w + x1) * 3 + 2] = 0;
        // bbox right border
        ppm.buffer[(y * ppm.w + x2) * 3] = 255;
        ppm.buffer[(y * ppm.w + x2) * 3 + 1] = 0;
        ppm.buffer[(y * ppm.w + x2) * 3 + 2] = 0;
    }

    outfile.write(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

inline void writePPMFileWithBBox(const std::string& filename, vPPM ppm, std::vector<BBox>& dets)
{
    std::ofstream outfile("./" + filename, std::ofstream::binary);
    assert(!outfile.fail());
    outfile << "P6"
            << "\n"
            << ppm.w << " " << ppm.h << "\n"
            << ppm.max << "\n";
    auto round = [](float x) -> int { return int(std::floor(x + 0.5f)); };

    for (auto bbox : dets)
    {
        for (int x = int(bbox.x1); x < int(bbox.x2); ++x)
        {
            // bbox top border
            ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3] = 255;
            ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 1] = 0;
            ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 2] = 0;
            // bbox bottom border
            ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3] = 255;
            ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 1] = 0;
            ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 2] = 0;
        }

        for (int y = int(bbox.y1); y < int(bbox.y2); ++y)
        {
            // bbox left border
            ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3] = 255;
            ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 1] = 0;
            ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 2] = 0;
            // bbox right border
            ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3] = 255;
            ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 1] = 0;
            ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 2] = 0;
        }
    }

    outfile.write(reinterpret_cast<char*>(&ppm.buffer[0]), ppm.w * ppm.h * 3);
}

class TimerBase
{
public:
    virtual void start() {}
    virtual void stop() {}
    float microseconds() const noexcept
    {
        return mMs * 1000.f;
    }
    float milliseconds() const noexcept
    {
        return mMs;
    }
    float seconds() const noexcept
    {
        return mMs / 1000.f;
    }
    void reset() noexcept
    {
        mMs = 0.f;
    }

protected:
    float mMs{0.0f};
};

class GpuTimer : public TimerBase
{
public:
    explicit GpuTimer(cudaStream_t stream)
        : mStream(stream)
    {
        CHECK(cudaEventCreate(&mStart));
        CHECK(cudaEventCreate(&mStop));
    }
    ~GpuTimer()
    {
        CHECK(cudaEventDestroy(mStart));
        CHECK(cudaEventDestroy(mStop));
    }
    void start()
    {
        CHECK(cudaEventRecord(mStart, mStream));
    }
    void stop()
    {
        CHECK(cudaEventRecord(mStop, mStream));
        float ms{0.0f};
        CHECK(cudaEventSynchronize(mStop));
        CHECK(cudaEventElapsedTime(&ms, mStart, mStop));
        mMs += ms;
    }

private:
    cudaEvent_t mStart, mStop;
    cudaStream_t mStream;
}; // class GpuTimer

template <typename Clock>
class CpuTimer : public TimerBase
{
public:
    using clock_type = Clock;

    void start()
    {
        mStart = Clock::now();
    }
    void stop()
    {
        mStop = Clock::now();
        mMs += std::chrono::duration<float, std::milli>{mStop - mStart}.count();
    }

private:
    std::chrono::time_point<Clock> mStart, mStop;
}; // class CpuTimer

using PreciseCpuTimer = CpuTimer<std::chrono::high_resolution_clock>;

inline std::vector<std::string> splitString(std::string str, char delimiter = ',')
{
    std::vector<std::string> splitVect;
    std::stringstream ss(str);
    std::string substr;

    while (ss.good())
    {
        getline(ss, substr, delimiter);
        splitVect.emplace_back(std::move(substr));
    }
    return splitVect;
}

inline int getC(nvinfer1::Dims const& d)
{
    return d.nbDims >= 3 ? d.d[d.nbDims - 3] : 1;
}

inline int getH(const nvinfer1::Dims& d)
{
    return d.nbDims >= 2 ? d.d[d.nbDims - 2] : 1;
}

inline int getW(const nvinfer1::Dims& d)
{
    return d.nbDims >= 1 ? d.d[d.nbDims - 1] : 1;
}

inline void loadLibrary(const std::string& path)
{
#ifdef _MSC_VER
    void* handle = LoadLibrary(path.c_str());
#else
    int32_t flags{RTLD_LAZY};
#if ENABLE_ASAN
    // https://github.com/google/sanitizers/issues/89
    // asan doesn't handle module unloading correctly and there are no plans on doing
    // so. In order to get proper stack traces, don't delete the shared library on
    // close so that asan can resolve the symbols correctly.
    flags |= RTLD_NODELETE;
#endif // ENABLE_ASAN

    void* handle = dlopen(path.c_str(), flags);
#endif
    if (handle == nullptr)
    {
#ifdef _MSC_VER
        sample::gLogError << "Could not load plugin library: " << path << std::endl;
#else
        sample::gLogError << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
#endif
    }
}

inline int32_t getSMVersion()
{
    int32_t deviceIndex = 0;
    CHECK(cudaGetDevice(&deviceIndex));

    int32_t major, minor;
    CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceIndex));
    CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceIndex));

    return ((major << 8) | minor);
}

inline bool isSMSafe()
{
    const int32_t smVersion = getSMVersion();
    return smVersion == 0x0700 || smVersion == 0x0702 || smVersion == 0x0705 || smVersion == 0x0800
        || smVersion == 0x0806 || smVersion == 0x0807;
}

inline int32_t getMaxPersistentCacheSize()
{
    int32_t deviceIndex{};
    CHECK(cudaGetDevice(&deviceIndex));

    int32_t maxPersistentL2CacheSize;
#if CUDART_VERSION >= 11030
    CHECK(cudaDeviceGetAttribute(&maxPersistentL2CacheSize, cudaDevAttrMaxPersistingL2CacheSize, deviceIndex));
#else
    maxPersistentL2CacheSize = 0;
#endif

    return maxPersistentL2CacheSize;
}

inline bool isDataTypeSupported(nvinfer1::DataType dataType)
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8())
        || (dataType == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16()))
    {
        return false;
    }

    return true;
}

class FileLock
{
public:
    FileLock(std::string const& fileName)
        : fileName(fileName)
    {
        std::string lockFileName = fileName + ".lock";
#ifdef _MSC_VER
        sample::gLogVerbose << "Trying to set exclusive file lock " << lockFileName << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        // MS docs said this is a blocking IO if "FILE_FLAG_OVERLAPPED" is not provided
        lock = CreateFileA(lockFileName.c_str(), GENERIC_WRITE, 0, NULL, OPEN_ALWAYS, 0, NULL);
        if (lock != INVALID_HANDLE_VALUE)
        {
            float const time
                = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTime).count();
            sample::gLogVerbose << "File locked in " << time << " seconds." << std::endl;
        }
        else
        {
            throw std::runtime_error("Failed to lock " + lockFileName + "!");
        }
#elif defined(__QNX__)
        // We once enabled the file lock on QNX, lockf(F_TLOCK) return -1 and the reported error is
        // The error generated was 89
        // That means : Function not implemented
#else
        fp = fopen(lockFileName.c_str(), "wb+");
        if (!fp)
        {
            throw std::runtime_error("Cannot open " + lockFileName + "!");
        }
        fd = fileno(fp);
        sample::gLogVerbose << "Trying to set exclusive file lock " << lockFileName << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        auto ret = lockf(fd, F_LOCK, 0);
        if (ret != 0)
        {
            fd = -1;
            fclose(fp);
            throw std::runtime_error("Failed to lock " + lockFileName + "!");
        }
        float const time = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTime).count();
        sample::gLogVerbose << "File locked in " << time << " seconds." << std::endl;
#endif
    }

    ~FileLock()
    {
        std::string lockFileName = fileName + ".lock";
#ifdef _MSC_VER
        if (lock != INVALID_HANDLE_VALUE)
        {
            sample::gLogVerbose << "Trying to remove exclusive file lock " << lockFileName << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
            CloseHandle(lock);
            float const time
                = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTime).count();
            sample::gLogVerbose << "File unlocked in " << time << " seconds." << std::endl;
        }
#elif defined(__QNX__)
        // We once enabled the file lock on QNX, lockf(F_TLOCK) return -1 and the reported error is
        // The error generated was 89
        // That means : Function not implemented
#else
        if (fd != -1)
        {
            sample::gLogVerbose << "Trying to remove exclusive file lock " << lockFileName << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
            auto ret = lockf(fd, F_ULOCK, 0);
            if (ret != 0)
            {
                sample::gLogVerbose << "Failed to unlock " << lockFileName << "!" << std::endl;
            }
            else
            {
                fd = -1;
                fclose(fp);
                float const time
                    = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - startTime).count();
                sample::gLogVerbose << "File unlocked in " << time << " seconds." << std::endl;
            }
        }
#endif
    }

private:
    FileLock() = delete;                           // no default ctor
    FileLock(FileLock const&) = delete;            // no copy ctor
    FileLock& operator=(FileLock const&) = delete; // no copy assignment

    const std::string fileName; // the file being protected
#ifdef _MSC_VER
    HANDLE lock;
#else
    FILE* fp;
    int32_t fd;
#endif
};

inline std::vector<char> loadTimingCacheFile(const std::string inFileName)
{
    std::unique_ptr<samplesCommon::FileLock> fileLock{new samplesCommon::FileLock(inFileName)};
    std::ifstream iFile(inFileName, std::ios::in | std::ios::binary);
    if (!iFile)
    {
        sample::gLogWarning << "Could not read timing cache from: " << inFileName
                            << ". A new timing cache will be generated and written." << std::endl;
        return std::vector<char>();
    }
    iFile.seekg(0, std::ifstream::end);
    size_t fsize = iFile.tellg();
    iFile.seekg(0, std::ifstream::beg);
    std::vector<char> content(fsize);
    iFile.read(content.data(), fsize);
    iFile.close();
    sample::gLogInfo << "Loaded " << fsize << " bytes of timing cache from " << inFileName << std::endl;
    return content;
}

inline void saveTimingCacheFile(const std::string outFileName, const nvinfer1::IHostMemory* blob)
{
    std::unique_ptr<samplesCommon::FileLock> fileLock{new samplesCommon::FileLock(outFileName)};
    std::ofstream oFile(outFileName, std::ios::out | std::ios::binary);
    if (!oFile)
    {
        sample::gLogWarning << "Could not write timing cache to: " << outFileName << std::endl;
        return;
    }
    oFile.write((char*) blob->data(), blob->size());
    oFile.close();
    sample::gLogInfo << "Saved " << blob->size() << " bytes of timing cache to " << outFileName << std::endl;
}

inline void updateTimingCacheFile(std::string const fileName, nvinfer1::ITimingCache const* timingCache)
{
    // Prepare empty timingCache in case that there is no existing file to read
    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger())};
    std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    std::unique_ptr<nvinfer1::ITimingCache> fileTimingCache{config->createTimingCache(static_cast<const void*>(nullptr), 0)};

    std::unique_ptr<samplesCommon::FileLock> fileLock{new samplesCommon::FileLock(fileName)};
    std::ifstream iFile(fileName, std::ios::in | std::ios::binary);
    if (iFile)
    {
        iFile.seekg(0, std::ifstream::end);
        size_t fsize = iFile.tellg();
        iFile.seekg(0, std::ifstream::beg);
        std::vector<char> content(fsize);
        iFile.read(content.data(), fsize);
        iFile.close();
        sample::gLogInfo << "Loaded " << fsize << " bytes of timing cache from " << fileName << std::endl;
        fileTimingCache.reset(config->createTimingCache(static_cast<const void*>(content.data()), content.size()));
        if (!fileTimingCache)
        {
            throw std::runtime_error("Failed to create timingCache from " + fileName + "!");
        }
    }
    fileTimingCache->combine(*timingCache, false);
    std::unique_ptr<nvinfer1::IHostMemory> blob{fileTimingCache->serialize()};
    if (!blob)
    {
        throw std::runtime_error("Failed to serialize ITimingCache!");
    }
    std::ofstream oFile(fileName, std::ios::out | std::ios::binary);
    if (!oFile)
    {
        sample::gLogWarning << "Could not write timing cache to: " << fileName << std::endl;
        return;
    }
    oFile.write((char*) blob->data(), blob->size());
    oFile.close();
    sample::gLogInfo << "Saved " << blob->size() << " bytes of timing cache to " << fileName << std::endl;
}

} // namespace samplesCommon

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    os << "(";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? ", " : "") << dims.d[i];
    }
    return os << ")";
}

#endif // TENSORRT_COMMON_H
