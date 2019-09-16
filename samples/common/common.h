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

using namespace nvinfer1;
using namespace plugin;

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
#define ENABLE_DLA_API 1
#endif

#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                         \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_RETURN_W_MSG(status, val, errMsg)                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(status))                                                                                                 \
        {                                                                                                              \
            std::cerr << errMsg << " Error in " << __FILE__ << ", function " << FN_NAME << "(), line " << __LINE__     \
                      << std::endl;                                                                                    \
            return val;                                                                                                \
        }                                                                                                              \
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

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
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

    virtual void reportLayerTime(const char* layerName, float ms)
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
            out << std::setw(maxLayerNameLength) << layerNameStr << " ";
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

// Locate path to file, given its filename or filepath suffix and possible dirs it might lie in
// Function will also walk back MAX_DEPTH dirs from CWD to check for such a file path
inline std::string locateFile(const std::string& filepathSuffix, const std::vector<std::string>& directories)
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
            filepath = dir + filepathSuffix;

        for (int i = 0; i < MAX_DEPTH && !found; i++)
        {
            std::ifstream checkFile(filepath);
            found = checkFile.is_open();
            if (found)
                break;
            filepath = "../" + filepath; // Try again in parent dir
        }

        if (found)
        {
            break;
        }

        filepath.clear();
    }

    if (filepath.empty())
    {
        std::string directoryList = std::accumulate(directories.begin() + 1, directories.end(), directories.front(),
            [](const std::string& a, const std::string& b) { return a + "\n\t" + b; });
        std::cout << "Could not find " << filepathSuffix << " in data directories:\n\t" << directoryList << std::endl;
        std::cout << "&&&& FAILED" << std::endl;
        exit(EXIT_FAILURE);
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

class HostMemory : public IHostMemory
{
public:
    HostMemory() = delete;
    void* data() const noexcept override
    {
        return mData;
    }
    std::size_t size() const noexcept override
    {
        return mSize;
    }
    DataType type() const noexcept override
    {
        return mType;
    }

protected:
    HostMemory(std::size_t size, DataType type)
        : mSize(size)
        , mType(type)
    {
    }
    void* mData;
    std::size_t mSize;
    DataType mType;
};

template <typename ElemType, DataType dataType>
class TypedHostMemory : public HostMemory
{
public:
    TypedHostMemory(std::size_t size)
        : HostMemory(size, dataType)
    {
        mData = new ElemType[size];
    };
    void destroy() noexcept override
    {
        delete[](ElemType*) mData;
        delete this;
    }
    ElemType* raw() noexcept
    {
        return static_cast<ElemType*>(data());
    }
};

using FloatMemory = TypedHostMemory<float, DataType::kFLOAT>;
using HalfMemory = TypedHostMemory<uint16_t, DataType::kHALF>;
using ByteMemory = TypedHostMemory<uint8_t, DataType::kINT8>;

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
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
inline std::shared_ptr<T> infer_object(T* obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, InferDeleter());
}

template <class Iter>
inline std::vector<size_t> argsort(Iter begin, Iter end, bool reverse = false)
{
    std::vector<size_t> inds(end - begin);
    std::iota(inds.begin(), inds.end(), 0);
    if (reverse)
    {
        std::sort(inds.begin(), inds.end(), [&begin](size_t i1, size_t i2) { return begin[i2] < begin[i1]; });
    }
    else
    {
        std::sort(inds.begin(), inds.end(), [&begin](size_t i1, size_t i2) { return begin[i1] < begin[i2]; });
    }
    return inds;
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

template <typename result_vector_t>
inline std::vector<std::string> classify(
    const std::vector<std::string>& refVector, const result_vector_t& output, const size_t topK)
{
    auto inds = samplesCommon::argsort(output.cbegin(), output.cend(), true);
    std::vector<std::string> result;
    for (size_t k = 0; k < topK; ++k)
    {
        result.push_back(refVector[inds[k]]);
    }
    return result;
}

// Returns top K indices, not values.
template <typename T>
inline std::vector<size_t> topK(const std::vector<T> inp, const size_t k)
{
    std::vector<size_t> result;
    std::vector<size_t> inds = samplesCommon::argsort(inp.cbegin(), inp.cend(), true);
    result.assign(inds.begin(), inds.begin() + k);
    return result;
}

template <typename T>
inline bool readASCIIFile(const std::string& fileName, const size_t size, std::vector<T>& out)
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
inline bool writeASCIIFile(const std::string& fileName, const std::vector<T>& in)
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

// Ensures that every tensor used by a network has a scale.
//
// All tensors in a network must have a range specified if a calibrator is not used.
// This function is just a utility to globally fill in missing scales for the entire network.
//
// If a tensor does not have a scale, it is assigned inScales or outScales as follows:
//
// * If the tensor is the input to a layer or output of a pooling node, its scale is assigned inScales.
// * Otherwise its scale is assigned outScales.
//
// The default parameter values are intended to demonstrate, for final layers in the network,
// cases where scaling factors are asymmetric.
inline void setAllTensorScales(INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                input->setDynamicRange(-inScales, inScales);
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
            ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == LayerType::kPOOLING)
                {
                    output->setDynamicRange(-inScales, inScales);
                }
                else
                {
                    output->setDynamicRange(-outScales, outScales);
                }
            }
        }
    }
}

inline void setDummyInt8Scales(const IBuilderConfig* c, INetworkDefinition* n)
{
    // Set dummy tensor scales if Int8 mode is requested.
    if (c->getFlag(BuilderFlag::kINT8))
    {
        gLogWarning
            << "Int8 calibrator not provided. Generating dummy per tensor scales. Int8 accuracy is not guaranteed."
            << std::endl;
        setAllTensorScales(n);
    }
}

inline void enableDLA(IBuilder* builder, IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
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
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
        }
        if (!builder->getInt8Mode() && !config->getFlag(BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            builder->setFp16Mode(true);
            config->setFlag(BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(useDLACore);
        config->setFlag(BuilderFlag::kSTRICT_TYPES);
    }
}

inline int parseDLA(int argc, char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        std::string arg(argv[i]);
        if (strncmp(argv[i], "--useDLACore=", 13) == 0)
            return std::stoi(argv[i] + 13);
    }
    return -1;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
    case DataType::kINT32:
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kINT8: return 1;
    }
    return 0;
}

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
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
inline void readPPMFile(const std::string& filename, samplesCommon::PPM<C, H, W>& ppm)
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
inline void writePPMFileWithBBox(const std::string& filename, PPM<C, H, W>& ppm, const BBox& bbox)
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
    GpuTimer(cudaStream_t stream)
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

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

inline int getC(const Dims& d)
{
    return d.nbDims >= 3 ? d.d[d.nbDims - 3] : 1;
}

inline int getH(const Dims& d)
{
    return d.nbDims >= 2 ? d.d[d.nbDims - 2] : 1;
}

inline int getW(const Dims& d)
{
    return d.nbDims >= 1 ? d.d[d.nbDims - 1] : 1;
}

inline void loadLibrary(const std::string& path)
{
#ifdef _MSC_VER
    void* handle = LoadLibrary(path.c_str());
#else
    void* handle = dlopen(path.c_str(), RTLD_LAZY);
#endif
    if (handle == nullptr)
    {
#ifdef _MSC_VER
        gLogError << "Could not load plugin library: " << path << std::endl;
#else
        gLogError << "Could not load plugin library: " << path << ", due to: " << dlerror() << std::endl;
#endif
    }
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
