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
#ifndef TRT_PARSER_UTILS_H
#define TRT_PARSER_UTILS_H

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>

#ifndef _MSC_VER
#include <unistd.h>
#else
#define NOMINMAX
#include <windows.h>
#endif

#include "NvInfer.h"

namespace parserutils
{
#define RETURN_AND_LOG_ERROR_IMPL(ret, message, parserName)                                             \
    do                                                                                                  \
    {                                                                                                   \
        std::string errorMsg = parserName + std::string{message};                                       \
        if (getLogger()) getLogger()->log(nvinfer1::ILogger::Severity::kERROR, errorMsg.c_str());       \
        else std::cerr << "WARNING: Logger not found, logging to stderr.\n" << errorMsg << std::endl;   \
        return (ret);                                                                                   \
    } while (0)

// Helper function to compute unpadded volume of a Dims (1 if 0 dimensional)
inline int64_t volume(const nvinfer1::Dims& d)
{
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}

// Show some debugging output about how much memory is free
inline void printMem(const char* where)
{
#if !defined _MSC_VER && !defined __QNX__
    const unsigned mb = 1024 * 1024;
    auto pages = static_cast<uint64_t>(sysconf(_SC_PHYS_PAGES));
    auto avPages = static_cast<uint64_t>(sysconf(_SC_AVPHYS_PAGES));
    auto pageSize = static_cast<uint64_t>(sysconf(_SC_PAGE_SIZE));
    std::cout << "   (memory) " << where << " : Free(MB) = " << avPages * pageSize / mb << " total(MB)=" << pages * pageSize / mb << std::endl;
#elif !defined __QNX__
    const unsigned mb = 1024 * 1024;
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    std::cout << "   (memory) " << where << " : Free(MB) = " << statex.ullAvailPhys / mb << " total(MB)=" << statex.ullTotalPhys / mb << std::endl;
#endif
}

// Compute size of datatypes
inline unsigned int elementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}

inline std::ostream& operator<<(std::ostream& o, const nvinfer1::Dims& dims)
{
    o << "[";
    for (int i = 0; i < dims.nbDims; i++)
        o << (i ? "," : "") << dims.d[i];
    o << "]";
    return o;
}

inline std::ostream& operator<<(std::ostream& o, nvinfer1::DataType dt)
{
    switch (dt)
    {
    case nvinfer1::DataType::kINT32: o << "Int32"; break;
    case nvinfer1::DataType::kFLOAT: o << "Float"; break;
    case nvinfer1::DataType::kHALF: o << "Half"; break;
    case nvinfer1::DataType::kINT8: o << "Int8"; break;
    case nvinfer1::DataType::kBOOL: o << "Bool"; break;
    }
    return o;
}

inline nvinfer1::Dims3 getCHW(const nvinfer1::Dims& d)
{
    assert(d.nbDims >= 3);
    return nvinfer1::Dims3(d.d[d.nbDims - 3], d.d[d.nbDims - 2], d.d[d.nbDims - 1]);
}

inline int32_t getC(const nvinfer1::Dims& d)
{
    return getCHW(d).d[0];
}

inline nvinfer1::Dims toDims(int32_t w, int32_t h) noexcept
{
    return nvinfer1::Dims{2, {w, h}};
}

inline int combineIndexDimensions(int batchSize, const nvinfer1::Dims& d)
{
    int x = batchSize;
    for (int i = 0; i < d.nbDims - 3; i++)
        x *= d.d[i];
    return x;
}

template <typename A, typename B>
inline A divUp(A m, B n)
{
    return (m + n - 1) / n;
}

} // namespace parserhelper

#endif // PARSER_HELPER_H
