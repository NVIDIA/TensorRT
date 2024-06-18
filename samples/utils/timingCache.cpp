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

#include "timingCache.h"
#include "NvInfer.h"
#include "fileLock.h"
#include "sampleUtils.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
using namespace nvinfer1;
namespace nvinfer1
{
namespace utils
{
std::vector<char> loadTimingCacheFile(ILogger& logger, std::string const& inFileName)
{
    try
    {
        std::unique_ptr<FileLock> fileLock{new FileLock(logger, inFileName)};
        std::ifstream iFile(inFileName, std::ios::in | std::ios::binary);
        if (!iFile)
        {
            std::stringstream ss;
            ss << "Could not read timing cache from: " << inFileName
               << ". A new timing cache will be generated and written.";
            logger.log(ILogger::Severity::kWARNING, ss.str().c_str());
            return std::vector<char>();
        }
        iFile.seekg(0, std::ifstream::end);
        size_t fsize = iFile.tellg();
        iFile.seekg(0, std::ifstream::beg);
        std::vector<char> content(fsize);
        iFile.read(content.data(), fsize);
        iFile.close();
        std::stringstream ss;
        ss << "Loaded " << fsize << " bytes of timing cache from " << inFileName;
        logger.log(ILogger::Severity::kINFO, ss.str().c_str());
        return content;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Exception detected: " << e.what() << std::endl;
    }
    return {};
}

std::unique_ptr<ITimingCache> buildTimingCacheFromFile(
    ILogger& logger, IBuilderConfig& config, std::string const& timingCacheFile, std::ostream& err)
{
    std::unique_ptr<nvinfer1::ITimingCache> timingCache{};
    auto timingCacheContents = loadTimingCacheFile(logger, timingCacheFile);
    timingCache.reset(config.createTimingCache(timingCacheContents.data(), timingCacheContents.size()));
    SMP_RETVAL_IF_FALSE(timingCache != nullptr, "TimingCache creation failed", nullptr, err);
    config.clearFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
    SMP_RETVAL_IF_FALSE(
        config.setTimingCache(*timingCache, true), "IBuilderConfig setTimingCache failed", nullptr, err);
    return timingCache;
}

void saveTimingCacheFile(ILogger& logger, std::string const& outFileName, IHostMemory const* blob)
{
    try
    {
        std::unique_ptr<FileLock> fileLock{new FileLock(logger, outFileName)};
        std::ofstream oFile(outFileName, std::ios::out | std::ios::binary);
        if (!oFile)
        {
            std::stringstream ss;
            ss << "Could not write timing cache to: " << outFileName;
            logger.log(ILogger::Severity::kWARNING, ss.str().c_str());
            return;
        }
        oFile.write(reinterpret_cast<char*>(blob->data()), blob->size());
        oFile.close();
        std::stringstream ss;
        ss << "Saved " << blob->size() << " bytes of timing cache to " << outFileName;
        logger.log(ILogger::Severity::kINFO, ss.str().c_str());
    }
    catch (std::exception const& e)
    {
        std::cerr << "Exception detected: " << e.what() << std::endl;
    }
}

void updateTimingCacheFile(nvinfer1::ILogger& logger, std::string const& fileName,
    nvinfer1::ITimingCache const* timingCache, nvinfer1::IBuilder& builder)
{
    try
    {
        // Prepare empty timingCache in case that there is no existing file to read
        std::unique_ptr<IBuilderConfig> config{builder.createBuilderConfig()};
        std::unique_ptr<ITimingCache> fileTimingCache{config->createTimingCache(static_cast<void const*>(nullptr), 0)};

        std::unique_ptr<FileLock> fileLock{new FileLock(logger, fileName)};
        std::ifstream iFile(fileName, std::ios::in | std::ios::binary);
        if (iFile)
        {
            iFile.seekg(0, std::ifstream::end);
            size_t fsize = iFile.tellg();
            iFile.seekg(0, std::ifstream::beg);
            std::vector<char> content(fsize);
            iFile.read(content.data(), fsize);
            iFile.close();
            std::stringstream ss;
            ss << "Loaded " << fsize << " bytes of timing cache from " << fileName;
            logger.log(ILogger::Severity::kINFO, ss.str().c_str());
            fileTimingCache.reset(config->createTimingCache(static_cast<void const*>(content.data()), content.size()));
            if (!fileTimingCache)
            {
                throw std::runtime_error("Failed to create timingCache from " + fileName + "!");
            }
        }
        fileTimingCache->combine(*timingCache, false);
        std::unique_ptr<IHostMemory> blob{fileTimingCache->serialize()};
        if (!blob)
        {
            throw std::runtime_error("Failed to serialize ITimingCache!");
        }
        std::ofstream oFile(fileName, std::ios::out | std::ios::binary);
        if (!oFile)
        {
            std::stringstream ss;
            ss << "Could not write timing cache to: " << fileName;
            logger.log(ILogger::Severity::kWARNING, ss.str().c_str());
            return;
        }
        oFile.write(reinterpret_cast<char*>(blob->data()), blob->size());
        oFile.close();
        std::stringstream ss;
        ss << "Saved " << blob->size() << " bytes of timing cache to " << fileName;
        logger.log(ILogger::Severity::kINFO, ss.str().c_str());
    }
    catch (std::exception const& e)
    {
        std::cerr << "Exception detected: " << e.what() << std::endl;
    }
}
} // namespace utils
} // namespace nvinfer1
