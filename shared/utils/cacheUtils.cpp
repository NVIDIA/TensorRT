/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cacheUtils.h"
#include "NvInfer.h"
#include "fileLock.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace nvinfer1::utils
{
std::vector<char> loadCacheFile(ILogger& logger, std::string const& inFileName)
{
    try
    {
        FileLock fileLock{logger, inFileName};
        std::ifstream iFile(inFileName, std::ios::in | std::ios::binary);
        if (!iFile)
        {
            std::stringstream ss;
            ss << "Could not read cache from: " << inFileName << ". A new cache will be generated and written.";
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
        ss << "Loaded " << fsize << " bytes of cache from file: " << inFileName;
        logger.log(ILogger::Severity::kINFO, ss.str().c_str());
        return content;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Exception while loading cache file " << inFileName << ": " << e.what() << std::endl;
    }
    return {};
}

std::unique_ptr<ITimingCache> buildTimingCacheFromFile(
    ILogger& logger, IBuilderConfig& config, std::string const& timingCacheFile)
{
    std::unique_ptr<nvinfer1::ITimingCache> timingCache{};
    std::vector<char> timingCacheContents = loadCacheFile(logger, timingCacheFile);

    timingCache.reset(config.createTimingCache(timingCacheContents.data(), timingCacheContents.size()));
    if (timingCache == nullptr)
    {
        logger.log(ILogger::Severity::kERROR, ("Failed to create ITimingCache from file " + timingCacheFile).c_str());
        return nullptr;
    }

    config.clearFlag(BuilderFlag::kDISABLE_TIMING_CACHE);
    if (!config.setTimingCache(*timingCache, true))
    {
        logger.log(ILogger::Severity::kERROR,
            ("IBuilderConfig#setTimingCache failed with timing cache from file " + timingCacheFile).c_str());
        return nullptr;
    }
    return timingCache;
}

void saveCacheFile(ILogger& logger, std::string const& outFileName, IHostMemory const* blob)
{
    try
    {
        FileLock fileLock{logger, outFileName};
        std::ofstream oFile(outFileName, std::ios::out | std::ios::binary);
        if (!oFile)
        {
            std::stringstream ss;
            ss << "Could not write cache to file: " << outFileName;
            logger.log(ILogger::Severity::kWARNING, ss.str().c_str());
            return;
        }
        oFile.write(reinterpret_cast<char const*>(blob->data()), blob->size());
        oFile.close();
        std::stringstream ss;
        ss << "Saved " << blob->size() << " bytes of cache to file: " << outFileName;
        logger.log(ILogger::Severity::kINFO, ss.str().c_str());
    }
    catch (std::exception const& e)
    {
        std::cerr << "Exception while saving cache file " << outFileName << ": " << e.what() << std::endl;
    }
}

void updateTimingCacheFile(nvinfer1::ILogger& logger, std::string const& fileName,
    nvinfer1::ITimingCache const* timingCache, nvinfer1::IBuilder& builder)
{
    try
    {
        std::unique_ptr<IBuilderConfig> config{builder.createBuilderConfig()};
        std::vector<char> timingCacheContents = loadCacheFile(logger, fileName);
        std::unique_ptr<ITimingCache> fileTimingCache{
            config->createTimingCache(timingCacheContents.data(), timingCacheContents.size())};

        fileTimingCache->combine(*timingCache, false);
        std::unique_ptr<IHostMemory> blob{fileTimingCache->serialize()};
        if (!blob)
        {
            throw std::runtime_error("Failed to serialize combined ITimingCache!");
        }

        FileLock fileLock{logger, fileName};
        std::ofstream oFile(fileName, std::ios::out | std::ios::binary);
        if (!oFile)
        {
            std::stringstream ss;
            ss << "Could not write timing cache to: " << fileName;
            logger.log(ILogger::Severity::kWARNING, ss.str().c_str());
            return;
        }

        oFile.write(reinterpret_cast<char const*>(blob->data()), blob->size());
        oFile.close();
        std::stringstream ss;
        ss << "Saved " << blob->size() << " bytes of timing cache to " << fileName;
        logger.log(ILogger::Severity::kINFO, ss.str().c_str());
    }
    catch (std::exception const& e)
    {
        std::cerr << "Exception while updating timing cache file " << fileName << ": " << e.what() << std::endl;
    }
}
} // namespace nvinfer1::utils
