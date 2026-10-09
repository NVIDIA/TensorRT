/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "sampleUtils.h"
#include "logger.h"

namespace sample
{

//! @brief Sanitizes the remote auto tuning config string by removing sensitive credentials
//!
//! This function removes usernames and passwords from URL-style configuration strings
//! to prevent sensitive authentication information from appearing in logs or debug output.
//! The credentials section (username:password) is replaced with "***" for security.
//!
//! Config format: protocol://username[:password]@hostname[:port]?param1=value1&param2=value2
//! Supported protocols: ssh, http, https, etc.
//!
//! Examples:
//!   Input:  "ssh://admin:secretpass@server.com:22?timeout=30"
//!   Output: "ssh://***@server.com:22?timeout=30"
//!
//! @param config The configuration string to sanitize
//! @return Sanitized configuration string with passwords and usernames replaced by ***
std::string sanitizeRemoteAutoTuningConfig(std::string const& config)
{
    if (config.empty())
    {
        return config;
    }

    try
    {
        // Find the protocol part (before ://)
        size_t protocolEnd = config.find("://");
        if (protocolEnd == std::string::npos)
        {
            return config; // Invalid format, return as is
        }

        // Find the credentials part (between :// and @)
        size_t credentialsStart = protocolEnd + 3;
        if (credentialsStart >= config.length())
        {
            return config; // Truncated after protocol
        }

        size_t credentialsEnd = config.find('@', credentialsStart);
        if (credentialsEnd == std::string::npos)
        {
            return config; // No credentials, return as is
        }

        // Extract parts and sanitize
        std::string protocol = config.substr(0, protocolEnd);
        std::string hostAndParams = config.substr(credentialsEnd);

        // Return sanitized version
        return protocol + "://***" + hostAndParams;
    }
    catch (std::exception const& e)
    {
        sample::gLogError << "Exception in sanitizeRemoteAutoTuningConfig: " << e.what() << std::endl;
        return config; // Return original on error
    }
    catch (...)
    {
        sample::gLogError << "Unknown exception in sanitizeRemoteAutoTuningConfig" << std::endl;
        return config; // Return original on error
    }
}

bool validateNonEmpty(std::string const& value, std::string const& flagName)
{
    if (value.empty())
    {
        sample::gLogError << flagName << " cannot be empty" << std::endl;
        return false;
    }
    return true;
}

bool validateRemoteAutoTuningConfig(std::string const& config)
{
    if (config.find("://") == std::string::npos)
    {
        sample::gLogError << "Invalid remote auto tuning config format. Expected format: "
                             "protocol://username[:password]@hostname[:port]?param1=value1&param2=value2"
                          << std::endl;
        return false;
    }
    return true;
}

std::vector<std::string> sanitizeArgv(int32_t argc, char** argv)
{
    std::vector<std::string> sanitizedArgs;
    sanitizedArgs.reserve(argc);

    for (int32_t i = 0; i < argc; ++i)
    {
        std::string arg = argv[i];

        // Sanitize remoteAutoTuningConfig argument
        if (auto const flag = std::string("--remoteAutoTuningConfig=");
            arg.size() > flag.size() && arg.substr(0, flag.size()) == flag)
        {
            arg = std::string(flag) + sanitizeRemoteAutoTuningConfig(arg.substr(flag.size()));
        }

        sanitizedArgs.push_back(arg);
    }

    return sanitizedArgs;
}

} // namespace sample
