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

#include "sampleDevice.h"

#include <iomanip>

namespace sample
{

void cudaCheck(cudaError_t ret, std::ostream& err)
{
    if (ret != cudaSuccess)
    {
        err << "Cuda failure: " << cudaGetErrorString(ret) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Construct GPU UUID string in the same format as nvidia-smi does.
std::string getUuidString(cudaUUID_t uuid)
{
    constexpr int32_t kUUID_SIZE = sizeof(cudaUUID_t);
    static_assert(kUUID_SIZE == 16, "Unexpected size for cudaUUID_t!");

    std::ostringstream ss;
    std::vector<int32_t> const splits = {0, 4, 6, 8, 10, kUUID_SIZE};

    ss << "GPU" << std::hex << std::setfill('0');
    for (int32_t splitIdx = 0; splitIdx < static_cast<int32_t>(splits.size()) - 1; ++splitIdx)
    {
        ss << "-";
        for (int32_t byteIdx = splits[splitIdx]; byteIdx < splits[splitIdx + 1]; ++byteIdx)
        {
            ss << std::setw(2) << +static_cast<uint8_t>(uuid.bytes[byteIdx]);
        }
    }
    return ss.str();
}

void setCudaDevice(int32_t device, std::ostream& os)
{
#if !TRT_WINML
    os << "=== Device Information ===" << std::endl;

    // Get the number of visible GPUs.
    int32_t nbDevices{-1};
    cudaCheck(cudaGetDeviceCount(&nbDevices));

    if (nbDevices <= 0)
    {
        os << "Cannot find any available devices (GPUs)!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Print out the GPU name and PCIe bus ID of each GPU.
    os << "Available Devices: " << std::endl;
    cudaDeviceProp properties;
    for (int32_t deviceIdx = 0; deviceIdx < nbDevices; ++deviceIdx)
    {
        cudaDeviceProp tempProperties;
        cudaCheck(cudaGetDeviceProperties(&tempProperties, deviceIdx));

        // clang-format off
        os << "  Device " << deviceIdx << ": \"" << tempProperties.name << "\" UUID: "
           << getUuidString(tempProperties.uuid) << std::endl;
        // clang-format on

        // Record the properties of the desired GPU.
        if (deviceIdx == device)
        {
            properties = tempProperties;
        }
    }

    // Exit with error if the requested device ID does not exist.
    if (device < 0 || device >= nbDevices)
    {
        os << "Cannot find device ID " << device << "!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Set to the corresponding GPU.
    cudaCheck(cudaSetDevice(device));

    // clang-format off
    os << "Selected Device: "      << properties.name                                               << std::endl;
    os << "Selected Device ID: "   << device                                                        << std::endl;
    os << "Selected Device UUID: " << getUuidString(properties.uuid)                                << std::endl;
    os << "Compute Capability: "   << properties.major << "." << properties.minor                   << std::endl;
    os << "SMs: "                  << properties.multiProcessorCount                                << std::endl;
    os << "Device Global Memory: " << (properties.totalGlobalMem >> 20) << " MiB"                   << std::endl;
    os << "Shared Memory per SM: " << (properties.sharedMemPerMultiprocessor >> 10) << " KiB"       << std::endl;
    os << "Memory Bus Width: "     << properties.memoryBusWidth << " bits"
                        << " (ECC " << (properties.ECCEnabled != 0 ? "enabled" : "disabled") << ")" << std::endl;
    os << "Application Compute Clock Rate: "   << properties.clockRate / 1000000.0F << " GHz"       << std::endl;
    os << "Application Memory Clock Rate: "    << properties.memoryClockRate / 1000000.0F << " GHz" << std::endl;
    os << std::endl;
    os << "Note: The application clock rates do not reflect the actual clock rates that the GPU is "
                                                                         << "currently running at." << std::endl;
    // clang-format on
#endif
}

int32_t getCudaDriverVersion()
{
    int32_t version{-1};
    cudaCheck(cudaDriverGetVersion(&version));
    return version;
}

int32_t getCudaRuntimeVersion()
{
    int32_t version{-1};
    cudaCheck(cudaRuntimeGetVersion(&version));
    return version;
}

} // namespace sample
