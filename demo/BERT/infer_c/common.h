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

#ifndef INFER_C_COMMON_H
#define INFER_C_COMMON_H

#include "logging.h"
#include <cuda_runtime_api.h>
#include <getopt.h>
#include <memory>
#include <vector>

struct Args
{
    bool help{false};
    std::string engine{};
    std::vector<int> batchSize;
    int sequenceLength{128};
    int iterations{200};
    int warmUpRuns{10};
    int randomSeed{12345};
    bool enableGraph{false};
};

//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
inline bool parseArgs(Args& args, int argc, char* argv[])
{
    while (1)
    {
        int arg;
        // clang-format off
        static struct option long_options[] =
        {
            {"help", no_argument, 0, 'h'},
            {"engine", required_argument, 0, 'e'},
            {"batch_size", required_argument, 0, 'b'},
            {"sequence_length", no_argument, 0, 's'},
            {"iterations", required_argument, 0, 'i'},
            {"warm_up_runs", required_argument, 0, 'w'},
            {"ramdon_seed", required_argument, 0, 'r'},
            {"enable_graph", no_argument, 0, 'g'},
            {nullptr, 0, nullptr, 0}
        };
        // clang-format on
        int option_index = 0;
        arg = getopt_long(argc, argv, "he:b:s:i:w:r:g", long_options, &option_index);
        if (arg == -1)
        {
            break;
        }

        switch (arg)
        {
        case 'h': args.help = true; return false;
        case 'e':
            if (optarg)
            {
                args.engine = optarg;
            }
            else
            {
                std::cerr << "ERROR: --engine requires option argument" << std::endl;
                return false;
            }
            break;
        case 'b':
            if (optarg)
            {
                args.batchSize.push_back(std::stoi(optarg));
            }
            else
            {
                std::cerr << "ERROR: --batch_size requires option argument" << std::endl;
                return false;
            }
            break;
        case 's':
            if (optarg)
            {
                args.sequenceLength = std::stoi(optarg);
            }
            else
            {
                std::cerr << "ERROR: --sequence_length requires option argument" << std::endl;
                return false;
            }
            break;
        case 'i':
            if (optarg)
            {
                args.iterations = std::stoi(optarg);
            }
            else
            {
                std::cerr << "ERROR: --iterations requires option argument" << std::endl;
                return false;
            }
            break;
        case 'w':
            if (optarg)
            {
                args.warmUpRuns = std::stoi(optarg);
            }
            else
            {
                std::cerr << "ERROR: --warm_up_runs requires option argument" << std::endl;
                return false;
            }
            break;
        case 'r':
            if (optarg)
            {
                args.randomSeed = std::stoi(optarg);
            }
            else
            {
                std::cerr << "ERROR: --random_seed requires option argument" << std::endl;
                return false;
            }
            break;
        case 'g': args.enableGraph = true; break;
        default: return false;
        }
    }
    return true;
}

// clang-format off
#define gpuErrChk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
// clang-format on

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        gLogError << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << "\n";
        if (abort)
        {
            exit(code);
        }
    }
}

template <typename T>
struct TrtDestroyer
{
    void operator()(T* t)
    {
        t->destroy();
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

#endif // INFER_C_COMMON_H
