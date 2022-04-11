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
#ifndef TENSORRT_ARGS_PARSER_H
#define TENSORRT_ARGS_PARSER_H

#include <string>
#include <vector>
#ifdef _MSC_VER
#include "..\common\windows\getopt.h"
#else
#include <getopt.h>
#endif
#include <iostream>

namespace samplesCommon
{

//!
//! \brief The SampleParams structure groups the basic parameters required by
//!        all sample networks.
//!
struct SampleParams
{
    int32_t batchSize{1};              //!< Number of inputs in a batch
    int32_t dlaCore{-1};               //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

//!
//! \brief The CaffeSampleParams structure groups the additional parameters required by
//!         networks that use caffe
//!
struct CaffeSampleParams : public SampleParams
{
    std::string prototxtFileName; //!< Filename of prototxt design file of a network
    std::string weightsFileName;  //!< Filename of trained weights file of a network
    std::string meanFileName;     //!< Filename of mean file of a network
};

//!
//! \brief The OnnxSampleParams structure groups the additional parameters required by
//!         networks that use ONNX
//!
struct OnnxSampleParams : public SampleParams
{
    std::string onnxFileName; //!< Filename of ONNX file of a network
};

//!
//! \brief The UffSampleParams structure groups the additional parameters required by
//!         networks that use Uff
//!
struct UffSampleParams : public SampleParams
{
    std::string uffFileName; //!< Filename of uff file of a network
};

//!
//! /brief Struct to maintain command-line arguments.
//!
struct Args
{
    bool runInInt8{false};
    bool runInFp16{false};
    bool help{false};
    int32_t useDLACore{-1};
    int32_t batch{1};
    std::vector<std::string> dataDirs;
    std::string saveEngine;
    std::string loadEngine;
    bool useILoop{false};
};

//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
inline bool parseArgs(Args& args, int32_t argc, char* argv[])
{
    while (1)
    {
        int32_t arg;
        static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"datadir", required_argument, 0, 'd'},
            {"int8", no_argument, 0, 'i'}, {"fp16", no_argument, 0, 'f'}, {"useILoop", no_argument, 0, 'l'},
            {"saveEngine", required_argument, 0, 's'}, {"loadEngine", required_argument, 0, 'o'},
            {"useDLACore", required_argument, 0, 'u'}, {"batch", required_argument, 0, 'b'}, {nullptr, 0, nullptr, 0}};
        int32_t option_index = 0;
        arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
        if (arg == -1)
        {
            break;
        }

        switch (arg)
        {
        case 'h': args.help = true; return true;
        case 'd':
            if (optarg)
            {
                args.dataDirs.push_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }
            break;
        case 's':
            if (optarg)
            {
                args.saveEngine = optarg;
            }
            break;
        case 'o':
            if (optarg)
            {
                args.loadEngine = optarg;
            }
            break;
        case 'i': args.runInInt8 = true; break;
        case 'f': args.runInFp16 = true; break;
        case 'l': args.useILoop = true; break;
        case 'u':
            if (optarg)
            {
                args.useDLACore = std::stoi(optarg);
            }
            break;
        case 'b':
            if (optarg)
            {
                args.batch = std::stoi(optarg);
            }
            break;
        default: return false;
        }
    }
    return true;
}

} // namespace samplesCommon

#endif // TENSORRT_ARGS_PARSER_H
