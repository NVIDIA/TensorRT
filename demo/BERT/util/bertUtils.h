/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef TRT_BERT_UTILS_H
#define TRT_BERT_UTILS_H

#include "cuda_profiler_api.h"
#include <getopt.h>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace bert
{

using WeightMap = std::map<std::string, Weights>;

struct BertConfig
{
    int numAttentionHeads; // N
    int hiddenSize;        // H*N
    int headSize;          // H
    int intermediateSize;  // 4*H*N
    int numHiddenLayers;

    bool use_fp16;

    BertConfig(int num_heads, int hidden_size, int intermediate_size, int num_hidden_layers, bool use_fp16)
        : hiddenSize(hidden_size)
        , numAttentionHeads(num_heads)
        , intermediateSize(intermediate_size)
        , numHiddenLayers(num_hidden_layers)
        , use_fp16(use_fp16)
    {

        assert(hiddenSize % numAttentionHeads == 0);
        headSize = hiddenSize / numAttentionHeads;
    }
};

inline void setTensorName(ITensor* tensor, const std::string& prefix, const std::string& name)
{
    tensor->setName((prefix + name).c_str());
}

inline void setOutputName(ILayer* layer, const std::string& prefix, const std::string& name, int out_idx = 0)
{
    setTensorName(layer->getOutput(out_idx), prefix, name);
}

//!
//! /brief Struct to maintain command-line arguments.
//!
struct Args
{
    bool runInFp16{false};
    bool help{false};
    int numHeads;
    std::string saveEngine{};
    std::vector<std::string> dataDirs;
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
        static struct option long_options[] = {{"help", no_argument, 0, 'h'}, {"datadir", required_argument, 0, 'd'},
            {"fp16", no_argument, 0, 'f'}, {"nheads", required_argument, 0, 'n'},
            {"saveEngine", required_argument, 0, 's'}, {nullptr, 0, nullptr, 0}};
        int option_index = 0;
        arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
        if (arg == -1)
            break;

        switch (arg)
        {
        case 'h': args.help = true; return false;
        case 'n':
            if (optarg)
                args.numHeads = std::stoi(optarg);
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }
            break;

        case 'd':
            if (optarg)
                args.dataDirs.push_back(optarg);
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
            else
            {
                std::cerr << "ERROR: --saveEngine requires option argument" << std::endl;
                return false;
            }
            break;
        case 'f': args.runInFp16 = true; break;
        default: return false;
        }
    }
    return true;
}

inline bool operator==(const nvinfer1::Dims& d1, const nvinfer1::Dims& d2)
{
    if (d1.d == d2.d)
    {
        return true;
    }
    if (d1.nbDims != d2.nbDims)
    {
        return false;
    }

    return std::equal(d1.d, d1.d + d1.nbDims, d2.d);
}
}
#endif // TRT_BERT_UTILS_H
