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

#pragma once

#include "cuda_profiler_api.h"
#include <getopt.h>

namespace bert
{

typedef std::map<std::string, Weights> WeightDict;
typedef std::map<std::string, ITensor*> TensorDict;

struct BertConfig
{
    int num_attention_heads;
    int hidden_size;
    int head_size;
    int intermediate_size;
    int num_hidden_layers;

    bool use_fp16;

    BertConfig(int num_heads_, int hidden_size_, int intermediate_size_, int num_hidden_layers_, bool use_fp16_)
        : hidden_size(hidden_size_)
        , num_attention_heads(num_heads_)
        , intermediate_size(intermediate_size_)
        , num_hidden_layers(num_hidden_layers_)
        , use_fp16(use_fp16_)
    {

        assert(hidden_size % num_attention_heads == 0);
        head_size = hidden_size / num_attention_heads;
    }
};

Weights noop{DataType::kFLOAT, nullptr, 0};

int size(ITensor* t)
{
    Dims dims = t->getDimensions();
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

void set_name(ITensor* tensor, const std::string& prefix, const std::string& name)
{
    tensor->setName((prefix + name).c_str());
}

void set_name(ILayer* layer, const std::string& prefix, const std::string& name, int out_idx = 0)
{
    set_name(layer->getOutput(out_idx), prefix, name);
}

//!
//! /brief Struct to maintain command-line arguments.
//!
struct Args
{
    bool runInFp16{false};
    bool help{false};
    int num_heads;
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
            {"fp16", no_argument, 0, 'f'}, {"nheads", required_argument, 0, 'n'}, {nullptr, 0, nullptr, 0}};
        int option_index = 0;
        arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
        if (arg == -1)
            break;

        switch (arg)
        {
        case 'h': args.help = true; return false;
        case 'n':
            if (optarg)
                args.num_heads = std::stoi(optarg);
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
        case 'f': args.runInFp16 = true; break;
        default: return false;
        }
    }
    return true;
}
}
