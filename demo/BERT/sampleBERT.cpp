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

#include "NvInfer.h"
#include "common.h"
#include "logger.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>

#include "bert_util.hpp"

#include "data_utils.hpp"

#include "bert_encoder.hpp"
#include "emb_layer_norm_plugin.hpp"
#include "task_specific.hpp"

using namespace bert;

Args gArgs;

const std::string gSampleName = "TensorRT.sample_bert";
const std::string TEST_INPUT_NAME = "test_inputs.weights_int32";
const std::string TEST_OUTPUT_NAME = "test_outputs.weights";
const std::string BERT_WEIGHTS_NAME = "bert.weights";
const int NUM_RUNS = 5;

void doInference(IExecutionContext& context, const std::map<std::string, nvinfer1::Weights>& in_cfg,
    std::map<std::string, std::vector<float>>& out_cfg, int batchSize, cudaStream_t stream,
    std::vector<float>& times_tot, std::vector<float>& times_cmp, int verbose = 1)
{

    int n_runs = times_tot.size();
    assert(n_runs == times_cmp.size());
    assert(n_runs > 0);

    const ICudaEngine& engine = context.getEngine();
    const int n_bind = engine.getNbBindings();
    assert(n_bind == in_cfg.size() + out_cfg.size());
    std::vector<void*> buffers(n_bind);
    alloc_bindings(engine, buffers, batchSize, in_cfg, verbose);
    alloc_bindings(engine, buffers, batchSize, out_cfg, verbose);

    void** bs = &buffers[0];

    std::vector<cudaEvent_t> starts_tot(n_runs);
    std::vector<cudaEvent_t> stops_tot(n_runs);
    std::vector<cudaEvent_t> starts_cmp(n_runs);
    std::vector<cudaEvent_t> stops_cmp(n_runs);

    for (int it = 0; it < n_runs; it++)
    {
        cudaEventCreate(&starts_tot[it]);
        cudaEventCreate(&stops_tot[it]);

        cudaEventCreate(&starts_cmp[it]);
        cudaEventCreate(&stops_cmp[it]);
    }

    cudaProfilerStart();
    for (int it = 0; it < n_runs; it++)
    {
        CHECK(cudaEventRecord(starts_tot[it], stream));
        upload(engine, buffers, batchSize, in_cfg, stream);
        CHECK(cudaEventRecord(starts_cmp[it], stream));
        context.enqueue(batchSize, bs, stream, nullptr);
        CHECK(cudaEventRecord(stops_cmp[it], stream));
        download(engine, buffers, batchSize, out_cfg, stream);
        CHECK(cudaEventRecord(stops_tot[it], stream));
    }
    CHECK(cudaDeviceSynchronize());

    cudaProfilerStop();
    float milliseconds = 0;
    for (int it = 0; it < n_runs; it++)
    {
        cudaEventElapsedTime(&milliseconds, starts_tot[it], stops_tot[it]);
        times_tot[it] = milliseconds;
        cudaEventElapsedTime(&milliseconds, starts_cmp[it], stops_cmp[it]);
        times_cmp[it] = milliseconds;

        cudaEventDestroy(starts_tot[it]);
        cudaEventDestroy(stops_tot[it]);
        cudaEventDestroy(starts_cmp[it]);
        cudaEventDestroy(stops_cmp[it]);

        printf("Run %d; Total: %fms Comp.only: %fms\n", it, times_tot[it], times_cmp[it]);
    }

    cudaProfilerStop();

    for (auto& devptr : buffers)
    {
        CHECK(cudaFree(devptr));
    }
}

// Create the Engine using only the API and not any parser.
nvinfer1::ICudaEngine* fromAPIToModel(nvinfer1::IBuilder* builder, const int num_heads, const int B, const int S)
{

    // Currently, the batch size is handled in the model, by passing an input
    // Tensor with an explicit batch dimension. There is a tranpose in the
    // attention layer, that relies on the exact batch size to be available at
    // network definition time. This will change in the near future.
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(5000_MB);
    builder->setFp16Mode(gArgs.runInFp16);
    if (gArgs.runInFp16)
    {
        gLogInfo << ("Running in FP 16 Mode\n");
        builder->setStrictTypeConstraints(true);
    }

    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    // infer these from the parameters
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int hidden_size = 0;

    WeightDict init_dict;

    const std::string wts_path(locateFile("bert.weights", gArgs.dataDirs));

    load_weights(wts_path, init_dict);
    infer_network_sizes(init_dict, hidden_size, intermediate_size, num_hidden_layers);
    assert(intermediate_size);
    assert(hidden_size);
    assert(num_hidden_layers);

    /// Embeddings Layer

    ITensor* input_ids = network->addInput("input_ids", DataType::kINT32, Dims2{B, S});

    ITensor* segment_ids = network->addInput("segment_ids", DataType::kINT32, Dims2{B, S});

    ITensor* input_mask = network->addInput("input_mask", DataType::kINT32, Dims2{B, S});

    const Weights& wbeta = init_dict.at("bert_embeddings_layernorm_beta");
    const Weights& wgamma = init_dict.at("bert_embeddings_layernorm_gamma");
    const Weights& wwordemb = init_dict.at("bert_embeddings_word_embeddings");
    const Weights& wtokemb = init_dict.at("bert_embeddings_token_type_embeddings");
    const Weights& wposemb = init_dict.at("bert_embeddings_position_embeddings");
    ITensor* inputs[3] = {input_ids, segment_ids, input_mask};

    auto emb_plug = EmbLayerNormPlugin("embeddings", gArgs.runInFp16, wbeta, wgamma, wwordemb, wposemb, wtokemb);
    IPluginV2Layer* emb_layer = network->addPluginV2(inputs, 3, emb_plug);
    set_name(emb_layer, "embeddings", "output");

    ITensor* embeddings = emb_layer->getOutput(0);
    ITensor* mask_idx = emb_layer->getOutput(1);

    /// BERT Encoder

    BertConfig config(num_heads, hidden_size, intermediate_size, num_hidden_layers, gArgs.runInFp16);

    ITensor* bert_out = bert_model(config, init_dict, network, embeddings, mask_idx, nullptr);

    /// SQuAD Output Layer

    ITensor* squad_logits = squad_output("cls_", config, init_dict, network, bert_out, nullptr);

    network->markOutput(*squad_logits);

    // Build the engine

    auto engine = builder->buildCudaEngine(*network);
    // we don't need the network any more
    network->destroy();

    // Once we have built the cuda engine, we can release all of our held memory.
    for (auto& w : init_dict)
        free(const_cast<void*>(w.second.values));
    return engine;
}

nvinfer1::ICudaEngine* APIToModel(const int num_heads, const int B, const int S)
{
    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // create the model to populate the network, then set the outputs and create an engine
    nvinfer1::ICudaEngine* engine = fromAPIToModel(builder, num_heads, B, S);

    assert(engine != nullptr);

    builder->destroy();
    return engine;
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_bert [-h or --help] [-d or --datadir=<path to data directory>] [--fp1 ]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. The given path(s) must contain the weights and test "
                 "inputs/outputs."
              << std::endl;
    std::cout << "--fp16          OPTIONAL: Run in FP16 mode." << std::endl;
    std::cout << "--nheads        Number of attention heads." << std::endl;
    std::cout << "--saveEngine    The path at which to write a serialized engine." << std::endl;
}

int main(int argc, char* argv[])
{
    int S = 0;
    int Bmax = 0;

    bool argsOK = parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gLogError << "No datadirs given" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.num_heads <= 0)
    {
        gLogError << "invalid number of heads" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    const std::string wts_path(locateFile(TEST_OUTPUT_NAME, gArgs.dataDirs));
    std::map<std::string, nvinfer1::Weights> test_outputs;
    load_weights(wts_path, test_outputs);

    std::vector<nvinfer1::Weights> in_ids;
    std::vector<nvinfer1::Weights> in_masks;
    std::vector<nvinfer1::Weights> segment_ids;
    std::vector<nvinfer1::Dims> in_dims;
    std::string records_path(locateFile(TEST_INPUT_NAME, gArgs.dataDirs));
    load_inputs(records_path, Bmax, S, in_ids, in_masks, segment_ids, in_dims);

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    const int num_heads = gArgs.num_heads;
    nvinfer1::ICudaEngine* engine = APIToModel(num_heads, Bmax, S);
    if (engine == nullptr)
    {
        gLogError << "Unable to build engine." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    std::ofstream engineFile(gArgs.saveEngine, std::ios::binary);
    if (!engineFile)
    {
        gLogError << "Cannot open engine file: " << gArgs.saveEngine << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    nvinfer1::IHostMemory* serializedEngine{engine->serialize()};
    if (serializedEngine == nullptr)
    {
        gLogError << "Engine serialization failed" << std::endl;
        return false;
    }

    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    serializedEngine->destroy();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr)
    {
        gLogError << "Unable to create runtime." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (context == nullptr)
    {
        gLogError << "Unable to create context." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    std::map<std::string, nvinfer1::Weights> in_cfg{std::make_pair("input_ids", in_ids[0]),
        std::make_pair("input_mask", in_masks[0]), std::make_pair("segment_ids", segment_ids[0])};

    std::string output_name("cls_squad_logits");
    std::map<std::string, std::vector<float>> out_cfg = {make_pair(output_name, std::vector<float>(2 * Bmax * S))};

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::vector<float> times_tot(NUM_RUNS); // total time
    std::vector<float> times_cmp(NUM_RUNS); // computation time

    doInference(*context, in_cfg, out_cfg, 1, stream, times_tot, times_cmp);

    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
    auto& output = out_cfg[output_name];
    const float* test = reinterpret_cast<const float*>(test_outputs["logits"].values);

    float mae = 0;
    float maxdiff = 0;
    for (int it = 0; it < test_outputs["logits"].count; it++)
    {
        float diff = std::abs(test[it] - output[it]);
        mae += diff;
        maxdiff = std::max(diff, maxdiff);
    }
    float avg_tot = std::accumulate(times_tot.begin(), times_tot.end(), 0.f, std::plus<float>()) / times_tot.size();
    float avg_cmp = std::accumulate(times_cmp.begin(), times_cmp.end(), 0.f, std::plus<float>()) / times_cmp.size();

    printf("B=%d S=%d MAE=%.12e MaxDiff=%.12e ", Bmax, S, (mae) / output.size(), maxdiff);
    printf(" Runtime(total avg)=%.6fms Runtime(comp ms)=%.6f\n", avg_tot, avg_cmp);

    // destroy the engine
    bool pass{true};

    return gLogger.reportTest(sampleTest, pass);
}
