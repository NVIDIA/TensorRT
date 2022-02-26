/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "NvInfer.h"
#include "argsParser.h"
#include "common.h"
#include "data/benchmarkWriter.h"
#include "data/bleuScoreWriter.h"
#include "data/dataReader.h"
#include "data/dataWriter.h"
#include "data/limitedSamplesDataReader.h"
#include "data/sequenceProperties.h"
#include "data/textReader.h"
#include "data/textWriter.h"
#include "data/vocabulary.h"
#include "deviceBuffer.h"
#include "logger.h"
#include "model/alignment.h"
#include "model/attention.h"
#include "model/beamSearchPolicy.h"
#include "model/componentWeights.h"
#include "model/contextNMT.h"
#include "model/decoder.h"
#include "model/embedder.h"
#include "model/encoder.h"
#include "model/likelihood.h"
#include "model/lstmDecoder.h"
#include "model/lstmEncoder.h"
#include "model/multiplicativeAlignment.h"
#include "model/projection.h"
#include "model/slpAttention.h"
#include "model/slpEmbedder.h"
#include "model/slpProjection.h"
#include "model/softmaxLikelihood.h"
#include "pinnedHostBuffer.h"
#include "trtUtil.h"

bool gPrintComponentInfo = true;
bool gFeedAttentionToInput = true;

int32_t gMaxBatchSize = 128;
int32_t gBeamWidth = 5;
int32_t gMaxInputSequenceLength = 150;
int32_t gMaxOutputSequenceLength = -1;
int32_t gMaxInferenceSamples = -1;
std::string gDataWriterStr = "bleu";
std::string gOutputTextFileName("translation_output.txt");
int32_t gMaxWorkspaceSize = 512_MiB;
std::string gDataDirectory("data/samples/nmt/deen");
bool gEnableProfiling = false;
bool gAggregateProfiling = false;
bool gFp16 = false;
bool gVerbose = false;
bool gInt8 = false;
int32_t gUseDLACore{-1};
int32_t gPadMultiple = 1;

const std::string gSampleName = "TensorRT.sample_nmt";

std::string gInputTextFileName("newstest2015.tok.bpe.32000.de");
std::string gReferenceOutputTextFileName("newstest2015.tok.bpe.32000.en");
std::string gInputVocabularyFileName("vocab.bpe.32000.de");
std::string gOutputVocabularyFileName("vocab.bpe.32000.en");
std::string gEncEmbedFileName("weights/encembed.bin");
std::string gEncRnnFileName("weights/encrnn.bin");
std::string gDecEmbedFileName("weights/decembed.bin");
std::string gDecRnnFileName("weights/decrnn.bin");
std::string gDecAttFileName("weights/decatt.bin");
std::string gDecMemFileName("weights/decmem.bin");
std::string gDecProjFileName("weights/decproj.bin");
nmtSample::Vocabulary::ptr gOutputVocabulary = std::make_shared<nmtSample::Vocabulary>();

std::string locateNMTFile(const std::string& fpathSuffix)
{
    std::vector<std::string> dirs{std::string(gDataDirectory) + "/", "data/nmt/deen/"};
    return locateFile(fpathSuffix, dirs);
}

nmtSample::SequenceProperties::ptr getOutputSequenceProperties()
{
    return gOutputVocabulary;
}

nmtSample::DataReader::ptr getDataReader()
{
    std::shared_ptr<std::istream> textInput(new std::ifstream(locateNMTFile(gInputTextFileName)));
    std::shared_ptr<std::istream> vocabInput(new std::ifstream(locateNMTFile(gInputVocabularyFileName)));
    ASSERT(textInput->good());
    ASSERT(vocabInput->good());

    auto vocabulary = std::make_shared<nmtSample::Vocabulary>();
    *vocabInput >> *vocabulary;

    auto reader = std::make_shared<nmtSample::TextReader>(textInput, vocabulary);

    if (gMaxInferenceSamples >= 0)
    {
        return std::make_shared<nmtSample::LimitedSamplesDataReader>(gMaxInferenceSamples, reader);
    }
    else
    {
        return reader;
    }
}

template <typename Component>
std::shared_ptr<Component> buildNMTComponentFromWeightsFile(const std::string& filename)
{
    auto weights = std::make_shared<nmtSample::ComponentWeights>();
    std::ifstream input(locateNMTFile(filename), std::ios::binary);
    ASSERT(input.good());
    input >> *weights;

    return std::make_shared<Component>(weights);
}

nmtSample::Embedder::ptr getInputEmbedder()
{
    return buildNMTComponentFromWeightsFile<nmtSample::SLPEmbedder>(gEncEmbedFileName);
}

nmtSample::Embedder::ptr getOutputEmbedder()
{
    return buildNMTComponentFromWeightsFile<nmtSample::SLPEmbedder>(gDecEmbedFileName);
}

nmtSample::Encoder::ptr getEncoder()
{
    return buildNMTComponentFromWeightsFile<nmtSample::LSTMEncoder>(gEncRnnFileName);
}

nmtSample::Alignment::ptr getAlignment()
{
    return buildNMTComponentFromWeightsFile<nmtSample::MultiplicativeAlignment>(gDecMemFileName);
}

nmtSample::Context::ptr getContext()
{
    return std::make_shared<nmtSample::Context>();
}

nmtSample::Decoder::ptr getDecoder()
{
    return buildNMTComponentFromWeightsFile<nmtSample::LSTMDecoder>(gDecRnnFileName);
}

nmtSample::Attention::ptr getAttention()
{
    return buildNMTComponentFromWeightsFile<nmtSample::SLPAttention>(gDecAttFileName);
}

nmtSample::Projection::ptr getProjection()
{
    return buildNMTComponentFromWeightsFile<nmtSample::SLPProjection>(gDecProjFileName);
}

nmtSample::Likelihood::ptr getLikelihood()
{
    return std::make_shared<nmtSample::SoftmaxLikelihood>();
}

nmtSample::BeamSearchPolicy::ptr getSearchPolicy(
    int32_t endSequenceId, nmtSample::LikelihoodCombinationOperator::ptr likelihoodCombinationOperator)
{
    return std::make_shared<nmtSample::BeamSearchPolicy>(endSequenceId, likelihoodCombinationOperator, gBeamWidth);
}

nmtSample::DataWriter::ptr getDataWriter()
{
    if (gDataWriterStr == "bleu")
    {
        std::shared_ptr<std::istream> textInput(new std::ifstream(locateNMTFile(gReferenceOutputTextFileName)));
        ASSERT(textInput->good());
        return std::make_shared<nmtSample::BLEUScoreWriter>(textInput, gOutputVocabulary);
    }
    else if (gDataWriterStr == "text")
    {
        std::remove(gOutputTextFileName.data());
        std::shared_ptr<std::ostream> textOutput(new std::ofstream(gOutputTextFileName));
        // cppcheck-suppress incorrectStringBooleanError
        ASSERT(textOutput->good()
            && "Please contact system administrator if you have no permission to write the file "
               "translation_output.txt");
        return std::make_shared<nmtSample::TextWriter>(textOutput, gOutputVocabulary);
    }
    else if (gDataWriterStr == "benchmark")
    {
        return std::make_shared<nmtSample::BenchmarkWriter>();
    }
    else
    {
        sample::gLogError << "Invalid data writer specified: " << gDataWriterStr << std::endl;
        ASSERT(0);
        return nmtSample::DataWriter::ptr();
    }
}

bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        sample::gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool parseInt(const char* arg, const char* name, int32_t& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atoi(arg + n + 3);
        sample::gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool parseBool(const char* arg, const char* longName, bool& value, char shortName = 0)
{
    bool match = false;

    if (shortName)
    {
        match = (arg[0] == '-') && (arg[1] == shortName);
    }
    if (!match && longName)
    {
        const size_t n = strlen(longName);
        match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, longName, n);
    }
    if (match)
    {
        sample::gLogInfo << longName << ": true" << std::endl;
        value = true;
    }
    return match;
}

void printUsage()
{
    printf("\nOptional params:\n");
    printf("  --help, -h                           Output help message and exit\n");
    printf("  --data_writer=bleu/text/benchmark    Type of the output the app generates (default = %s)\n",
        gDataWriterStr.c_str());
    printf("  --output_file=<path_to_file>         Path to the output file when data_writer=text (default = %s)\n",
        gOutputTextFileName.c_str());
    printf("  --batch=<N>                          Batch size (default = %d)\n", gMaxBatchSize);
    printf("  --beam=<N>                           Beam width (default = %d)\n", gBeamWidth);
    printf("  --max_input_sequence_length=<N>      Maximum length for input sequences (default = %d)\n",
        gMaxInputSequenceLength);
    printf(
        "  --max_output_sequence_length=<N>     Maximum length for output sequences (default = %d), negative value "
        "indicates no limit\n",
        gMaxOutputSequenceLength);
    printf(
        "  --max_inference_samples=<N>          Maximum sample count to run inference for, negative values indicates "
        "no limit is set (default = %d)\n",
        gMaxInferenceSamples);
    printf("  --verbose                            Output verbose-level messages by TensorRT\n");
    printf("  --max_workspace_size=<N>             Maximum workspace size (default = %d)\n", gMaxWorkspaceSize);
    printf(
        "  --data_dir=<path_to_data_directory>  Path to the directory where data and weights are located (default = "
        "%s)\n",
        gDataDirectory.c_str());
    printf(
        "  --profile                            Profile TensorRT execution layer by layer. Use benchmark data_writer "
        "when profiling on, disregard benchmark results\n");
    printf("  --aggregate_profile                  Merge profiles from multiple TensorRT engines\n");
    printf("  --fp16                               Switch on fp16 math\n");
    printf("  --int8                               Switch on int8 math\n");
    printf(
        "  --useDLACore=N                       Specify a DLA engine for layers that support DLA. Value can range from "
        "0 to n-1, where n is the number of DLA engines on the platform.\n");
    printf(
        "  --padMultiple=N                      Specify multiple to pad out matrix dimensions to test performance\n");
}

bool parseNMTArgs(samplesCommon::Args& args, int32_t argc, char* argv[])
{
    if (argc < 1)
    {
        printUsage();
        return false;
    }

    bool showHelp = false;
    for (int32_t j = 1; j < argc; j++)
    {
        if (parseBool(argv[j], "help", showHelp, 'h'))
            continue;
        if (parseString(argv[j], "data_writer", gDataWriterStr))
            continue;
        if (parseString(argv[j], "output_file", gOutputTextFileName))
            continue;
        if (parseInt(argv[j], "batch", gMaxBatchSize))
            continue;
        if (parseInt(argv[j], "beam", gBeamWidth))
            continue;
        if (parseInt(argv[j], "max_input_sequence_length", gMaxInputSequenceLength))
            continue;
        if (parseInt(argv[j], "max_output_sequence_length", gMaxOutputSequenceLength))
            continue;
        if (parseInt(argv[j], "max_inference_samples", gMaxInferenceSamples))
            continue;
        if (parseBool(argv[j], "verbose", gVerbose))
            continue;
        if (parseInt(argv[j], "max_workspace_size", gMaxWorkspaceSize))
            continue;
        if (parseString(argv[j], "data_dir", gDataDirectory))
            continue;
        if (parseBool(argv[j], "profile", gEnableProfiling))
            continue;
        if (parseBool(argv[j], "aggregate_profile", gAggregateProfiling))
            continue;
        if (parseBool(argv[j], "fp16", gFp16))
            continue;
        if (parseBool(argv[j], "int8", gInt8))
            continue;
        if (parseInt(argv[j], "useDLACore", gUseDLACore))
            continue;
        if (parseInt(argv[j], "padMultiple", gPadMultiple))
            continue;
    }

    if (showHelp)
    {
        printUsage();
        args.help = true;
        return false;
    }

    return true;
}

nvinfer1::ICudaEngine* getEncoderEngine(
    nmtSample::Embedder::ptr inputEmbedder, nmtSample::Encoder::ptr encoder, nmtSample::Alignment::ptr alignment)
{
    nvinfer1::IBuilder* encoderBuilder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    ASSERT(encoderBuilder != nullptr);
    nvinfer1::IBuilderConfig* encoderConfig = encoderBuilder->createBuilderConfig();
    encoderBuilder->setMaxBatchSize(gMaxBatchSize);
    encoderConfig->setMaxWorkspaceSize(gMaxWorkspaceSize);
    if (gFp16)
    {
        encoderConfig->setFlag(BuilderFlag::kFP16);
    }
    if (gInt8)
    {
        encoderConfig->setFlag(BuilderFlag::kINT8);
    }

    nvinfer1::INetworkDefinition* encoderNetwork = encoderBuilder->createNetworkV2(0);

    // Define inputs for the encoder
    nvinfer1::Dims inputDims{1, {gMaxInputSequenceLength}};
    auto inputEncoderDataTensor = encoderNetwork->addInput("input_encoder_data", nvinfer1::DataType::kINT32, inputDims);
    ASSERT(inputEncoderDataTensor != nullptr);
    nvinfer1::Dims inputSequenceLengthsDims{0, {}};
    auto actualInputSequenceLengthsTensor = encoderNetwork->addInput(
        "actual_input_sequence_lengths", nvinfer1::DataType::kINT32, inputSequenceLengthsDims);
    ASSERT(actualInputSequenceLengthsTensor != nullptr);
    nvinfer1::Dims inputSequenceLengthsWithUnitIndexDims{1, {1}};
    auto actualInputSequenceLengthsWithUnitIndexTensor
        = encoderNetwork->addInput("actual_input_sequence_lengths_with_index_dim", nvinfer1::DataType::kINT32,
            inputSequenceLengthsWithUnitIndexDims);
    ASSERT(actualInputSequenceLengthsWithUnitIndexTensor != nullptr);

    auto stateSizes = encoder->getStateSizes();
    std::vector<nvinfer1::ITensor*> encoderInputStatesTensors(stateSizes.size());
    for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_encoder_states_" << i;
        encoderInputStatesTensors[i] = encoderNetwork->addInput(
            ss.str().c_str(), gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, stateSizes[i]);
        ASSERT(encoderInputStatesTensors[i] != nullptr);
    }

    nvinfer1::ITensor* initializeDecoderIndicesTensor = nullptr;
    {
        nvinfer1::Dims inputDims{1, {gBeamWidth}};
        initializeDecoderIndicesTensor
            = encoderNetwork->addInput("initialize_decoder_indices", nvinfer1::DataType::kINT32, inputDims);
        ASSERT(initializeDecoderIndicesTensor != nullptr);
    }

    nvinfer1::ITensor* inputEncoderEmbeddedTensor;
    inputEmbedder->addToModel(encoderNetwork, inputEncoderDataTensor, &inputEncoderEmbeddedTensor);
    inputEncoderEmbeddedTensor->setName("input_data_embedded");

    nvinfer1::ITensor* memoryStatesTensor;
    std::vector<nvinfer1::ITensor*> encoderOutputStatesTensors(stateSizes.size());
    encoder->addToModel(encoderNetwork, gMaxInputSequenceLength, inputEncoderEmbeddedTensor,
        actualInputSequenceLengthsTensor, &encoderInputStatesTensors[0], &memoryStatesTensor,
        &encoderOutputStatesTensors[0]);
    memoryStatesTensor->setName("memory_states");
    encoderNetwork->markOutput(*memoryStatesTensor);
    memoryStatesTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);

    if (alignment->getAttentionKeySize() > 0)
    {
        nvinfer1::ITensor* attentionKeysTensor;
        alignment->addAttentionKeys(encoderNetwork, memoryStatesTensor, &attentionKeysTensor);
        attentionKeysTensor->setName("attention_keys");
        encoderNetwork->markOutput(*attentionKeysTensor);
        attentionKeysTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    // Replicate sequence lengths for the decoder
    {
        auto gatherLayer = encoderNetwork->addGather(
            *actualInputSequenceLengthsWithUnitIndexTensor, *initializeDecoderIndicesTensor, 0);
        ASSERT(gatherLayer != nullptr);
        gatherLayer->setName("Replicate input sequence lengths for decoder");
        auto actualInputSequenceLengthsReplicatedTensor = gatherLayer->getOutput(0);
        ASSERT(actualInputSequenceLengthsReplicatedTensor != nullptr);
        actualInputSequenceLengthsReplicatedTensor->setName("actual_input_sequence_lengths_replicated");
        encoderNetwork->markOutput(*actualInputSequenceLengthsReplicatedTensor);
        actualInputSequenceLengthsReplicatedTensor->setType(nvinfer1::DataType::kINT32);
    }

    {
        for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
        {
            ASSERT(encoderOutputStatesTensors[i] != nullptr);

            // Insert index (Z=1) dimension into tensor
            nvinfer1::ITensor* encoderOutputStatesTensorWithUnitIndex;
            {
                auto shuffleLayer = encoderNetwork->addShuffle(*encoderOutputStatesTensors[i]);
                ASSERT(shuffleLayer != nullptr);
                {
                    std::stringstream ss;
                    ss << "Reshape encoder states for decoder initialization " << i;
                    shuffleLayer->setName(ss.str().c_str());
                }
                nvinfer1::Dims shuffleDims;
                {
                    shuffleDims.nbDims = stateSizes[i].nbDims + 1;
                    shuffleDims.d[0] = 1;
                    for (int32_t j = 0; j < stateSizes[i].nbDims; ++j)
                    {
                        shuffleDims.d[j + 1] = stateSizes[i].d[j];
                    }
                }
                shuffleLayer->setReshapeDimensions(shuffleDims);
                encoderOutputStatesTensorWithUnitIndex = shuffleLayer->getOutput(0);
                ASSERT(encoderOutputStatesTensorWithUnitIndex != nullptr);
            }
            auto gatherLayer = encoderNetwork->addGather(
                *encoderOutputStatesTensorWithUnitIndex, *initializeDecoderIndicesTensor, 0);
            ASSERT(gatherLayer != nullptr);
            {
                std::stringstream ss;
                ss << "Replicate encoder states for decoder initialization " << i;
                gatherLayer->setName(ss.str().c_str());
            }
            auto inputDecoderHiddenStatesTensor = gatherLayer->getOutput(0);
            ASSERT(inputDecoderHiddenStatesTensor != nullptr);
            std::stringstream ss;
            ss << "input_decoder_states_" << i;
            inputDecoderHiddenStatesTensor->setName(ss.str().c_str());
            encoderNetwork->markOutput(*inputDecoderHiddenStatesTensor);
            inputDecoderHiddenStatesTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
        }
    }

    samplesCommon::setDummyInt8DynamicRanges(encoderConfig, encoderNetwork);
    samplesCommon::enableDLA(encoderBuilder, encoderConfig, gUseDLACore);
    auto encoderPlan = encoderBuilder->buildSerializedNetwork(*encoderNetwork, *encoderConfig);
    ASSERT(encoderPlan != nullptr);
    auto runtime = createInferRuntime(sample::gLogger.getTRTLogger());
    ASSERT(runtime != nullptr);
    auto res = runtime->deserializeCudaEngine(encoderPlan->data(), encoderPlan->size());
    runtime->destroy();
    encoderPlan->destroy();
    encoderNetwork->destroy();
    encoderBuilder->destroy();
    encoderConfig->destroy();
    return res;
}

nvinfer1::ICudaEngine* getGeneratorEngine(nmtSample::Embedder::ptr outputEmbedder, nmtSample::Decoder::ptr decoder,
    nmtSample::Alignment::ptr alignment, nmtSample::Context::ptr context, nmtSample::Attention::ptr attention,
    nmtSample::Projection::ptr projection, nmtSample::Likelihood::ptr likelihood)
{
    nvinfer1::IBuilder* generatorBuilder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    ASSERT(generatorBuilder != nullptr);
    nvinfer1::IBuilderConfig* generatorConfig = generatorBuilder->createBuilderConfig();
    generatorBuilder->setMaxBatchSize(gMaxBatchSize);
    generatorConfig->setMaxWorkspaceSize(gMaxWorkspaceSize);
    if (gFp16)
    {
        generatorConfig->setFlag(BuilderFlag::kFP16);
    }
    if (gInt8)
    {
        generatorConfig->setFlag(BuilderFlag::kINT8);
    }

    nvinfer1::INetworkDefinition* generatorNetwork = generatorBuilder->createNetworkV2(0);

    // Define inputs for the generator
    auto stateSizes = decoder->getStateSizes();
    std::vector<nvinfer1::ITensor*> decoderInputStatesTensors(stateSizes.size());
    for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        nvinfer1::Dims statesDims;
        {
            statesDims.nbDims = stateSizes[i].nbDims + 1;
            statesDims.d[0] = gBeamWidth;
            for (int32_t j = 0; j < stateSizes[i].nbDims; ++j)
            {
                statesDims.d[j + 1] = stateSizes[i].d[j];
            }
        }
        decoderInputStatesTensors[i] = generatorNetwork->addInput(
            ss.str().c_str(), gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, statesDims);
        ASSERT(decoderInputStatesTensors[i] != nullptr);
    }
    nvinfer1::Dims inputDecoderDataDims{1, {gBeamWidth}};
    auto inputDecoderDataTensor
        = generatorNetwork->addInput("input_decoder_data", nvinfer1::DataType::kINT32, inputDecoderDataDims);
    ASSERT(inputDecoderDataTensor != nullptr);
    nvinfer1::Dims inputSequenceLengthsTeplicatedDims{2, {gBeamWidth, 1}};
    auto actualInputSequenceLengthsReplicatedTensor = generatorNetwork->addInput(
        "actual_input_sequence_lengths_replicated", nvinfer1::DataType::kINT32, inputSequenceLengthsTeplicatedDims);
    ASSERT(actualInputSequenceLengthsReplicatedTensor != nullptr);
    nvinfer1::Dims memoryStatesDims{2, {gMaxInputSequenceLength, alignment->getSourceStatesSize()}};
    auto memoryStatesTensor = generatorNetwork->addInput(
        "memory_states", gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, memoryStatesDims);
    ASSERT(memoryStatesTensor != nullptr);
    nvinfer1::ITensor* attentionKeysTensor = nullptr;
    if (alignment->getAttentionKeySize() > 0)
    {
        nvinfer1::Dims attentionKeysDims{2, {gMaxInputSequenceLength, alignment->getAttentionKeySize()}};
        attentionKeysTensor = generatorNetwork->addInput(
            "attention_keys", gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, attentionKeysDims);
        ASSERT(attentionKeysTensor != nullptr);
    }
    nvinfer1::ITensor* inputAttentionTensor = nullptr;
    if (gFeedAttentionToInput)
    {
        nvinfer1::Dims inputAttentionDims{2, {gBeamWidth, attention->getAttentionSize()}};
        inputAttentionTensor = generatorNetwork->addInput(
            "input_attention", gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, inputAttentionDims);
        ASSERT(inputAttentionTensor != nullptr);
    }
    nvinfer1::Dims inputLikelihoodsDims{2, {gBeamWidth, 1}};
    auto inputLikelihoodsTensor
        = generatorNetwork->addInput("input_likelihoods", nvinfer1::DataType::kFLOAT, inputLikelihoodsDims);
    ASSERT(inputLikelihoodsTensor != nullptr);
    nvinfer1::Dims inputLikelihoodsReplicateIndicesDims{1, {gBeamWidth}};
    auto inputLikelihoodsReplicateIndicesTensor = generatorNetwork->addInput(
        "replicate_likelihoods_indices", nvinfer1::DataType::kINT32, inputLikelihoodsReplicateIndicesDims);
    ASSERT(inputLikelihoodsReplicateIndicesTensor != nullptr);

    // Add output embedder
    nvinfer1::ITensor* inputDecoderEmbeddedTensor;
    outputEmbedder->addToModel(generatorNetwork, inputDecoderDataTensor, &inputDecoderEmbeddedTensor);
    ASSERT(inputDecoderEmbeddedTensor != nullptr);

    // Add concatination of previous attention vector and embedded input for the decoder
    nvinfer1::ITensor* inputDecoderEmbeddedConcatinatedWithAttentionTensor{nullptr};
    if (gFeedAttentionToInput)
    {
        nvinfer1::ITensor* inputTensors[] = {inputDecoderEmbeddedTensor, inputAttentionTensor};
        auto concatLayer = generatorNetwork->addConcatenation(inputTensors, 2);
        ASSERT(concatLayer != nullptr);
        concatLayer->setName("Concatenate embedded input and attention");
        concatLayer->setAxis(1);
        inputDecoderEmbeddedConcatinatedWithAttentionTensor = concatLayer->getOutput(0);
        ASSERT(inputDecoderEmbeddedConcatinatedWithAttentionTensor != nullptr);
    }

    // Add decoder (single timestep)
    nvinfer1::ITensor* outputDecoderDataTensor{nullptr};
    std::vector<nvinfer1::ITensor*> decoderOutputStatesTensors(stateSizes.size());
    decoder->addToModel(generatorNetwork,
        gFeedAttentionToInput ? inputDecoderEmbeddedConcatinatedWithAttentionTensor : inputDecoderEmbeddedTensor,
        &decoderInputStatesTensors[0], &outputDecoderDataTensor, &decoderOutputStatesTensors[0]);
    for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "output_decoder_states_" << i;
        decoderOutputStatesTensors[i]->setName(ss.str().c_str());
        generatorNetwork->markOutput(*decoderOutputStatesTensors[i]);
        decoderOutputStatesTensors[i]->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    // Add alignment scores
    nvinfer1::ITensor* alignmentScoresTensor;
    alignment->addToModel(generatorNetwork,
        (alignment->getAttentionKeySize() > 0) ? attentionKeysTensor : memoryStatesTensor, outputDecoderDataTensor,
        &alignmentScoresTensor);

    // Add context
    nvinfer1::ITensor* contextTensor;
    context->addToModel(generatorNetwork, actualInputSequenceLengthsReplicatedTensor, memoryStatesTensor,
        alignmentScoresTensor, &contextTensor);

    // Add attention
    nvinfer1::ITensor* attentionTensor;
    attention->addToModel(generatorNetwork, outputDecoderDataTensor, contextTensor, &attentionTensor);
    if (gFeedAttentionToInput)
    {
        attentionTensor->setName("output_attention");
        generatorNetwork->markOutput(*attentionTensor);
        attentionTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    // Add projection
    nvinfer1::ITensor* logitsTensor;
    projection->addToModel(generatorNetwork, attentionTensor, &logitsTensor);

    // Replicate input likelihoods across all TopK options
    auto gatherLayer = generatorNetwork->addGather(*inputLikelihoodsTensor, *inputLikelihoodsReplicateIndicesTensor, 1);
    ASSERT(gatherLayer != nullptr);
    gatherLayer->setName("Replicate beam likelihoods");
    auto inputLikelihoodsReplicatedTensor = gatherLayer->getOutput(0);
    ASSERT(inputLikelihoodsReplicatedTensor != nullptr);

    // Add per-ray top-k options generation
    nvinfer1::ITensor* outputCombinedLikelihoodsTensor;
    nvinfer1::ITensor* outputRayOptionIndicesTensor;
    nvinfer1::ITensor* outputVocabularyIndicesTensor;
    likelihood->addToModel(generatorNetwork, gBeamWidth, logitsTensor, inputLikelihoodsReplicatedTensor,
        &outputCombinedLikelihoodsTensor, &outputRayOptionIndicesTensor, &outputVocabularyIndicesTensor);
    outputCombinedLikelihoodsTensor->setName("output_combined_likelihoods");
    generatorNetwork->markOutput(*outputCombinedLikelihoodsTensor);
    outputRayOptionIndicesTensor->setName("output_ray_option_indices");
    generatorNetwork->markOutput(*outputRayOptionIndicesTensor);
    outputRayOptionIndicesTensor->setType(nvinfer1::DataType::kINT32);
    outputVocabularyIndicesTensor->setName("output_vocabulary_indices");
    generatorNetwork->markOutput(*outputVocabularyIndicesTensor);
    outputVocabularyIndicesTensor->setType(nvinfer1::DataType::kINT32);

    samplesCommon::setDummyInt8DynamicRanges(generatorConfig, generatorNetwork);
    samplesCommon::enableDLA(generatorBuilder, generatorConfig, gUseDLACore);
    auto generatorPlan = generatorBuilder->buildSerializedNetwork(*generatorNetwork, *generatorConfig);
    ASSERT(generatorPlan != nullptr);
    auto runtime = createInferRuntime(sample::gLogger.getTRTLogger());
    ASSERT(runtime != nullptr);
    auto res = runtime->deserializeCudaEngine(generatorPlan->data(), generatorPlan->size());
    runtime->destroy();
    generatorPlan->destroy();
    generatorNetwork->destroy();
    generatorBuilder->destroy();
    generatorConfig->destroy();
    return res;
}

nvinfer1::ICudaEngine* getGeneratorShuffleEngine(
    const std::vector<nvinfer1::Dims>& decoderStateSizes, int32_t attentionSize)
{
    nvinfer1::IBuilder* shuffleBuilder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    ASSERT(shuffleBuilder != nullptr);
    nvinfer1::IBuilderConfig* shuffleConfig = shuffleBuilder->createBuilderConfig();
    shuffleBuilder->setMaxBatchSize(gMaxBatchSize);
    shuffleConfig->setMaxWorkspaceSize(gMaxWorkspaceSize);
    if (gFp16)
    {
        shuffleConfig->setFlag(BuilderFlag::kFP16);
    }
    if (gInt8)
    {
        shuffleConfig->setFlag(BuilderFlag::kINT8);
    }

    nvinfer1::INetworkDefinition* shuffleNetwork = shuffleBuilder->createNetworkV2(0);

    nvinfer1::Dims sourceRayIndicesDims{1, {gBeamWidth}};
    auto sourceRayIndicesTensor
        = shuffleNetwork->addInput("source_ray_indices", nvinfer1::DataType::kINT32, sourceRayIndicesDims);
    ASSERT(sourceRayIndicesTensor != nullptr);

    std::vector<nvinfer1::ITensor*> previousOutputDecoderStatesTensors(decoderStateSizes.size());
    for (int32_t i = 0; i < static_cast<int32_t>(decoderStateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "previous_output_decoder_states_" << i;
        nvinfer1::Dims statesDims;
        {
            statesDims.nbDims = decoderStateSizes[i].nbDims + 1;
            statesDims.d[0] = gBeamWidth;
            for (int32_t j = 0; j < decoderStateSizes[i].nbDims; ++j)
            {
                statesDims.d[j + 1] = decoderStateSizes[i].d[j];
            }
        }
        previousOutputDecoderStatesTensors[i] = shuffleNetwork->addInput(
            ss.str().c_str(), gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, statesDims);
        ASSERT(previousOutputDecoderStatesTensors[i] != nullptr);
    }

    nvinfer1::ITensor* previousOutputAttentionTensor = nullptr;
    if (gFeedAttentionToInput)
    {
        nvinfer1::Dims previousOutputAttentionDims{2, {gBeamWidth, attentionSize}};
        previousOutputAttentionTensor = shuffleNetwork->addInput("previous_output_attention",
            gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT, previousOutputAttentionDims);
        ASSERT(previousOutputAttentionTensor != nullptr);
    }

    for (int32_t i = 0; i < static_cast<int32_t>(decoderStateSizes.size()); ++i)
    {
        auto gatherLayer
            = shuffleNetwork->addGather(*previousOutputDecoderStatesTensors[i], *sourceRayIndicesTensor, 0);
        ASSERT(gatherLayer != nullptr);
        {
            std::stringstream ss;
            ss << "Shuffle decoder states " << i;
            gatherLayer->setName(ss.str().c_str());
        }
        auto inputDecoderHiddenStatesTensor = gatherLayer->getOutput(0);
        ASSERT(inputDecoderHiddenStatesTensor != nullptr);
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        inputDecoderHiddenStatesTensor->setName(ss.str().c_str());
        shuffleNetwork->markOutput(*inputDecoderHiddenStatesTensor);
        inputDecoderHiddenStatesTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    if (gFeedAttentionToInput)
    {
        auto gatherLayer = shuffleNetwork->addGather(*previousOutputAttentionTensor, *sourceRayIndicesTensor, 0);
        ASSERT(gatherLayer != nullptr);
        gatherLayer->setName("Shuffle attention");
        auto inputAttentionTensor = gatherLayer->getOutput(0);
        ASSERT(inputAttentionTensor != nullptr);
        inputAttentionTensor->setName("input_attention");
        shuffleNetwork->markOutput(*inputAttentionTensor);
        inputAttentionTensor->setType(gFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
    }

    samplesCommon::setDummyInt8DynamicRanges(shuffleConfig, shuffleNetwork);
    samplesCommon::enableDLA(shuffleBuilder, shuffleConfig, gUseDLACore);
    auto shufflePlan = shuffleBuilder->buildSerializedNetwork(*shuffleNetwork, *shuffleConfig);
    ASSERT(shufflePlan != nullptr);
    auto runtime = createInferRuntime(sample::gLogger.getTRTLogger());
    ASSERT(runtime != nullptr);
    auto res = runtime->deserializeCudaEngine(shufflePlan->data(), shufflePlan->size());
    runtime->destroy();
    shufflePlan->destroy();
    shuffleNetwork->destroy();
    shuffleBuilder->destroy();
    shuffleConfig->destroy();
    return res;
}

//! \brief assign device pointers to the correct location in the bindings vector.
//!
//! Given a binding map which stores the name to device pointer mapping, in
//! a generic fashion, insert into the bindings vector the device pointer
//! at the correct index for a given engine.
void processBindings(
    std::vector<void*>& bindings, std::unordered_map<std::string, void*>& bindingMap, nvinfer1::ICudaEngine* engine)
{
    for (auto& a : bindingMap)
    {
        auto bindIdx = engine->getBindingIndex(a.first.c_str());
        ASSERT(bindIdx >= 0 && bindIdx < engine->getNbBindings());
        bindings[bindIdx] = a.second;
    }
}

int32_t main(int32_t argc, char** argv)
{
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    samplesCommon::Args args;
    bool argsOK = parseNMTArgs(args, argc, argv);
    if (args.help)
    {
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (gVerbose)
    {
        sample::setReportableSeverity(ILogger::Severity::kVERBOSE);
    }

    // Set up output vocabulary
    {
        std::string vocabularyFilePath = gOutputVocabularyFileName;
        std::ifstream vocabStream(locateNMTFile(vocabularyFilePath));
        if (!vocabStream.good())
        {
            sample::gLogError << "Cannot open file " << vocabularyFilePath << std::endl;
            return sample::gLogger.reportFail(sampleTest);
        }
        vocabStream >> *gOutputVocabulary;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto outputSequenceProperties = getOutputSequenceProperties();
    auto dataReader = getDataReader();
    auto inputEmbedder = getInputEmbedder();
    auto outputEmbedder = getOutputEmbedder();
    auto encoder = getEncoder();
    auto decoder = getDecoder();
    auto alignment = getAlignment();
    auto context = getContext();
    auto attention = getAttention();
    auto projection = getProjection();
    auto likelihood = getLikelihood();
    auto searchPolicy
        = getSearchPolicy(outputSequenceProperties->getEndSequenceId(), likelihood->getLikelihoodCombinationOperator());
    auto dataWriter = getDataWriter();

    if (gPrintComponentInfo)
    {
        sample::gLogInfo << "Component Info:" << std::endl;
        sample::gLogInfo << "- Data Reader: " << dataReader->getInfo() << std::endl;
        sample::gLogInfo << "- Input Embedder: " << inputEmbedder->getInfo() << std::endl;
        sample::gLogInfo << "- Output Embedder: " << outputEmbedder->getInfo() << std::endl;
        sample::gLogInfo << "- Encoder: " << encoder->getInfo() << std::endl;
        sample::gLogInfo << "- Decoder: " << decoder->getInfo() << std::endl;
        sample::gLogInfo << "- Alignment: " << alignment->getInfo() << std::endl;
        sample::gLogInfo << "- Context: " << context->getInfo() << std::endl;
        sample::gLogInfo << "- Attention: " << attention->getInfo() << std::endl;
        sample::gLogInfo << "- Projection: " << projection->getInfo() << std::endl;
        sample::gLogInfo << "- Likelihood: " << likelihood->getInfo() << std::endl;
        sample::gLogInfo << "- Search Policy: " << searchPolicy->getInfo() << std::endl;
        sample::gLogInfo << "- Data Writer: " << dataWriter->getInfo() << std::endl;
        sample::gLogInfo << "End of Component Info" << std::endl;
    }

    std::vector<nvinfer1::Dims> stateSizes = decoder->getStateSizes();

    // A number of consistency checks between components
    ASSERT(alignment->getSourceStatesSize() == encoder->getMemoryStatesSize());
    {
        std::vector<nvinfer1::Dims> encoderStateSizes = encoder->getStateSizes();
        ASSERT(stateSizes.size() == encoderStateSizes.size());
        for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
            ASSERT(nmtSample::getVolume(stateSizes[i]) == nmtSample::getVolume(encoderStateSizes[i]));
    }
    ASSERT(projection->getOutputSize() == outputEmbedder->getInputDimensionSize());

    auto inputOriginalHostBuffer
        = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize * gMaxInputSequenceLength);
    auto inputHostBuffer
        = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize * gMaxInputSequenceLength);
    auto inputOriginalSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize);
    auto inputSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize);
    auto maxOutputSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize);
    auto outputSequenceLengthsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize);
    auto outputCombinedLikelihoodHostBuffer
        = std::make_shared<nmtSample::PinnedHostBuffer<float>>(gMaxBatchSize * gBeamWidth);
    auto outputVocabularyIndicesHostBuffer
        = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    auto outputRayOptionIndicesHostBuffer
        = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    auto sourceRayIndicesHostBuffer
        = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    auto sourceLikelihoodsHostBuffer = std::make_shared<nmtSample::PinnedHostBuffer<float>>(gMaxBatchSize * gBeamWidth);

    // Allocated buffers on GPU to be used as inputs and outputs for TenorRT
    auto inputEncoderDeviceBuffer
        = std::make_shared<nmtSample::DeviceBuffer<int32_t>>(gMaxBatchSize * gMaxInputSequenceLength);
    auto inputSequenceLengthsDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int32_t>>(gMaxBatchSize);
    auto inputSequenceLengthsReplicatedDeviceBuffer
        = std::make_shared<nmtSample::DeviceBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    auto memoryStatesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(
        gMaxBatchSize * gMaxInputSequenceLength * encoder->getMemoryStatesSize());
    auto attentionKeysDeviceBuffer
        = std::make_shared<nmtSample::DeviceBuffer<float>>((alignment->getAttentionKeySize() > 0)
                ? gMaxBatchSize * gMaxInputSequenceLength * alignment->getAttentionKeySize()
                : 0);
    std::vector<nmtSample::DeviceBuffer<float>::ptr> encoderStatesLastTimestepDeviceBuffers;
    for (auto stateSize : stateSizes)
        encoderStatesLastTimestepDeviceBuffers.push_back(
            std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * nmtSample::getVolume(stateSize)));
    std::vector<nmtSample::DeviceBuffer<float>::ptr> inputDecoderStatesDeviceBuffers;
    for (auto stateSize : stateSizes)
        inputDecoderStatesDeviceBuffers.push_back(std::make_shared<nmtSample::DeviceBuffer<float>>(
            gMaxBatchSize * gBeamWidth * nmtSample::getVolume(stateSize)));
    std::vector<nmtSample::DeviceBuffer<float>::ptr> outputDecoderStatesDeviceBuffers;
    for (auto stateSize : stateSizes)
        outputDecoderStatesDeviceBuffers.push_back(std::make_shared<nmtSample::DeviceBuffer<float>>(
            gMaxBatchSize * gBeamWidth * nmtSample::getVolume(stateSize)));
    auto inputAttentionDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(
        gFeedAttentionToInput ? gMaxBatchSize * gBeamWidth * attention->getAttentionSize() : 0);
    auto outputAttentionDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(
        gFeedAttentionToInput ? gMaxBatchSize * gBeamWidth * attention->getAttentionSize() : 0);
    auto outputCombinedLikelihoodDeviceBuffer
        = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth);
    auto outputRayOptionIndicesDeviceBuffer
        = std::make_shared<nmtSample::DeviceBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    auto sourceRayIndicesDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    auto inputDecoderDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    auto inputLikelihoodsDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth);

    std::vector<nmtSample::DeviceBuffer<float>::ptr> zeroInputEncoderStatesDeviceBuffers;
    for (auto stateSize : stateSizes)
    {
        auto buf = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * nmtSample::getVolume(stateSize));
        CUDA_CHECK(cudaMemsetAsync(*buf, 0, gMaxBatchSize * nmtSample::getVolume(stateSize) * sizeof(float), stream));
        zeroInputEncoderStatesDeviceBuffers.push_back(buf);
    }

    std::vector<nmtSample::DeviceBuffer<float>::ptr> zeroInputDecoderStatesDeviceBuffers;
    for (auto stateSize : stateSizes)
    {
        auto buf = std::make_shared<nmtSample::DeviceBuffer<float>>(
            gMaxBatchSize * gBeamWidth * nmtSample::getVolume(stateSize));
        CUDA_CHECK(cudaMemsetAsync(
            *buf, 0, gMaxBatchSize * gBeamWidth * nmtSample::getVolume(stateSize) * sizeof(float), stream));
        zeroInputDecoderStatesDeviceBuffers.push_back(buf);
    }

    auto zeroInputAttentionDeviceBuffer = std::make_shared<nmtSample::DeviceBuffer<float>>(
        gFeedAttentionToInput ? gMaxBatchSize * gBeamWidth * attention->getAttentionSize() : 0);
    if (gFeedAttentionToInput)
    {
        CUDA_CHECK(cudaMemsetAsync(*zeroInputAttentionDeviceBuffer, 0,
            gMaxBatchSize * gBeamWidth * attention->getAttentionSize() * sizeof(float), stream));
    }
    auto startSeqInputDecoderDeviceBuffer
        = std::make_shared<nmtSample::DeviceBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    {
        auto startSeqInputDecoderHostBuffer
            = std::make_shared<nmtSample::PinnedHostBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
        std::fill_n((int32_t*) *startSeqInputDecoderHostBuffer, gMaxBatchSize * gBeamWidth,
            outputSequenceProperties->getStartSequenceId());
        CUDA_CHECK(cudaMemcpyAsync(*startSeqInputDecoderDeviceBuffer, *startSeqInputDecoderHostBuffer,
            gMaxBatchSize * gBeamWidth * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto zeroInitializeDecoderIndicesDeviceBuffer
        = std::make_shared<nmtSample::DeviceBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    CUDA_CHECK(cudaMemsetAsync(
        *zeroInitializeDecoderIndicesDeviceBuffer, 0, gMaxBatchSize * gBeamWidth * sizeof(int32_t), stream));
    auto initialInputLikelihoodsDeviceBuffer
        = std::make_shared<nmtSample::DeviceBuffer<float>>(gMaxBatchSize * gBeamWidth);
    {
        auto likelihoodCombinationOperator = likelihood->getLikelihoodCombinationOperator();
        auto initialInputLikelihoodsHostBuffer
            = std::make_shared<nmtSample::PinnedHostBuffer<float>>(gMaxBatchSize * gBeamWidth);
        for (int32_t sampleId = 0; sampleId < gMaxBatchSize; ++sampleId)
        {
            (*initialInputLikelihoodsHostBuffer)[sampleId * gBeamWidth] = likelihoodCombinationOperator->init();
            for (int32_t rayId = 1; rayId < gBeamWidth; ++rayId)
                (*initialInputLikelihoodsHostBuffer)[sampleId * gBeamWidth + rayId]
                    = likelihoodCombinationOperator->smallerThanMinimalLikelihood();
        }
        CUDA_CHECK(cudaMemcpyAsync(*initialInputLikelihoodsDeviceBuffer, *initialInputLikelihoodsHostBuffer,
            gMaxBatchSize * gBeamWidth * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    auto zeroReplicateLikelihoodsIndicesDeviceBuffer
        = std::make_shared<nmtSample::DeviceBuffer<int32_t>>(gMaxBatchSize * gBeamWidth);
    CUDA_CHECK(cudaMemsetAsync(
        *zeroReplicateLikelihoodsIndicesDeviceBuffer, 0, gMaxBatchSize * gBeamWidth * sizeof(int32_t), stream));

    // Create TensorRT engines
    auto encoderEngine = std::unique_ptr<nvinfer1::ICudaEngine>(getEncoderEngine(inputEmbedder, encoder, alignment));
    auto generatorEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        getGeneratorEngine(outputEmbedder, decoder, alignment, context, attention, projection, likelihood));
    auto generatorShuffleEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
        getGeneratorShuffleEngine(decoder->getStateSizes(), attention->getAttentionSize()));

    // Setup TensorRT bindings
    std::vector<void*> encoderBindings(encoderEngine->getNbBindings());
    std::unordered_map<std::string, void*> encBindingMap;
    encBindingMap["input_encoder_data"] = *inputEncoderDeviceBuffer;
    encBindingMap["actual_input_sequence_lengths"] = *inputSequenceLengthsDeviceBuffer;
    encBindingMap["actual_input_sequence_lengths_with_index_dim"] = *inputSequenceLengthsDeviceBuffer;
    encBindingMap["actual_input_sequence_lengths_replicated"] = *inputSequenceLengthsReplicatedDeviceBuffer;
    encBindingMap["initialize_decoder_indices"] = *zeroInitializeDecoderIndicesDeviceBuffer;
    for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_encoder_states_" << i;
        encBindingMap[ss.str()] = *zeroInputEncoderStatesDeviceBuffers[i];
    }
    encBindingMap["memory_states"] = *memoryStatesDeviceBuffer;
    if (alignment->getAttentionKeySize() > 0)
    {
        encBindingMap["attention_keys"] = *attentionKeysDeviceBuffer;
    }
    {
        for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
        {
            std::stringstream ss;
            ss << "input_decoder_states_" << i;
            encBindingMap[ss.str()] = *inputDecoderStatesDeviceBuffers[i];
        }
    }
    processBindings(encoderBindings, encBindingMap, encoderEngine.get());

    std::vector<void*> generatorBindings(generatorEngine->getNbBindings());
    std::unordered_map<std::string, void*> genBindingMap;
    genBindingMap["input_decoder_data"] = *inputDecoderDeviceBuffer;
    for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        genBindingMap[ss.str()] = *inputDecoderStatesDeviceBuffers[i];
    }
    genBindingMap["actual_input_sequence_lengths_replicated"] = *inputSequenceLengthsReplicatedDeviceBuffer;
    genBindingMap["memory_states"] = *memoryStatesDeviceBuffer;
    if (alignment->getAttentionKeySize() > 0)
    {
        genBindingMap["attention_keys"] = *attentionKeysDeviceBuffer;
    }
    for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "output_decoder_states_" << i;
        genBindingMap[ss.str()] = *outputDecoderStatesDeviceBuffers[i];
    }
    genBindingMap["output_combined_likelihoods"] = *outputCombinedLikelihoodDeviceBuffer;
    genBindingMap["output_vocabulary_indices"] = *inputDecoderDeviceBuffer;
    genBindingMap["output_ray_option_indices"] = *outputRayOptionIndicesDeviceBuffer;
    if (gFeedAttentionToInput)
    {
        genBindingMap["input_attention"] = *inputAttentionDeviceBuffer;
        genBindingMap["output_attention"] = *outputAttentionDeviceBuffer;
    }
    genBindingMap["input_likelihoods"] = *inputLikelihoodsDeviceBuffer;
    genBindingMap["replicate_likelihoods_indices"] = *zeroReplicateLikelihoodsIndicesDeviceBuffer;
    processBindings(generatorBindings, genBindingMap, generatorEngine.get());

    std::vector<void*> generatorBindingsFirstStep = generatorBindings;
    std::unordered_map<std::string, void*> genBindingFirstStepMap;
    genBindingFirstStepMap["input_decoder_data"] = *startSeqInputDecoderDeviceBuffer;
    if (gFeedAttentionToInput)
    {
        genBindingFirstStepMap["input_attention"] = *zeroInputAttentionDeviceBuffer;
    }
    genBindingFirstStepMap["input_likelihoods"] = *initialInputLikelihoodsDeviceBuffer;
    processBindings(generatorBindingsFirstStep, genBindingFirstStepMap, generatorEngine.get());

    std::vector<void*> generatorShuffleBindings(generatorShuffleEngine->getNbBindings());
    std::unordered_map<std::string, void*> genShuffleBindingMap;
    genShuffleBindingMap["source_ray_indices"] = *sourceRayIndicesDeviceBuffer;
    for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "previous_output_decoder_states_" << i;
        genShuffleBindingMap[ss.str()] = *outputDecoderStatesDeviceBuffers[i];
    }
    for (int32_t i = 0; i < static_cast<int32_t>(stateSizes.size()); ++i)
    {
        std::stringstream ss;
        ss << "input_decoder_states_" << i;
        genShuffleBindingMap[ss.str()] = *inputDecoderStatesDeviceBuffers[i];
    }
    if (gFeedAttentionToInput)
    {
        genShuffleBindingMap["previous_output_attention"] = *outputAttentionDeviceBuffer;
        genShuffleBindingMap["input_attention"] = *inputAttentionDeviceBuffer;
    }
    processBindings(generatorShuffleBindings, genShuffleBindingMap, generatorShuffleEngine.get());

    // Create Tensor RT contexts
    auto encoderContext = std::unique_ptr<nvinfer1::IExecutionContext>(encoderEngine->createExecutionContext());
    auto generatorContext = std::unique_ptr<nvinfer1::IExecutionContext>(generatorEngine->createExecutionContext());
    auto generatorShuffleContext
        = std::unique_ptr<nvinfer1::IExecutionContext>(generatorShuffleEngine->createExecutionContext());

    std::vector<SimpleProfiler> profilers;
    if (gEnableProfiling)
    {
        profilers.push_back(SimpleProfiler("Host"));
        profilers.push_back(SimpleProfiler("Encoder"));
        profilers.push_back(SimpleProfiler("Decoder"));
        profilers.push_back(SimpleProfiler("Beam shuffle"));
        encoderContext->setProfiler(&profilers[1]);
        generatorContext->setProfiler(&profilers[2]);
        generatorShuffleContext->setProfiler(&profilers[3]);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    dataWriter->initialize();

    std::vector<int32_t> outputHostBuffer;
    auto startDataRead = std::chrono::high_resolution_clock::now();
    int32_t inputSamplesRead = dataReader->read(
        gMaxBatchSize, gMaxInputSequenceLength, *inputOriginalHostBuffer, *inputOriginalSequenceLengthsHostBuffer);
    if (gEnableProfiling)
        profilers[0].reportLayerTime("Data Read",
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startDataRead)
                .count());
    // Outer loop over batches of samples
    auto startLatency = std::chrono::high_resolution_clock::now();
    int32_t batchCount = 0;
    while (inputSamplesRead > 0)
    {
        ++batchCount;

        // Sort input sequences in the batch in the order of decreasing length
        // The idea is that shorter input sequences gets translated faster so we can reduce batch size quickly for the
        // generator
        auto startBatchSort = std::chrono::high_resolution_clock::now();
        std::vector<int32_t> samplePositions(inputSamplesRead);
        {
            std::vector<std::pair<int32_t, int32_t>> sequenceSampleIdAndLength(inputSamplesRead);
            for (int32_t sampleId = 0; sampleId < inputSamplesRead; ++sampleId)
                sequenceSampleIdAndLength[sampleId]
                    = std::make_pair(sampleId, ((const int32_t*) *inputOriginalSequenceLengthsHostBuffer)[sampleId]);
            std::sort(sequenceSampleIdAndLength.begin(), sequenceSampleIdAndLength.end(),
                [](const std::pair<int32_t, int32_t>& a, const std::pair<int32_t, int32_t>& b) -> bool {
                    return a.second > b.second;
                });
            for (int32_t position = 0; position < inputSamplesRead; ++position)
            {
                int32_t sampleId = sequenceSampleIdAndLength[position].first;
                ((int32_t*) *inputSequenceLengthsHostBuffer)[position]
                    = ((const int32_t*) *inputOriginalSequenceLengthsHostBuffer)[sampleId];
                std::copy_n(((const int32_t*) *inputOriginalHostBuffer) + sampleId * gMaxInputSequenceLength,
                    gMaxInputSequenceLength, ((int32_t*) *inputHostBuffer) + position * gMaxInputSequenceLength);
                samplePositions[sampleId] = position;
            }
        }
        if (gEnableProfiling)
            profilers[0].reportLayerTime("Intra-batch Sort",
                std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startBatchSort)
                    .count());

        CUDA_CHECK(cudaMemcpyAsync(*inputEncoderDeviceBuffer, *inputHostBuffer,
            inputSamplesRead * gMaxInputSequenceLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(*inputSequenceLengthsDeviceBuffer, *inputSequenceLengthsHostBuffer,
            inputSamplesRead * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        // Overlap host and device: Read data for the next batch while encode for this one is running
        std::future<int32_t> nextInputSamplesReadFuture = std::async(std::launch::async, [&]() {
            return dataReader->read(gMaxBatchSize, gMaxInputSequenceLength, *inputOriginalHostBuffer,
                *inputOriginalSequenceLengthsHostBuffer);
        });

        if (!encoderContext->enqueue(inputSamplesRead, &encoderBindings[0], stream, nullptr))
        {
            sample::gLogError << "Error in encoder context enqueue" << std::endl;
            return sample::gLogger.reportTest(sampleTest, false);
        }

        // Limit output sequences length to input_sequence_length * 2
        std::transform((const int32_t*) *inputSequenceLengthsHostBuffer,
            (const int32_t*) *inputSequenceLengthsHostBuffer + inputSamplesRead,
            (int32_t*) *maxOutputSequenceLengthsHostBuffer, [](int32_t i) {
                int32_t r = i * 2;
                if (gMaxOutputSequenceLength >= 0)
                    r = std::min(r, gMaxOutputSequenceLength);
                return r;
            });
        searchPolicy->initialize(inputSamplesRead, *maxOutputSequenceLengthsHostBuffer);
        int32_t batchMaxOutputSequenceLength = *std::max_element((int32_t*) *maxOutputSequenceLengthsHostBuffer,
            (int32_t*) *maxOutputSequenceLengthsHostBuffer + inputSamplesRead);
        outputHostBuffer.resize(gMaxBatchSize * batchMaxOutputSequenceLength);

        // Inner loop over generator timesteps
        int32_t validSampleCount = searchPolicy->getTailWithNoWorkRemaining();
        for (int32_t outputTimestep = 0; (outputTimestep < batchMaxOutputSequenceLength) && (validSampleCount > 0);
             ++outputTimestep)
        {
            // Generator initialization and beam shuffling
            if (outputTimestep == 0)
            {
                if (!generatorContext->enqueue(validSampleCount, &generatorBindingsFirstStep[0], stream, nullptr))
                {
                    sample::gLogError << "Error in generator context enqueue step" << outputTimestep << std::endl;
                    return sample::gLogger.reportTest(sampleTest, false);
                }
            }
            else
            {
                if (!generatorShuffleContext->enqueue(validSampleCount, &generatorShuffleBindings[0], stream, nullptr))
                {
                    sample::gLogError << "Error in generator shuffle context enqueue step " << outputTimestep
                                      << std::endl;
                    return sample::gLogger.reportTest(sampleTest, false);
                }
                if (!generatorContext->enqueue(validSampleCount, &generatorBindings[0], stream, nullptr))
                {
                    sample::gLogError << "Error in generator context enqueue step" << outputTimestep << std::endl;
                    return sample::gLogger.reportTest(sampleTest, false);
                }
            }

            CUDA_CHECK(cudaMemcpyAsync(*outputCombinedLikelihoodHostBuffer, *outputCombinedLikelihoodDeviceBuffer,
                validSampleCount * gBeamWidth * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(*outputVocabularyIndicesHostBuffer, *inputDecoderDeviceBuffer,
                validSampleCount * gBeamWidth * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(*outputRayOptionIndicesHostBuffer, *outputRayOptionIndicesDeviceBuffer,
                validSampleCount * gBeamWidth * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));

            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto startBeamSearch = std::chrono::high_resolution_clock::now();
            searchPolicy->processTimestep(validSampleCount, *outputCombinedLikelihoodHostBuffer,
                *outputVocabularyIndicesHostBuffer, *outputRayOptionIndicesHostBuffer, *sourceRayIndicesHostBuffer,
                *sourceLikelihoodsHostBuffer);
            if (gEnableProfiling)
                profilers[0].reportLayerTime("Beam Search",
                    std::chrono::duration<float, std::milli>(
                        std::chrono::high_resolution_clock::now() - startBeamSearch)
                        .count());

            CUDA_CHECK(cudaMemcpyAsync(*sourceRayIndicesDeviceBuffer, *sourceRayIndicesHostBuffer,
                validSampleCount * gBeamWidth * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(*inputLikelihoodsDeviceBuffer, *sourceLikelihoodsHostBuffer,
                validSampleCount * gBeamWidth * sizeof(float), cudaMemcpyHostToDevice, stream));

            validSampleCount = searchPolicy->getTailWithNoWorkRemaining();
        } // for(int32_t outputTimestep

        auto startBacktrack = std::chrono::high_resolution_clock::now();
        searchPolicy->readGeneratedResult(
            inputSamplesRead, batchMaxOutputSequenceLength, &outputHostBuffer[0], *outputSequenceLengthsHostBuffer);
        if (gEnableProfiling)
            profilers[0].reportLayerTime("Read Result",
                std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startBacktrack)
                    .count());

        auto startDataWrite = std::chrono::high_resolution_clock::now();
        for (int32_t sampleId = 0; sampleId < inputSamplesRead; ++sampleId)
        {
            int32_t position = samplePositions[sampleId];
            dataWriter->write(&outputHostBuffer[0] + position * batchMaxOutputSequenceLength,
                ((const int32_t*) *outputSequenceLengthsHostBuffer)[position],
                ((const int32_t*) *inputSequenceLengthsHostBuffer)[position]);
        }
        if (gEnableProfiling)
            profilers[0].reportLayerTime("Data Write",
                std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startDataWrite)
                    .count());

        auto startDataRead = std::chrono::high_resolution_clock::now();
        inputSamplesRead = nextInputSamplesReadFuture.get();
        if (gEnableProfiling)
            profilers[0].reportLayerTime("Data Read",
                std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startDataRead)
                    .count());
    }
    float totalLatency
        = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - startLatency).count();

    dataWriter->finalize();
    float score
        = gDataWriterStr == "bleu" ? static_cast<nmtSample::BLEUScoreWriter*>(dataWriter.get())->getScore() : -1.0F;

    if (gDataWriterStr == "benchmark")
    {
        sample::gLogInfo << "Average latency (without data read) = " << totalLatency / static_cast<float>(batchCount)
                         << " ms" << std::endl;
    }

    if (gEnableProfiling)
    {
        if (gAggregateProfiling)
        {
            SimpleProfiler aggregateProfiler("Aggregate", profilers);
            sample::gLogInfo << aggregateProfiler << std::endl;
        }
        else
        {
            for (const auto& profiler : profilers)
            {
                sample::gLogInfo << profiler << std::endl;
            }
        }
    }

    cudaStreamDestroy(stream);

    bool pass = gDataWriterStr != "bleu" || score >= 25.0F;

    return sample::gLogger.reportTest(sampleTest, pass);
}
