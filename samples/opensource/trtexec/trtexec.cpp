/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace nvuffparser;
using namespace nvonnxparser;

const std::string gSampleName = "TensorRT.trtexec";

struct Params
{
    std::string deployFile{};
    std::string modelFile{};
    std::string engine{};
    std::string saveEngine{};
    std::string loadEngine{};
    std::string calibrationCache{"CalibrationTable"};
    std::string outputCalibrationCache{"CalibrationTable"};
    std::string uffFile{};
    std::string onnxModelFile{};
    std::vector<std::string> inputs{};
    std::vector<std::string> outputs{};
    std::vector<std::pair<std::string, Dims3>> uffInputs{};
    int device{0};
    int batchSize{1};
    int workspaceSize{16};
    int iterations{10};
    int avgRuns{10};
    int useDLACore{-1};
    bool safeMode{false};
    bool fp16{false};
    bool int8{false};
    bool verbose{false};
    bool allowGPUFallback{false};
    float pct{99};
    bool useSpinWait{false};
    bool dumpOutput{false};
    bool dumpLayerTime{false};
    bool help{false};
    std::vector<std::string> plugins;
} gParams;

inline int volume(Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

std::map<std::string, Dims3> gInputDimensions;

std::vector<std::string> split(const std::string& s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        res.push_back(item);
    }
    return res;
}

float percentile(float percentage, std::vector<float>& times)
{
    int all = static_cast<int>(times.size());
    int exclude = static_cast<int>((1 - percentage / 100) * all);
    if (0 <= exclude && exclude <= all)
    {
        std::sort(times.begin(), times.end());
        return times[all == exclude ? 0 : all - 1 - exclude];
    }
    return std::numeric_limits<float>::infinity();
}

class RndInt8Calibrator : public IInt8EntropyCalibrator2
{
public:
    RndInt8Calibrator(int totalSamples, std::string cacheFile, std::string outputCacheFile)
        : mTotalSamples(totalSamples)
        , mCurrentSample(0)
        , mCacheFile(cacheFile)
	, mOutputCacheFile(outputCacheFile)
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        for (auto& elem : gInputDimensions)
        {
            int elemCount = volume(elem.second);

            std::vector<float> rnd_data(elemCount);
            for (auto& val : rnd_data)
                val = distribution(generator);

            void* data;
            CHECK(cudaMalloc(&data, elemCount * sizeof(float)));
            CHECK(cudaMemcpy(data, &rnd_data[0], elemCount * sizeof(float), cudaMemcpyHostToDevice));

            mInputDeviceBuffers.insert(std::make_pair(elem.first, data));
        }
    }

    ~RndInt8Calibrator()
    {
        for (auto& elem : mInputDeviceBuffers)
            CHECK(cudaFree(elem.second));
    }

    int getBatchSize() const override
    {
        return 1;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (mCurrentSample >= mTotalSamples)
            return false;

        for (int i = 0; i < nbBindings; ++i)
            bindings[i] = mInputDeviceBuffers[names[i]];

        ++mCurrentSample;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCacheFile, std::ios::binary);
        input >> std::noskipws;
        if (input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(mOutputCacheFile, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    int mTotalSamples;
    int mCurrentSample;
    std::string mCacheFile;
    std::string mOutputCacheFile;
    std::map<std::string, void*> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
};

void configureBuilder(IBuilder* builder, RndInt8Calibrator& calibrator)
{
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize(static_cast<size_t>(gParams.workspaceSize) << 20);
    builder->setFp16Mode(gParams.fp16);
    if (gParams.int8)
    {
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(&calibrator);
    }

    if (gParams.safeMode)
    {
        builder->setEngineCapability(
            gParams.useDLACore >= 0 ? EngineCapability::kSAFE_DLA : EngineCapability::kSAFE_GPU);
    }
}

ICudaEngine* caffeToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    if (builder == nullptr)
    {
        return nullptr;
    }

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(gParams.deployFile.c_str(),
        gParams.modelFile.empty() ? 0 : gParams.modelFile.c_str(), *network, DataType::kFLOAT);

    if (!blobNameToTensor)
    {
        return nullptr;
    }

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gParams.inputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
        gLogInfo << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
                 << dims.d[2] << std::endl;
    }

    // specify which tensors are outputs
    for (auto& s : gParams.outputs)
    {
        if (blobNameToTensor->find(s.c_str()) == nullptr)
        {
            gLogError << "could not find output blob " << s << std::endl;
            return nullptr;
        }
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getOutput(i)->getDimensions());
        gLogInfo << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x" << dims.d[1] << "x"
                 << dims.d[2] << std::endl;
    }

    // Build the engine
    RndInt8Calibrator calibrator(1, gParams.calibrationCache, gParams.outputCalibrationCache);
    configureBuilder(builder, calibrator);

    samplesCommon::enableDLA(builder, gParams.useDLACore, gParams.allowGPUFallback);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
    {
        gLogError << "could not build engine" << std::endl;
    }

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

ICudaEngine* uffToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    if (builder == nullptr)
    {
        return nullptr;
    }

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    IUffParser* parser = createUffParser();

    // specify which tensors are outputs
    for (auto& s : gParams.outputs)
    {
        if (!parser->registerOutput(s.c_str()))
        {
            gLogError << "Failed to register output " << s << std::endl;
            return nullptr;
        }
    }

    // specify which tensors are inputs (and their dimensions)
    for (auto& s : gParams.uffInputs)
    {
        if (!parser->registerInput(s.first.c_str(), s.second, UffInputOrder::kNCHW))
        {
            gLogError << "Failed to register input " << s.first << std::endl;
            return nullptr;
        }
    }

    if (!parser->parse(gParams.uffFile.c_str(), *network))
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gParams.inputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
    }

    // Build the engine
    RndInt8Calibrator calibrator(1, gParams.calibrationCache, gParams.outputCalibrationCache);
    configureBuilder(builder, calibrator);

    samplesCommon::enableDLA(builder, gParams.useDLACore);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        gLogError << "could not build engine" << std::endl;

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

ICudaEngine* onnxToTRTModel()
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    if (builder == nullptr)
    {
        return nullptr;
    }
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    // parse the onnx model to populate the network, then set the outputs
    IParser* parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(gParams.onnxModelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    {
        gLogError << "failed to parse onnx file" << std::endl;
        return nullptr;
    }

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        Dims3 dims = static_cast<Dims3&&>(network->getInput(i)->getDimensions());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
    }

    // Build the engine
    RndInt8Calibrator calibrator(1, gParams.calibrationCache, gParams.outputCalibrationCache);
    configureBuilder(builder, calibrator);

    samplesCommon::enableDLA(builder, gParams.useDLACore);

    ICudaEngine* engine = builder->buildCudaEngine(*network);

    if (engine == nullptr)
    {
        gLogError << "could not build engine" << std::endl;
    }

    parser->destroy();
    network->destroy();
    builder->destroy();
    return engine;
}

void doInference(ICudaEngine& engine)
{
    IExecutionContext* context = engine.createExecutionContext();

    // Dump inferencing time per layer basis
    SimpleProfiler profiler("Layer time");
    if (gParams.dumpLayerTime)
    {
        context->setProfiler(&profiler);
    }

    // Use an aliasing shared_ptr since we don't want engine to be deleted when bufferManager goes out of scope.
    std::shared_ptr<ICudaEngine> emptyPtr{};
    std::shared_ptr<ICudaEngine> aliasPtr(emptyPtr, &engine);
    samplesCommon::BufferManager bufferManager(aliasPtr, gParams.batchSize);
    std::vector<void*> buffers = bufferManager.getDeviceBindings();

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    unsigned int cudaEventFlags = gParams.useSpinWait ? cudaEventDefault : cudaEventBlockingSync;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventFlags));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventFlags));

    std::vector<float> times(gParams.avgRuns);
    for (int j = 0; j < gParams.iterations; j++)
    {
        float totalGpu{0};  // GPU timer
        float totalHost{0}; // Host timer

        for (int i = 0; i < gParams.avgRuns; i++)
        {
            auto tStart = std::chrono::high_resolution_clock::now();
            cudaEventRecord(start, stream);
            context->enqueue(gParams.batchSize, &buffers[0], stream, nullptr);
            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);

            auto tEnd = std::chrono::high_resolution_clock::now();
            totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
            float ms;
            cudaEventElapsedTime(&ms, start, end);
            times[i] = ms;
            totalGpu += ms;
        }

        totalGpu /= gParams.avgRuns;
        totalHost /= gParams.avgRuns;
        gLogInfo << "Average over " << gParams.avgRuns << " runs is " << totalGpu << " ms (host walltime is "
                 << totalHost << " ms, " << static_cast<int>(gParams.pct) << "\% percentile time is "
                 << percentile(gParams.pct, times) << ")." << std::endl;
    }

    if (gParams.dumpOutput)
    {
        bufferManager.copyOutputToHost();
        int nbBindings = engine.getNbBindings();
        for (int i = 0; i < nbBindings; i++)
        {
            if (!engine.bindingIsInput(i))
            {
                const char* tensorName = engine.getBindingName(i);
                gLogInfo << "Dumping output tensor " << tensorName << ":" << std::endl;
                bufferManager.dumpBuffer(gLogInfo, tensorName);
            }
        }
    }

    if (gParams.dumpLayerTime)
    {
        gLogInfo << profiler;
    }

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    context->destroy();
}

static void printUsage()
{
    printf("\n");
    printf("Mandatory params:\n");
    printf("  --deploy=<file>          Caffe deploy file\n");
    printf("  OR --uff=<file>          UFF file\n");
    printf("  OR --onnx=<file>         ONNX Model file\n");
    printf("  OR --loadEngine=<file>   Load a saved engine\n");

    printf("\nMandatory params for UFF:\n");
    printf(
        "  --uffInput=<name>,C,H,W Input blob name and its dimensions for UFF parser (can be specified multiple "
        "times)\n");
    printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

    printf("\nMandatory params for Caffe:\n");
    printf("  --output=<name>      Output blob name (can be specified multiple times)\n");

    printf("\nOptional params:\n");
    printf("  --model=<file>          Caffe model file (default = no model, random weights used)\n");
    printf("  --batch=N               Set batch size (default = %d)\n", gParams.batchSize);
    printf("  --device=N              Set cuda device to N (default = %d)\n", gParams.device);
    printf("  --iterations=N          Run N iterations (default = %d)\n", gParams.iterations);
    printf("  --avgRuns=N             Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n",
        gParams.avgRuns);
    printf(
        "  --percentile=P          For each iteration, report the percentile time at P percentage (0<=P<=100, with 0 "
        "representing min, and 100 representing max; default = %.1f%%)\n",
        gParams.pct);
    printf("  --workspace=N           Set workspace size in megabytes (default = %d)\n", gParams.workspaceSize);
    printf("  --safe                  Only test the functionality available in safety restricted flows.\n");
    printf("  --fp16                  Run in fp16 mode (default = false). Permits 16-bit kernels\n");
    printf("  --int8                  Run in int8 mode (default = false). Currently no support for ONNX model.\n");
    printf("  --verbose               Use verbose logging (default = false)\n");
    printf("  --saveEngine=<file>     Save a serialized engine to file.\n");
    printf("  --loadEngine=<file>     Load a serialized engine from file.\n");
    printf("  --plugins=<file>        Load a TensorRT custom plugin.\n");
    printf("  --calib=<file>          Read INT8 calibration cache file.  Currently no support for ONNX model.\n");
    printf("  --calibOut=<file>       Write INT8 calibration cache file.  Currently no support for ONNX model.\n");
    printf(
        "  --useDLACore=N          Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
        "where n is the number of DLA engines on the platform.\n");
    printf(
        "  --allowGPUFallback      If --useDLACore flag is present and if a layer can't run on DLA, then run on GPU. "
        "\n");
    printf(
        "  --useSpinWait           Actively wait for work completion. This option may decrease multi-process "
        "synchronization time at the cost of additional CPU usage. (default = false)\n");
    printf("  --dumpOutput            Dump outputs at end of test. \n");
    printf("  --dumpLayerTime         Dump inferencing time of each layer at end of test. \n");
    printf("  -h, --help              Print usage\n");
    fflush(stdout);
}

bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

template <typename T>
bool parseAtoi(const char* arg, const char* name, T& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = static_cast<T>(atoi(arg + n + 3));
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
    return parseAtoi<int>(arg, name, value);
}

bool parseUnsigned(const char* arg, const char* name, unsigned int& value)
{
    return parseAtoi<unsigned int>(arg, name, value);
}

// parse a boolean option of the form --name, or optionally, -letter.
bool parseBool(const char* arg, const char* name, bool& value, char letter = '\0')
{
    bool match
        = arg[0] == '-' && ((arg[1] == '-' && !strcmp(arg + 2, name)) || (letter && arg[1] == letter && !arg[2]));
    if (match)
    {
        // Always report the long form of the option.
        gLogInfo << name << std::endl;
        value = true;
    }
    return match;
}

bool parseFloat(const char* arg, const char* name, float& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atof(arg + n + 3);
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool validateArgs()
{
    // UFF and Caffe files require output nodes to be specified.
    if ((!gParams.uffFile.empty() || !gParams.deployFile.empty()) && gParams.outputs.empty())
    {
        gLogError << "ERROR: At least one output must be specified." << std::endl;
        return false;
    }
    if (!gParams.uffFile.empty() && gParams.uffInputs.empty())
    {
        gLogError << "ERROR: At least one UFF input must be specified to run UFF models." << std::endl;
        return false;
    }
    if (!gParams.loadEngine.empty() && !gParams.saveEngine.empty())
    {
        gLogError << "ERROR: --saveEngine and --loadEngine cannot be specified at the same time." << std::endl;
        return false;
    }
    return true;
}

bool parseArgs(int argc, char* argv[])
{
    if (argc < 2)
    {
        printUsage();
        return false;
    }

    for (int j = 1; j < argc; j++)
    {
        if (parseString(argv[j], "model", gParams.modelFile) || parseString(argv[j], "deploy", gParams.deployFile))
        {
            continue;
        }
        if (parseString(argv[j], "saveEngine", gParams.saveEngine))
        {
            continue;
        }
        if (parseString(argv[j], "loadEngine", gParams.loadEngine))
        {
            continue;
        }
        if (parseString(argv[j], "engine", gParams.engine))
        {
            gLogError << "--engine has been deprecated. Please use --saveEngine and --loadEngine instead." << std::endl;
            return false;
        }
        if (parseString(argv[j], "uff", gParams.uffFile))
        {
            continue;
        }

        if (parseString(argv[j], "onnx", gParams.onnxModelFile))
        {
            continue;
        }

        if (parseString(argv[j], "calib", gParams.calibrationCache))
            continue;

        if (parseString(argv[j], "calibOut", gParams.outputCalibrationCache))
            continue;

        std::string input;
        if (parseString(argv[j], "input", input))
        {
            gLogWarning << "--input has been deprecated and ignored." << std::endl;
            continue;
        }

        std::string output;
        if (parseString(argv[j], "output", output))
        {
            gParams.outputs.push_back(output);
            continue;
        }

        std::string uffInput;
        if (parseString(argv[j], "uffInput", uffInput))
        {
            std::vector<std::string> uffInputStrs = split(uffInput, ',');
            if (uffInputStrs.size() != 4)
            {
                gLogError << "Invalid uffInput: " << uffInput << std::endl;
                return false;
            }

            gParams.uffInputs.push_back(std::make_pair(uffInputStrs[0],
                Dims3(atoi(uffInputStrs[1].c_str()), atoi(uffInputStrs[2].c_str()), atoi(uffInputStrs[3].c_str()))));
            continue;
        }

        if (parseInt(argv[j], "batch", gParams.batchSize) || parseInt(argv[j], "iterations", gParams.iterations)
            || parseInt(argv[j], "avgRuns", gParams.avgRuns) || parseInt(argv[j], "device", gParams.device)
            || parseInt(argv[j], "workspace", gParams.workspaceSize)
            || parseInt(argv[j], "useDLACore", gParams.useDLACore))
            continue;

        if (parseFloat(argv[j], "percentile", gParams.pct))
            continue;

        std::string plugin;
        if (parseString(argv[j], "plugins", plugin))
        {
            gParams.plugins.push_back(plugin);
            continue;
        }

        if (parseBool(argv[j], "safe", gParams.safeMode) || parseBool(argv[j], "fp16", gParams.fp16)
            || parseBool(argv[j], "int8", gParams.int8) || parseBool(argv[j], "verbose", gParams.verbose)
            || parseBool(argv[j], "allowGPUFallback", gParams.allowGPUFallback)
            || parseBool(argv[j], "useSpinWait", gParams.useSpinWait)
            || parseBool(argv[j], "dumpOutput", gParams.dumpOutput)
            || parseBool(argv[j], "dumpLayerTime", gParams.dumpLayerTime)
            || parseBool(argv[j], "help", gParams.help, 'h'))
            continue;

        gLogError << "Unknown argument: " << argv[j] << std::endl;
        return false;
    }

    return validateArgs();
}

static ICudaEngine* createEngine()
{
    ICudaEngine* engine{nullptr};

    // Load serialized engine file if specified by user
    if (!gParams.loadEngine.empty())
    {
        std::vector<char> engineData;
        size_t fsize{0};

        {
            // Open engine file
            std::ifstream engineFile(gParams.loadEngine, std::ios::binary);
            if (!engineFile.good())
            {
                gLogInfo << "Error loading engine file: " << gParams.loadEngine << std::endl;
                return engine;
            }

            // Read engine file to memory
            engineFile.seekg(0, engineFile.end);
            fsize = engineFile.tellg();
            engineFile.seekg(0, engineFile.beg);
            engineData.resize(fsize);
            engineFile.read(engineData.data(), fsize);
            engineFile.close();
        }

        // Create runtime
        IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
        if (gParams.useDLACore >= 0)
        {
            runtime->setDLACore(gParams.useDLACore);
        }

        // Create engine
        engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
        gLogInfo << gParams.loadEngine << " has been successfully loaded." << std::endl;

        runtime->destroy();
        return engine;
    }

    // User has not provided an engine file
    if ((!gParams.deployFile.empty()) || (!gParams.uffFile.empty()) || (!gParams.onnxModelFile.empty()))
    {
        if (!gParams.uffFile.empty())
        {
            engine = uffToTRTModel();
        }
        else if (!gParams.onnxModelFile.empty())
        {
            engine = onnxToTRTModel();
        }
        else
        {
            engine = caffeToTRTModel();
        }

        if (!engine)
        {
            gLogError << "Engine could not be created" << std::endl;
            return nullptr;
        }

        // User wants to save engine to file
        if (!gParams.saveEngine.empty())
        {
            // Open output file
            std::ofstream engineFile(gParams.saveEngine, std::ios::binary);
            if (!engineFile)
            {
                gLogError << "Could not open output engine file: " << gParams.saveEngine << std::endl;
                return nullptr;
            }

            IHostMemory* serializedEngine = engine->serialize();
            if (serializedEngine == nullptr)
            {
                gLogError << "Could not serialize engine." << std::endl;
                return nullptr;
            }

            engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
            serializedEngine->destroy();
            gLogInfo << "Engine has been successfully saved to: " << gParams.saveEngine << std::endl;
        }

        return engine;
    }

    // Complain about empty deploy file
    gLogError << "Deploy file not specified" << std::endl;

    return nullptr;
}

int main(int argc, char** argv)
{
    // create a TensorRT model from the caffe/uff/onnx model and serialize it to a stream

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    if (!parseArgs(argc, argv))
    {
        return gLogger.reportFail(sampleTest);
    }

    if (gParams.help)
    {
        printUsage();
        return gLogger.reportPass(sampleTest);
    }

    if (gParams.verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }

    cudaSetDevice(gParams.device);

    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    for (const auto& plugin : gParams.plugins)
    {
	if (EXIT_SUCCESS != loadLibrary(plugin))
        {
            return gLogger.reportFail(sampleTest);
        }
    }

    ICudaEngine* engine = createEngine();
    if (!engine)
    {
        gLogError << "Engine could not be created" << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    if (gParams.uffFile.empty() && gParams.onnxModelFile.empty())
    {
        nvcaffeparser1::shutdownProtobufLibrary();
    }
    else if (gParams.deployFile.empty() && gParams.onnxModelFile.empty())
    {
        nvuffparser::shutdownProtobufLibrary();
    }

    doInference(*engine);
    engine->destroy();

    return gLogger.reportPass(sampleTest);
}
