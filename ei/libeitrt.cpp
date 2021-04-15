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
 * 
 *  Modified 2021 Edge Impulse
 */

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

#include "libeitrt.h"

class EiTrt
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    EiTrt(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build(const char* model_file_name);

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(float* input, float* output, int output_size);

    ICudaEngine* createCudaEngine(const char* model_file_name);
    ICudaEngine* getCudaEngine(const char* model_file_name);

    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    std::string input_name;
    std::string output_name;
    int32_t input_size; //calculated from dimensions

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser, const char* model_file_name);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, float* input);

    bool reportOutput(const samplesCommon::BufferManager& buffers, float* output, int32_t output_size);
};

void writeBuffer(void* buffer, size_t size, string const& path)
{
    ofstream stream(path.c_str(), ios::binary);

    if (stream)
    {
        stream.write(static_cast<char*>(buffer), size);
    }
}

// Returns empty string iff can't read the file
string readBuffer(string const& path)
{
    string buffer;
    ifstream stream(path.c_str(), ios::binary);

    if (stream)
    {
        stream >> noskipws;
        copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
    }

    return buffer;
}

ICudaEngine* EiTrt::createCudaEngine(const char* model_file_name)
{
    sample::gLogInfo << "Creating engine from ONNX model\n";
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return nullptr;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return nullptr;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return nullptr;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return nullptr;
    }

    auto constructed = constructNetwork(builder, network, config, parser, model_file_name);
    if (!constructed)
    {
        return nullptr;
    }

    return builder->buildEngineWithConfig(*network, *config);
}

ICudaEngine* EiTrt::getCudaEngine(const char* model_file_name)
{
    string enginePath{model_file_name}; 
    enginePath += ".engine";
    ICudaEngine* engine{nullptr};

    string buffer = readBuffer(enginePath);
    
    if (buffer.size())
    {
        // Try to deserialize engine.
        SampleUniquePtr<nvinfer1::IRuntime> runtime{createInferRuntime(sample::gLogger)};
        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);

    } else {
        sample::gLogInfo << "No existing engine found\n";
    }

    if (!engine)
    {
        // Fallback to creating engine from scratch.
        engine = createCudaEngine(model_file_name);

        if (engine)
        {
            SampleUniquePtr<nvinfer1::IHostMemory> engine_plan{engine->serialize()};
            // Try to save engine for future uses.
            writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);
        }
    } else {
        sample::gLogInfo << "Successfully deserialized existing engine\n";
    }
    return engine;
}

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool EiTrt::build(const char* model_file_name)
{
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>( getCudaEngine(model_file_name), samplesCommon::InferDeleter() );
    if (!mEngine)
    {
        return false;
    }

    auto mInputDims = mEngine->getBindingDimensions(0);
    sample::gLogInfo << "Input tensor name: " << mEngine->getBindingName(0) << std::endl;
    sample::gLogInfo << "Output tensor name: " << mEngine->getBindingName(1) << std::endl;
    input_name = mEngine->getBindingName(0);
    output_name = mEngine->getBindingName(1);
    sample::gLogInfo << "Parsing input dimensions:\n";
    input_size = 1;
    for(int i = 0; i < mInputDims.nbDims; i++) {
        sample::gLogInfo << mInputDims.d[i] << endl;
        input_size *= mInputDims.d[i];
    }
    sample::gLogInfo << "Total input size: " << input_size << endl;

    mOutputDims = mEngine->getBindingDimensions(1);

    context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool EiTrt::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser, const char* model_file_name)
{
    auto parsed = parser->parseFromFile(model_file_name,
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize(512_MiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool EiTrt::infer(float* input, float* output, int output_size) 
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    // Read the input data into the managed buffers
    if (!processInput(buffers, input))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    auto start = chrono::steady_clock::now();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    auto end = chrono::steady_clock::now();
 
    sample::gLogInfo << "Elapsed inference time in nanoseconds : "
        << chrono::duration_cast<chrono::nanoseconds>(end - start).count()
        << " ns" << endl;
 
    sample::gLogInfo << "Elapsed time in microseconds : "
        << chrono::duration_cast<chrono::microseconds>(end - start).count()
        << " µs" << endl;

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Report results
    if (!reportOutput(buffers, output, output_size))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool EiTrt::processInput(const samplesCommon::BufferManager& buffers, float* input)
{
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(input_name));
    if( hostDataBuffer ) {
        memcpy(hostDataBuffer, input, sizeof(float)*input_size);
        sample::gLogInfo << "First and last features: " << hostDataBuffer[0] << "," << hostDataBuffer[input_size-1] << endl;
    } else {
        sample::gLogError << "failed to get buffer by input tensor name." << endl;
        return false;
    }
 
    return true;
}

//!
//! \brief Classifies digits and report result
//!
//! \return whether the classification output matches expectations
//!
bool EiTrt::reportOutput(const samplesCommon::BufferManager& buffers, float* output, int32_t output_size)
{
    float* output_from_engine = static_cast<float*>(buffers.getHostBuffer(output_name));

    if( output_from_engine ) {
        memcpy(output, output_from_engine, output_size*sizeof(float)); //use output size provided in call.
        // output size derived by TensorRT parser is always -1.  Even though web onnx parser says 1x2 (or 1x3, etc)
        sample::gLogInfo << "Output size (according to onnx parser, currently not used): " << mOutputDims.d[1] << endl;
        float val{0.0f};
        int idx{0};

        // For some other model, we had to calculate the softmax manually here
        // For EI model, we already do the softmax in the model
        sample::gLogInfo << "Output:" << std::endl;
        for (int i = 0; i < output_size; i++)
        {
            sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                            << " "
                            << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*')
                            << std::endl;
        }
        sample::gLogInfo << std::endl;
        return true;
    } else {
        sample::gLogError << " Failed to get buffer by output tensor name";
        return false;
    }  
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams()
{
    samplesCommon::OnnxSampleParams params;
    params.dlaCore = -1;
    params.int8 = false;
    params.fp16 = false;

    return params;
}

EiTrt* libeitrt::create_EiTrt(const char* model_file_name, bool debug)
{
    // TODO set debug level: kERROR
    // sample::setReportableSeverity(
    sample::gLogInfo << "EI TensorRT lib v1.4" << std::endl;
    auto handle = new EiTrt(initializeSampleParams());
    // TODO proper error checking and return null
    handle->build(model_file_name);
    return handle;
}

int libeitrt::infer(EiTrt* ei_trt_handle, float* input, float* output, int output_size)
{
    if (!ei_trt_handle->infer(input, output, output_size))
    {
        return -2;
    }
    return 0;
}