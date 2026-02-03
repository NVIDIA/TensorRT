/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test library that uses TensorRT and can be dynamically loaded.
 * This reproduces issue #4617 where TensorRT crashes when used from
 * a dynamically loaded library.
 */

#include "NvInfer.h"
#include <iostream>
#include <memory>

using namespace nvinfer1;

// Simple logger for TensorRT
class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // Only print errors and warnings
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// Exported function that creates and destroys a TensorRT builder
extern "C" int test_tensorrt_builder()
{
    std::cout << "Creating TensorRT builder..." << std::endl;
    
    // Create a builder
    IBuilder* builder = createInferBuilder(gLogger);
    if (!builder)
    {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return 1;
    }
    
    std::cout << "Builder created successfully" << std::endl;
    std::cout << "Number of DLA cores: " << builder->getNbDLACores() << std::endl;
    
    // Create a network
    uint32_t flags = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(flags);
    if (!network)
    {
        std::cerr << "Failed to create network" << std::endl;
        delete builder;
        return 1;
    }
    
    std::cout << "Network created successfully" << std::endl;
    
    // Clean up
    delete network;
    delete builder;
    
    std::cout << "Builder destroyed successfully" << std::endl;
    
    return 0;
}

// Test function that exercises IBuilder lifecycle
extern "C" int test_builder_lifecycle()
{
    std::cout << "Testing IBuilder lifecycle..." << std::endl;
    
    for (int i = 0; i < 3; i++)
    {
        std::cout << "Iteration " << (i + 1) << std::endl;
        
        IBuilder* builder = createInferBuilder(gLogger);
        if (!builder)
        {
            std::cerr << "Failed to create builder in iteration " << (i + 1) << std::endl;
            return 1;
        }
        
        // Create and destroy a builder config
        IBuilderConfig* config = builder->createBuilderConfig();
        if (config)
        {
            delete config;
        }
        
        delete builder;
    }
    
    std::cout << "Builder lifecycle test completed successfully" << std::endl;
    return 0;
}
