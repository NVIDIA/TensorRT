/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Main test program that loads the TensorRT library dynamically using dlopen().
 * This reproduces issue #4617 where TensorRT crashes when the library is unloaded.
 */

#include <dlfcn.h>
#include <iostream>
#include <cstdlib>

typedef int (*test_func_t)();

int main(int argc, char** argv)
{
    std::cout << "=== TensorRT Issue #4617 Reproducer ===" << std::endl;
    std::cout << "Testing TensorRT usage from dynamically loaded library" << std::endl;
    std::cout << std::endl;
    
    // Load the test library
    std::cout << "Loading test library..." << std::endl;
    void* handle = dlopen("./libtest_tensorrt.so", RTLD_LAZY);
    if (!handle)
    {
        std::cerr << "Failed to load library: " << dlerror() << std::endl;
        return 1;
    }
    std::cout << "Library loaded successfully" << std::endl;
    std::cout << std::endl;
    
    // Clear any existing error
    dlerror();
    
    // Get the test function
    test_func_t test_tensorrt_builder = (test_func_t)dlsym(handle, "test_tensorrt_builder");
    const char* dlsym_error = dlerror();
    if (dlsym_error)
    {
        std::cerr << "Failed to load symbol 'test_tensorrt_builder': " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }
    
    // Get the lifecycle test function
    test_func_t test_builder_lifecycle = (test_func_t)dlsym(handle, "test_builder_lifecycle");
    dlsym_error = dlerror();
    if (dlsym_error)
    {
        std::cerr << "Failed to load symbol 'test_builder_lifecycle': " << dlsym_error << std::endl;
        dlclose(handle);
        return 1;
    }
    
    // Run the tests
    std::cout << "Running test_tensorrt_builder()..." << std::endl;
    int result = test_tensorrt_builder();
    if (result != 0)
    {
        std::cerr << "test_tensorrt_builder() failed with code " << result << std::endl;
        dlclose(handle);
        return result;
    }
    std::cout << std::endl;
    
    std::cout << "Running test_builder_lifecycle()..." << std::endl;
    result = test_builder_lifecycle();
    if (result != 0)
    {
        std::cerr << "test_builder_lifecycle() failed with code " << result << std::endl;
        dlclose(handle);
        return result;
    }
    std::cout << std::endl;
    
    // Close the library - this is where the crash typically occurs
    std::cout << "Closing library..." << std::endl;
    std::cout << "NOTE: If TensorRT has the bug, the program may crash here" << std::endl;
    std::cout << "      due to dlclose() being called from a static C++ destructor" << std::endl;
    
    if (dlclose(handle) != 0)
    {
        std::cerr << "Failed to close library: " << dlerror() << std::endl;
        return 1;
    }
    
    std::cout << "Library closed successfully" << std::endl;
    std::cout << std::endl;
    std::cout << "=== Test completed successfully ===" << std::endl;
    std::cout << "If you see this message, the issue is either fixed or not present" << std::endl;
    
    return 0;
}
