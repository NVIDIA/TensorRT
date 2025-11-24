/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stress test for issue #4617: repeatedly load and unload the TensorRT library
 * to verify the fix works correctly under stress conditions.
 */

#include <dlfcn.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

typedef int (*test_func_t)();

// Test configuration
const int NUM_ITERATIONS = 100;
const int NUM_THREADS = 4;

std::atomic<int> successCount{0};
std::atomic<int> failureCount{0};

void runLoadUnloadTest(int threadId, int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        // Load the library
        void* handle = dlopen("./libtest_tensorrt.so", RTLD_LAZY);
        if (!handle)
        {
            std::cerr << "Thread " << threadId << ", iteration " << i 
                      << ": Failed to load library: " << dlerror() << std::endl;
            failureCount++;
            continue;
        }
        
        // Get and call the test function
        test_func_t test_func = (test_func_t)dlsym(handle, "test_tensorrt_builder");
        if (!test_func)
        {
            std::cerr << "Thread " << threadId << ", iteration " << i 
                      << ": Failed to load symbol: " << dlerror() << std::endl;
            dlclose(handle);
            failureCount++;
            continue;
        }
        
        int result = test_func();
        if (result != 0)
        {
            std::cerr << "Thread " << threadId << ", iteration " << i 
                      << ": Test function failed with code " << result << std::endl;
            dlclose(handle);
            failureCount++;
            continue;
        }
        
        // Unload the library - this is where the crash would occur
        if (dlclose(handle) != 0)
        {
            std::cerr << "Thread " << threadId << ", iteration " << i 
                      << ": Failed to close library: " << dlerror() << std::endl;
            failureCount++;
            continue;
        }
        
        successCount++;
        
        // Small delay to allow cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

int main(int argc, char** argv)
{
    std::cout << "=== TensorRT Issue #4617 Stress Test ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Threads: " << NUM_THREADS << std::endl;
    std::cout << "  Iterations per thread: " << NUM_ITERATIONS << std::endl;
    std::cout << "  Total operations: " << (NUM_THREADS * NUM_ITERATIONS) << std::endl;
    std::cout << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Create threads
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threads.emplace_back(runLoadUnloadTest, i, NUM_ITERATIONS);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads)
    {
        thread.join();
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << std::endl;
    std::cout << "=== Test Results ===" << std::endl;
    std::cout << "Successful operations: " << successCount << std::endl;
    std::cout << "Failed operations: " << failureCount << std::endl;
    std::cout << "Duration: " << duration.count() << " ms" << std::endl;
    std::cout << "Operations per second: " 
              << (successCount * 1000.0 / duration.count()) << std::endl;
    
    if (failureCount > 0)
    {
        std::cout << std::endl;
        std::cout << "=== TEST FAILED ===" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "=== TEST PASSED ===" << std::endl;
    return 0;
}
