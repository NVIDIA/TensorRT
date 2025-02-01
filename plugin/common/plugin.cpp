/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/plugin.h"
#include <variant>

namespace nvinfer1
{
namespace pluginInternal
{

// Helper to create per-context cudnn/cublas singleton managed by std::shared_ptr.
// Unlike conventional singletons, singleton created with this will be released
// when not needed, instead of on process exit.
// Objects of this class shall always be declared static / global, and shall never own cudnn/cublas handle
// resources.
template <typename T>
class PerContextPluginHandleSingletonCreator
{
public:
    // creator returning std::unique_ptr is by design.
    // It forces separation of memory for T and memory for control blocks.
    // So when T is released, but we still have observer weak_ptr in mObservers, the T mem block can be released.
    // creator itself must not own cudnn/cublas handle resources. Only the object it creates can.
    PerContextPluginHandleSingletonCreator(std::function<std::unique_ptr<T>()> cublasCreator)
        : mCreator(std::move(cublasCreator))
    {
    }

    PerContextPluginHandleSingletonCreator(std::function<std::unique_ptr<T>(char const*)> cudnnCreator)
        : mCreator(std::move(cudnnCreator))
    {
    }

    // \param executionContextIdentifier Unique pointer to identify contexts having overlapping lifetime.
    // \param callerPluginName Optional name of the plugin that invokes this creator
    std::shared_ptr<T> operator()(void* executionContextIdentifier, char const* callerPluginName = nullptr)
    {
        std::lock_guard<std::mutex> lk{mMutex};
        std::shared_ptr<T> result = mObservers[executionContextIdentifier].lock();
        if (result == nullptr)
        {
            auto deleter = [this, executionContextIdentifier](T* obj) {
                if (obj == nullptr)
                {
                    return;
                }
                delete obj;
                // Clears observer to avoid growth of mObservers, in case users create/destroy
                // plugin handle contexts frequently.
                std::shared_ptr<T> observedObjHolder;
                // The destructor of observedObjHolder may attempt to acquire a lock on mMutex.
                // To avoid deadlock, it's critical to release the lock here held by lk first,
                // before destroying observedObjHolder. Hence observedObjHolder must be declared
                // before lk.
                std::lock_guard<std::mutex> lk{mMutex};
                // Must check observer again because another thread may create new instance for
                // this ctx just before we lock mMutex. We can't infer that the observer is
                // stale from the fact that obj is destroyed, because shared_ptr ref-count
                // checking and observer removing are not in one atomic operation, and the
                // observer may be changed to observe another instance.
                auto it = mObservers.find(executionContextIdentifier);
                if (it != mObservers.end())
                {
                    observedObjHolder = it->second.lock();
                    if (observedObjHolder == nullptr)
                    {
                        mObservers.erase(executionContextIdentifier);
                    }
                }
            };
            if (std::holds_alternative<std::function<std::unique_ptr<T>()>>(mCreator))
            {
                auto concreteCreator = std::get<std::function<std::unique_ptr<T>()>>(mCreator);
                result = std::shared_ptr<T>{concreteCreator().release(), std::move(deleter)};
            }
            else if (std::holds_alternative<std::function<std::unique_ptr<T>(char const*)>>(mCreator))
            {
                auto concreteCreator = std::get<std::function<std::unique_ptr<T>(char const*)>>(mCreator);
                result = std::shared_ptr<T>{concreteCreator(callerPluginName).release(), std::move(deleter)};
            }
            else
            {
                PLUGIN_ERROR("Unsupported cuDNN/cuBLAS wrapper creator type!");
            }
            // Update the per-context observer with the new resource
            mObservers.at(executionContextIdentifier) = result;
        }

        return result;
    };

private:
    std::variant<
        /* createPluginCublasWrapperImpl */ std::function<std::unique_ptr<T>()>,
        /* createPluginCudnnWrapperImpl */ std::function<std::unique_ptr<T>(char const*)>>
        mCreator;
    mutable std::mutex mMutex;
    // cudnn/cublas handle resources are per-context.
    std::unordered_map</*contextIdentifier*/ void*, std::weak_ptr<T>> mObservers;
}; // class PerContextPluginHandleSingletonCreator

std::unique_ptr<CudnnWrapper> createPluginCudnnWrapperImpl(char const* callerPluginName)
{
    // callerPluginName is used to enrich downstream cudnn error message with caller plugin info
    return std::make_unique<CudnnWrapper>(/*initHandle*/ true, callerPluginName);
}

std::unique_ptr<CublasWrapper> createPluginCublasWrapperImpl()
{
    return std::make_unique<CublasWrapper>(/*initHandle*/ true);
}

static PerContextPluginHandleSingletonCreator<CudnnWrapper> gCreatePluginCudnnHandleWrapper(
    createPluginCudnnWrapperImpl);
static PerContextPluginHandleSingletonCreator<CublasWrapper> gCreatePluginCublasHandleWrapper(
    createPluginCublasWrapperImpl);

std::shared_ptr<CudnnWrapper> createPluginCudnnWrapper(void* executionContextIdentifier, char const* callerPluginName)
{
    return gCreatePluginCudnnHandleWrapper(executionContextIdentifier, callerPluginName);
}

std::shared_ptr<CublasWrapper> createPluginCublasWrapper(void* executionContextIdentifier)
{
    return gCreatePluginCublasHandleWrapper(executionContextIdentifier);
}

} // namespace pluginInternal

namespace plugin
{

void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, PluginFieldCollection const* fc)
{
    for (int32_t i = 0; i < fc->nbFields; i++)
    {
        requiredFieldNames.erase(fc->fields[i].name);
    }
    if (!requiredFieldNames.empty())
    {
        std::stringstream msg{};
        msg << "PluginFieldCollection missing required fields: {";
        char const* separator = "";
        for (auto const& field : requiredFieldNames)
        {
            msg << separator << field;
            separator = ", ";
        }
        msg << "}";
        std::string msg_str = msg.str();
        PLUGIN_ERROR(msg_str.c_str());
    }
}

int32_t dimToInt32(int64_t d)
{
    if (d < std::numeric_limits<int32_t>::min() || d > std::numeric_limits<int32_t>::max())
    {
        std::stringstream msg{};
        msg << "Plugin cannot handle dimension outside of int32_t range: " << d;
        std::string msg_str = msg.str();
        PLUGIN_ERROR(msg_str.c_str());
    }
    return static_cast<int32_t>(d);
}

bool supportsMemPoolsHelper()
{
    int32_t device;
    PLUGIN_CUASSERT(cudaGetDevice(&device));
    int32_t value;
    PLUGIN_CUASSERT(cudaDeviceGetAttribute(&value, cudaDevAttrMemoryPoolsSupported, device));
    return value != 0;
}

bool supportsMemPools()
{
    static bool sResult = supportsMemPoolsHelper();
    return sResult;
}

} // namespace plugin
} // namespace nvinfer1
