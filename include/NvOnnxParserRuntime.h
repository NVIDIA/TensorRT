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

#ifndef NV_ONNX_PARSER_RUNTIME_H
#define NV_ONNX_PARSER_RUNTIME_H

#include "NvOnnxParser.h"

namespace nvonnxparser
{

 /** \class IPluginFactory
 *
 * \brief a destroyable plugin factory object
 */
class IPluginFactory : public nvinfer1::IPluginFactory
{
public:
    /** \brief destroy this object
     */
    virtual void destroy() = 0;
protected:
    virtual ~IPluginFactory() {}
};

} // namespace nvonnxparser

extern "C" TENSORRTAPI void* createNvOnnxParserPluginFactory_INTERNAL(void* logger, int version);

namespace nvonnxparser
{

#ifdef SWIG
inline IPluginFactory* createPluginFactory(nvinfer1::ILogger* logger)
{
    return static_cast<IPluginFactory*>(
        createNvOnnxParserPluginFactory_INTERNAL(logger, NV_ONNX_PARSER_VERSION));
}
#endif // SWIG

namespace
{

/** \brief Create a new plugin factory for deserializing engines built using
 *         the ONNX parser.
 *
 * This plugin factory handles deserialization of the plugins that are built
 * into the ONNX parser. Engines built using the ONNX parser must use this
 * plugin factory during deserialization.
 *
 * \param logger The logger to use
 *
 * \return a new plugin factory object or NULL if an error occurred
 * \see IPluginFactory
 */
inline IPluginFactory* createPluginFactory(nvinfer1::ILogger& logger)
{
    return static_cast<IPluginFactory*>(
        createNvOnnxParserPluginFactory_INTERNAL(&logger, NV_ONNX_PARSER_VERSION));
}

} // namespace

} // namespace nvonnxparser

#endif // NV_ONNX_PARSER_RUNTIME_H
