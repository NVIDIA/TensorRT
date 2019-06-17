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

#include "NvCaffeParser.h"
#include "caffeParser.h"

using namespace nvcaffeparser1;

void nvcaffeparser1::shutdownProtobufLibrary()
{
    google::protobuf::ShutdownProtobufLibrary();
}

extern "C" void* createNvCaffeParser_INTERNAL()
{
    return nvcaffeparser1::createCaffeParser();
}

ICaffeParser* nvcaffeparser1::createCaffeParser()
{
    return new CaffeParser;
}