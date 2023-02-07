/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_CAFFE_PARSER_READ_PROTO_H
#define TRT_CAFFE_PARSER_READ_PROTO_H

#include <fstream>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

#include "caffeMacros.h"
#include "trtcaffe.pb.h"

namespace nvcaffeparser1
{
// There are some challenges associated with importing caffe models. One is that
// a .caffemodel file just consists of layers and doesn't have the specs for its
// input and output blobs.
//
// So we need to read the deploy file to get the input

bool readBinaryProto(trtcaffe::NetParameter* net, const char* file, size_t bufSize)
{
    CHECK_NULL_RET_VAL(net, false)
    CHECK_NULL_RET_VAL(file, false)
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in | std::ios::binary);
    if (!stream)
    {
        RETURN_AND_LOG_ERROR(false, "Could not open file " + std::string(file));
    }

    IstreamInputStream rawInput(&stream);
    CodedInputStream codedInput(&rawInput);
#if GOOGLE_PROTOBUF_VERSION >= 3011000
        codedInput.SetTotalBytesLimit(int(bufSize));
#else
        // Note: This WARs the very low default size limit (64MB)
        codedInput.SetTotalBytesLimit(int(bufSize), -1);
#endif
    bool ok = net->ParseFromCodedStream(&codedInput);
    stream.close();

    if (!ok)
    {
        RETURN_AND_LOG_ERROR(false, "Could not parse binary model file");
    }

    return ok;
}

bool readTextProto(trtcaffe::NetParameter* net, const char* file)
{
    CHECK_NULL_RET_VAL(net, false)
    CHECK_NULL_RET_VAL(file, false)
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in);
    if (!stream)
    {
        RETURN_AND_LOG_ERROR(false, "Could not open file " + std::string(file));
    }

    IstreamInputStream input(&stream);
    bool ok = google::protobuf::TextFormat::Parse(&input, net);
    stream.close();
    return ok;
}
} //namespace nvcaffeparser1
#endif //TRT_CAFFE_PARSER_READ_PROTO_H
