#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

############################# GENERATE C++ PROTO FILES ###################################
add_custom_target(parsers DEPENDS nvonnxparser)

add_definitions("-D_PROTOBUF_INSTALL_DIR=${Protobuf_INSTALL_DIR}")
add_compile_options("-Dgoogle=google_trtrepack")
set(TENSORRT_ROOT ${PROJECT_SOURCE_DIR})
set(TENSORRT_BUILD ${TRT_OUT_DIR})

# Write the libs out to the top-level of the build directory.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${TENSORRT_BUILD})
message(NOTICE "CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")

# Write the libs out to the top-level of the build directory.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${TENSORRT_BUILD})
message(NOTICE "CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")

include_directories(
   ${Protobuf_INCLUDE_DIR}
)

add_subdirectory(onnx)
