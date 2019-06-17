#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

find_package(PkgConfig)
pkg_search_module(ZLIB REQUIRED zlib)
set(zlib_INCLUDE_DIR ${ZLIB_INCLUDE_DIRS})
set(ADD_LINK_DIRECTORY ${ADD_LINK_DIRECTORY} ${ZLIB_LIBRARY_DIRS})
set(ADD_CFLAGS ${ADD_CFLAGS} ${ZLIB_CFLAGS_OTHER})

add_custom_target(zlib)
add_custom_target(zlib_copy_headers_to_destination)

