#!/bin/bash
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
# This short shell script will extract all the strong "text" symbols from the
# shared library and create a new "stub" shared library with the same symbols.
# The body of these functions will be empty and therefore have no dependencies.
# This scripts uses whatever CC is defined in the user's environment.
#

set -o pipefail

# check arguments
if [ $# -ne 2 ] ; then
    echo "Usage: $(basename $0) IN_LIBFILE OUT_LIBFILE"
    exit 1
fi

IN_LIBFILE="$1"
OUT_LIBFILE="$2"

# check compiler
if [ -z "${CC}" ] ; then
    echo "Error: Environment variable 'CC' has not been defined"
    exit 1
fi

SONAME=$(readelf -d "${IN_LIBFILE}" | grep '(SONAME)' | cut -d [ -f 2 | cut -d ] -f 1)

# make stub library
nm -D "${IN_LIBFILE}" | \
    awk '{if ($2 == "T") { print "void",$3,"() {}" }}' | \
    "${CC}" -x c -O0 -fPIC -shared -Wl,-soname=${SONAME} -Wl,--strip-all -o "${OUT_LIBFILE}" -

exit $?

