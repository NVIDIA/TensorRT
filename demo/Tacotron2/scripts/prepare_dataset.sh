#!/usr/bin/env bash
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

set -e

DATADIR="LJSpeech-1.1"
BZ2ARCHIVE="${DATADIR}.tar.bz2"
ENDPOINT="http://data.keithito.com/data/speech/$BZ2ARCHIVE"

if [ ! -d "$DATADIR" ]; then
  echo "dataset is missing, unpacking ..."
  if [ ! -f "$BZ2ARCHIVE" ]; then
    echo "dataset archive is missing, downloading ..."
    wget "$ENDPOINT"
  fi
  tar jxvf "$BZ2ARCHIVE"
fi
