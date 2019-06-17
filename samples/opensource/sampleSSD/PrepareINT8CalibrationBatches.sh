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

if [ -z "$1" ]
then
    TEMP_DIR=./
else
    TEMP_DIR=$1/
fi

PASCAL_VOC2007_DATASET=$TEMP_DIR/VOCtrainval_06-Nov-2007.tar
wget -O $PASCAL_VOC2007_DATASET http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xf $PASCAL_VOC2007_DATASET -C $TEMP_DIR

if ! type "convert" > /dev/null; then
    echo "Install convert utility using apt-get to proceed"
    exit 1
fi

python batchPrepare.py --inDir $TEMP_DIR/VOCdevkit/VOC2007/JPEGImages/ --outDir $TEMP_DIR
