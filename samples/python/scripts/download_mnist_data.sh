#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

mkdir -p /usr/src/tensorrt/data/mnist && cd /usr/src/tensorrt/data/mnist
wget -nc http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -nc http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -nc http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip -kf train-images-idx3-ubyte.gz
gunzip -kf t10k-images-idx3-ubyte.gz
gunzip -kf t10k-labels-idx1-ubyte.gz
cd -
