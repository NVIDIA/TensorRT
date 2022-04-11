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

SHELL=/bin/bash -o pipefail
TARGET?=$(shell uname -m)
LIBDIR?=lib
VERBOSE?=0
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif
CUDA_TRIPLE=x86_64-linux
CUBLAS_TRIPLE=x86_64-linux-gnu
DLSW_TRIPLE=x86_64-linux-gnu
ifeq ($(TARGET), aarch64)
CUDA_TRIPLE=aarch64-linux
CUBLAS_TRIPLE=aarch64-linux-gnu
DLSW_TRIPLE=aarch64-linux-gnu
endif
ifeq ($(TARGET), qnx)
CUDA_TRIPLE=aarch64-qnx
CUBLAS_TRIPLE=aarch64-qnx-gnu
DLSW_TRIPLE=aarch64-unknown-nto-qnx
endif
ifeq ($(TARGET), ppc64le)
CUDA_TRIPLE=ppc64le-linux
CUBLAS_TRIPLE=ppc64le-linux
DLSW_TRIPLE=ppc64le-linux
endif
ifeq ($(TARGET), android64)
DLSW_TRIPLE=aarch64-linux-androideabi
CUDA_TRIPLE=$(DLSW_TRIPLE)
CUBLAS_TRIPLE=$(DLSW_TRIPLE)
endif
export TARGET
export VERBOSE
export LIBDIR
export CUDA_TRIPLE
export CUBLAS_TRIPLE
export DLSW_TRIPLE

samples = SemanticSegmentation

.PHONY: all clean help
all:
	$(AT)$(foreach sample,$(samples), $(MAKE) -C $(sample) &&) :

clean:
	$(AT)$(foreach sample,$(samples), $(MAKE) clean -C $(sample) &&) :

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample,$(samples), echo -e "\t$(sample)" &&) :
	$(AT)echo -e "\nCommands:"
	$(AT)echo -e "\tall - build all samples."
	$(AT)echo -e "\tclean - clean all samples."
	$(AT)echo -e "\nVariables:"
	$(AT)echo -e "\tTARGET - Specify the target to build for."
	$(AT)echo -e "\tVERBOSE - Specify verbose output."
	$(AT)echo -e "\tCUDA_INSTALL_DIR - Directory where cuda installs to."
