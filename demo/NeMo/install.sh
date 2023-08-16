#!/bin/sh
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Sourcing messes up the directory detection with readlink.
if [ ! "${0##*/}" = "install.sh" ]; then
	echo "Please run this install script, don't source it." >&2
	echo "Use -h for usage and help." >&2
	return 1
fi

NEMO_DIR=$(dirname "$(readlink -f "$0")")
DEMO_DIR=$(dirname "${NEMO_DIR}")
SCRIPT_DIR=$(dirname "${DEMO_DIR}")/scripts

DEPENDENCIES_DIR="temp"
BUILD_SRCLIBS=1
BUILD_NINJA=0
ARG_JOBS=1
ARG_HELP=0

install_essential_tools() {
	pip_not_found=$(pip --version 2>&1 | grep -o "not found");
	if [ "$pip_not_found" != "" ];
	then
		echo " > Installing pip..."
		apt-get update
		apt-get install -y python3-dev
		cd "${1}" || exit
		if [ ! -f "get-pip.py" ]; then
			apt-get install -y wget
			wget https://bootstrap.pypa.io/get-pip.py
		fi
		python3 get-pip.py
		cd ..
	fi

	git_not_found=$(git --version 2>&1 | grep -o "not found");
	if [ "$git_not_found" != "" ];
	then
		echo " > Installing git..."
		apt-get update
		apt-get install -y git
	fi
}

install_ninja() {
	if [ ! -d "ninja" ]; then
		git clone https://github.com/ninja-build/ninja.git
	fi
	cd ninja || exit
	git checkout v1.11.1

	if [ ! -x "./ninja" ]; then
		CMD="python3 configure.py --bootstrap"
		echo " >> ${CMD}"
		eval "${CMD}"
		unset CMD
	else
		echo " > ninja already built!"
	fi

	PATH_WITH_NINJA="$(pwd):${PATH}"
	# Path exported for the current program scope only.
	export PATH="${PATH_WITH_NINJA}"
	unset PATH_WITH_NINJA
	cd ..
}

PACKAGE_NEEDS_REINSTALL=0

check_if_managed_install() {
	PACKAGE_NEEDS_REINSTALL=0
	dist_path="${1}"
	# https://packaging.python.org/en/latest/specifications/direct-url/
	if [ ! -f "${dist_path}/direct_url.json" ]; then
		PACKAGE_NEEDS_REINSTALL=1
		return
	fi
	if [ "$(grep -c "${NEMO_DIR}" "${dist_path}/direct_url.json")" != "1" ]; then
		PACKAGE_NEEDS_REINSTALL=1
	fi
}

apex_install_logic() {
	if [ ! -d "apex" ]; then
		git clone https://github.com/NVIDIA/apex.git
	fi

	cd apex || exit
	APEX_PATH="$(pwd)"
	git config --global --add safe.directory "${APEX_PATH}"
	unset APEX_PATH

	git checkout 5b5d41034b506591a316c308c3d2cd14d5187e23
	git apply "${NEMO_DIR}"/apex.patch # Bypass CUDA version check in apex

	torchcppext=$(pip show torch | grep Location | cut -d' ' -f2)"/torch/utils/cpp_extension.py"
	if [ ! -f "$torchcppext" ]; then
		echo "Could not locate torch installation using pip"
		exit 1
	fi
	sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))/pass/' "$torchcppext" # Bypass CUDA version check in torch
	unset torchcppext

	CMD="MAX_JOBS=${ARG_JOBS} python3 setup.py bdist_wheel -v --cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam"
	echo " >> ${CMD}"
	eval "${CMD}"
	unset CMD

	python3 -m pip install "$(find './dist' -name '*.whl' | head -n1)"
	cd ../
}

check_if_apex_needs_reinstall() {
	apex_loc="$(pip show apex | grep '^Location' | awk '{print $2}')"
	apex_dist_loc="$(find "${apex_loc}" -depth -maxdepth 1 -name 'apex*dist-info' -type d | head -n1)"

	check_if_managed_install "${apex_dist_loc}"
	apex_needs_reinstall=${PACKAGE_NEEDS_REINSTALL}
	echo "${apex_needs_reinstall}"

	unset apex_dist_loc
	unset apex_loc
}

install_apex() {
	has_apex=$(pip list | grep "^apex " | grep "apex" -o | awk '{print $1}' | awk '{print length}')
	apex_needs_reinstall=0

	if [ "$has_apex" != "4" ]; then
		apex_install_logic
	else
		check_if_apex_needs_reinstall
		if [ "$apex_needs_reinstall" != "0" ]; then
			echo " > Reinstalling Apex per demo version..."
			python3 -m pip uninstall -y apex
			apex_install_logic
		else
			echo " > Apex already installed!"
		fi
	fi
	unset apex_needs_reinstall
	unset has_apex
}

megatron_install_logic() {
	if [ ! -d "Megatron-LM" ]; then
		git clone -b main https://github.com/NVIDIA/Megatron-LM.git
	fi

	cd Megatron-LM || exit
	MEGATRON_PATH="$(pwd)"
	git config --global --add safe.directory "${MEGATRON_PATH}"
	unset MEGATRON_PATH

	git checkout 992da75a1fd90989eb1a97be8d9ff3eca993aa83
	CMD="python3 -m pip install ./"
	echo " >> ${CMD}"
	eval "${CMD}"
	unset CMD
	cd ../
}

check_if_megatron_needs_reinstall() {
	megatron_loc="$(pip show megatron-core | grep '^Location' | awk '{print $2}')"
	megatron_dist_loc="$(find "${megatron_loc}" -depth -maxdepth 1 -name 'megatron*dist-info' -type d | head -n1)"

	check_if_managed_install "${megatron_dist_loc}"
	megatron_needs_reinstall=${PACKAGE_NEEDS_REINSTALL}

	unset megatron_dist_loc
	unset megatron_loc
}

install_megatron() {
	has_megatron=$(pip list | grep "^megatron-core " | grep "megatron-core" -o | awk '{print $1}' | awk '{print length}')
	megatron_needs_reinstall=0

	if [ "$has_megatron" != "13" ]; then
		megatron_install_logic
	else
		check_if_megatron_needs_reinstall
		if [ "$megatron_needs_reinstall" != "0" ]; then
			echo " > Reinstalling Megatron per demo version..."
			python3 -m pip uninstall -y megatron-core
			megatron_install_logic
		else
			echo " > Megatron already installed!"
		fi
	fi
	unset megatron_needs_reinstall
	unset has_megatron
}

flash_attention_install_logic() {
	if [ ! -d "flash-attention" ]; then
		git clone https://github.com/HazyResearch/flash-attention.git
	fi

	cd flash-attention || exit
	FLASH_ATTENTION_PATH="$(pwd)"
	git config --global --add safe.directory "${FLASH_ATTENTION_PATH}"
	unset FLASH_ATTENTION_PATH

	git checkout v1.0.6
	CMD="MAX_JOBS=${ARG_JOBS} python3 setup.py bdist_wheel"
	echo " >> ${CMD}"
	eval "${CMD}"
	unset CMD
	python3 -m pip install "$(find './dist' -name '*.whl' | head -n1)"
	cd ..
}

check_if_flash_attention_needs_reinstall() {
	flash_attn_loc="$(pip show flash-attn | grep '^Location' | awk '{print $2}')"
	flash_attn_dist_loc="$(find "${flash_attn_loc}" -depth -maxdepth 1 -name 'flash_attn*dist-info' -type d | head -n1)"

	check_if_managed_install "${flash_attn_dist_loc}"
	flash_attn_needs_reinstall=${PACKAGE_NEEDS_REINSTALL}

	unset flash_attn_dist_loc
	unset flash_attn_loc
}

install_flash_attention() {
	has_flashattn=$(pip list | grep "^flash-attn " | grep "flash-attn" -o | awk '{print $1}' | awk '{print length}')
	flash_attn_needs_reinstall=0

	if [ "$has_flashattn" != "10" ]; then
		flash_attention_install_logic
	else
		check_if_flash_attention_needs_reinstall
		if [ "$flash_attn_needs_reinstall" != "0" ]; then
			echo " > Reinstalling flash_attn per demo version..."
			python3 -m pip uninstall -y flash-attn
			flash_attention_install_logic
		else
			echo " > flash-attention already installed!"
		fi
	fi

	unset flash_attn_needs_reinstall
	unset has_flashattn
}

transformer_engine_install_logic() {
	if [ ! -d "TransformerEngine" ]; then
		git clone https://github.com/NVIDIA/TransformerEngine.git
	fi

	cd TransformerEngine || exit
	TRANSFORMER_ENGINE_PATH="$(pwd)"
	git config --global --add safe.directory "${TRANSFORMER_ENGINE_PATH}"
	unset TRANSFORMER_ENGINE_PATH

	git checkout 804f120322a13cd5f21ea8268860607dcecd055c
	git submodule update --recursive --init
	CMD="MAKEFLAGS=-j${ARG_JOBS} MAX_JOBS=${ARG_JOBS} python3 setup.py bdist_wheel --framework=pytorch"
	echo " >> ${CMD}"
	eval "${CMD}"
	unset CMD
	python3 -m pip install "$(find './dist' -name '*.whl' | head -n1)"
	cd ..

	# Check for common point of failure with TE.
	has_te_loc=$(pip list | grep "^transformer-engine " | grep "transformer-engine" -o | awk '{print $1}' | awk '{print length}')
	[ "$has_te_loc" != "18" ] && {
		echo " > TransformerEngine install failed. Probable cause of failures:"
		echo "   - CUDNN location was not picked up. If your CUDNN include dir"
		echo "     is /path/to/cudnn/include and lib is /path/to/cudnn/lib,   "
		echo "     Invoke the script as CUDNN_PATH=/path/to/cudnn sh install.sh ..."
		exit 1
	}
	unset has_te_loc
}

check_if_transformer_engine_needs_reinstall() {
	te_loc="$(pip show transformer-engine | grep '^Location' | awk '{print $2}')"
	te_dist_loc="$(find "${te_loc}" -depth -maxdepth 1 -name 'transformer_engine*dist-info' -type d | head -n1)"

	check_if_managed_install "${te_dist_loc}"
	te_needs_reinstall=${PACKAGE_NEEDS_REINSTALL}

	unset te_dist_loc
	unset te_loc
}

install_transformer_engine() {
	has_te=$(pip list | grep "^transformer-engine " | grep "transformer-engine" -o | awk '{print $1}' | awk '{print length}')
	te_needs_reinstall=0

	if [ "$has_te" != "18" ]; then
		transformer_engine_install_logic
	else
		check_if_transformer_engine_needs_reinstall
		if [ "$te_needs_reinstall" != "0" ]; then
			echo " > Reinstalling TransformerEngine per demo version..."
			python3 -m pip uninstall -y transformer-engine
			transformer_engine_install_logic
		else
			echo " > TransformerEngine already installed!"
		fi
	fi

	unset te_needs_reinstall
	unset has_te

	# Patch TE files.
	sh "${NEMO_DIR}/patch_te.sh"
}

nemo_install_logic() {
	if [ ! -d "NeMo" ]; then
		git clone -b main https://github.com/NVIDIA/NeMo.git
	fi

	cd NeMo || exit
	NeMo_PATH="$(pwd)"
	git config --global --add safe.directory "${NeMo_PATH}"
	unset NeMo_PATH

	git checkout bf270794267e0240d8a8b2f2514c80c6929c76f1
	bash reinstall.sh
	cd ../
}

check_if_nemo_needs_reinstall() {
	nemo_loc="$(pip show nemo-toolkit | grep '^Location' | awk '{print $2}')"
	nemo_dist_loc="$(find "${nemo_loc}" -depth -maxdepth 1 -name 'nemo_toolkit*dist-info' -type d | head -n1)"

	check_if_managed_install "${nemo_dist_loc}"
	nemo_needs_reinstall=${PACKAGE_NEEDS_REINSTALL}

	unset nemo_dist_loc
	unset nemo_loc
}

install_nemo() {
	has_nemo=$(pip list | grep "^nemo-toolkit " | grep "nemo-toolkit" -o | awk '{print $1}' | awk '{print length}')
	nemo_needs_reinstall=0

	if [ "$has_nemo" != "12" ]; then
		nemo_install_logic
	else
		check_if_nemo_needs_reinstall
		if [ "$nemo_needs_reinstall" != "0" ]; then
			echo " > Reinstalling NeMo per demo version..."
			python3 -m pip uninstall -y nemo-toolkit
			nemo_install_logic
		else
			echo " > NeMo already installed!"
		fi
	fi
}

while [ "$#" -gt 0 ]; do
	case $1 in
	--deps)
		DEPENDENCIES_DIR="$2"
		shift
		;;
	-j | --jobs)
		ARG_JOBS="$2"
		shift
		;;
	--ninja) BUILD_NINJA=1 ;;
	--skipsrc) BUILD_SRCLIBS=0 ;;
	-h | --help) ARG_HELP=1 ;;
	*)
		echo "Unknown parameter passed: $1"
		echo "For help type: $0 --help"
		exit 1
		;;
	esac
	shift
done

if [ "$ARG_HELP" -eq "1" ]; then
	echo "Usage: sh $0 [options]"
	echo "All arguments are optional."
	echo " --help or -h         : Print this help menu."
	echo " [--deps] {temp}      : Path to download and build dependencies."
	echo " [-j | --jobs] {1}    : Number of jobs to use for building from source."
	echo " [--ninja]            : Flag to build ninja (if not present) to speed up installation."
	# skipsrc is not documented to prevent users from invoking it directly.
	exit
fi

DEPENDENCIES_DIR="${NEMO_DIR}/${DEPENDENCIES_DIR}"
echo " > Using ${DEPENDENCIES_DIR}' to store dependencies."
mkdir -p "${DEPENDENCIES_DIR}"
install_essential_tools "${DEPENDENCIES_DIR}"

echo " > Installing Requirements.txt..."
pip install --upgrade pip
pip install nvidia-pyindex || {
	echo "Could not install nvidia-pyindex, stopping install"
	exit 1
}
# # One of the hidden dependencies require Cython, but doesn't specify it.
# # https://github.com/VKCOM/YouTokenToMe/pull/108
# # WAR by installing Cython before requirements.
pip install "Cython==0.29.36" || {
	echo "Could not install Cython, stopping install"
	exit 1
}
# PyYaml, Cython and pip don't play well together.
# https://github.com/yaml/pyyaml/issues/601
pip install "pyyaml==5.4.1" --no-build-isolation || {
	echo "Could not install PyYaml, stopping install"
	exit 1
}
pip install -r requirements.txt || {
	echo "Could not install dependencies, stopping install"
	exit 1
}

# Installation from source
if [ "$BUILD_SRCLIBS" -eq "1" ]; then
	(command -v -- "ninja" >/dev/null 2>&1) || [ "$BUILD_NINJA" -eq "0" ] && echo " > Could not locate ninja, consider passing the --ninja flag to speedup dependency installation."
fi

cd "${DEPENDENCIES_DIR}" || exit
if (! command -v -- "ninja" >/dev/null 2>&1) && [ "$BUILD_NINJA" -eq "1" ]; then
	echo " > Building ninja..."
	install_ninja
fi

if [ "$BUILD_SRCLIBS" -eq "1" ]; then
	echo " > Installing Apex..."
	install_apex
fi

echo " > Installing Megatron-LM..."
install_megatron

if [ "$BUILD_SRCLIBS" -eq "1" ]; then
	echo " > Installing flash-attention..."
	install_flash_attention
fi

if [ "$BUILD_SRCLIBS" -eq "1" ]; then
	echo " > Installing TransformerEngine..."
	install_transformer_engine
fi

echo " > Installing NeMo..."
install_nemo

if [ ! -f "${NEMO_DIR}/GPT3/convert_te_onnx_to_trt_onnx.py" ]; then
	echo " > Copying opset19 conversion script..."
	if [ ! -f "${SCRIPT_DIR}/convert_te_onnx_to_trt_onnx.py" ]; then
		echo "Opset19 conversion script is not located at <ROOT_DIR>/scripts/convert_te_onnx_to_trt_onnx.py"
		return 1
	fi
	cp "${SCRIPT_DIR}/convert_te_onnx_to_trt_onnx.py" "${NEMO_DIR}/GPT3/convert_te_onnx_to_trt_onnx.py"
fi

cd ../

unset ARG_HELP
unset ARG_JOBS
unset BUILD_NINJA
unset DEPENDENCIES_DIR
unset SCRIPT_DIR
unset DEMO_DIR
unset NEMO_DIR
