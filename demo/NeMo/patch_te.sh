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
if [ ! "${0##*/}" = "patch_te.sh" ]; then
	echo "Please run this patch script, don't source it." >&2
	return 1
fi

NEMO_DIR=$(dirname "$(readlink -f "$0")")

te_loc="$(pip show transformer_engine | grep '^Location' | awk '{print $2}')"
cd "${te_loc}/transformer_engine" || {
	echo "Could not locate transformer-engine python package. Please check if installation proceeded correctly."
	exit 1
}
# Use sys.executable when calling pip within subprocess to recognize virtualenv.
# If patch is already applied, skip it and proceed with the rest of the script, quit otherwise.
# NOTE: patch needs to be updated to track the commit of TE in install.sh.
OUT="$(patch --forward common/__init__.py <"${NEMO_DIR}"/transformer_engine.patch)" || echo "${OUT}" | grep "Skipping patch" -q || {
	echo "Could not patch transformer engine because ${OUT}"
	exit 1
}
unset OUT
cd - || exit
unset te_loc
