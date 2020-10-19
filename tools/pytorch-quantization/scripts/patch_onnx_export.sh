#!/bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
# Patching script for supporting fake quantization in torch ONNX export 
# See docs/source/userguide.rst for details

PatchFile() {
   TARGET_FILE=$1
   PATCH_FILE=$2
   if [ $(diff --unchanged-line-format= --old-line-format= --new-line-format='%L' "$TARGET_FILE" "$PATCH_FILE" | wc -l) -eq "0" ]; then
      echo "$TARGET_FILE is already patched with $PATCH_FILE"
   else
      printf "\n#Fake quantization export\n" >> $TARGET_FILE
      cat $PATCH_FILE >> $TARGET_FILE
      echo "$TARGET_FILE was patched."
   fi
}

# Check if the patch file exists
CWD="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PATCH_C="${CWD}/onnx_export_per_channel.patch"
if [ -f $PATCH_C ]; then
   echo "Found patch: $PATCH_C"
else
   echo "$PATCH_C does not exist."
   exit 1
fi
PATCH_T="${CWD}/onnx_export_per_tensor.patch"
if [ -f $PATCH_T ]; then
   echo "Found patch: $PATCH_T"
else
   echo "$PATCH_T does not exist."
   exit 1
fi

# Check if the ONNX opset file exists
DIR=$(pip show torch | grep Location | cut -d' ' -f 2)
FILE13="${DIR}/torch/onnx/symbolic_opset13.py"
FILE10="${DIR}/torch/onnx/symbolic_opset10.py"

# Patch per tensor
if [ -f $FILE10 ]; then
   # Patch per tensor in opset 10
   PatchFile "$FILE10" "$PATCH_T"
else
   echo "File ${FILE10} does not exist."
   exit 1
fi

# Patch per channel
if [ -f $FILE13 ]; then
   # Patch in opset 13
   PatchFile "$FILE13" "$PATCH_C"
else
   # Patch in opset 10
   PatchFile "$FILE10" "$PATCH_C"
fi

echo "Done."
exit 0
