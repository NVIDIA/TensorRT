#!/usr/bin/python
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

# Script to convert from TensorFlow weights to TensorRT weights for multilayer perceptron sample.
# Change the remap to properly remap the weights to the name from your trained model
# to the sample expected format.

import sys
import struct
import argparse

try:
    from tensorflow.python import pywrap_tensorflow as pyTF
except ImportError as err:
    sys.stderr.write("""Error: Failed to import module ({})""".format(err))
    sys.exit()

parser = argparse.ArgumentParser(description='TensorFlow to TensorRT Weight Dumper')

parser.add_argument('-m', '--model', required=True, help='The checkpoint file basename, example basename(model.ckpt-766908.data-00000-of-00001) -> model.ckpt-766908')
parser.add_argument('-o', '--output', required=True, help='The weight file to dump all the weights to.')

opt = parser.parse_args()

print "Outputting the trained weights in TensorRT's wts v2 format. This format is documented as:"
print "Line 0: <number of buffers in the file>"
print "Line 1-Num: [buffer name] [buffer type] [(buffer shape{e.g. (1, 2, 3)}] <buffer shaped size bytes of data>"

inputbase = opt.model
outputbase = opt.output

# This dictionary translates from the TF weight names to the weight names expected 
# by the sampleMLP sample. This is the location that needs to be changed if training
# something other than what is specified in README.txt.
remap = {
    'Variable': 'hiddenWeights0',
    'Variable_1': 'hiddenWeights1',
    'Variable_2': 'outputWeights',
    'Variable_3': 'hiddenBias0',
    'Variable_4': 'hiddenBias1',
    'Variable_5': 'outputBias'
}

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

try:
   reader = pyTF.NewCheckpointReader(inputbase)
   tensorDict = reader.get_variable_to_shape_map()
   outputFileName = outputbase + ".wts2"
   outputFile = open(outputFileName, 'w')
   count = 0

   for key in sorted(tensorDict):
       # Don't count weights that aren't used for inferencing.
       if ("Adam" in key or "power" in key):
           continue
       count += 1
   outputFile.write("%s\n"%(count))

   for key in sorted(tensorDict):
       # In order to save space, we don't dump weights that aren't required.
       if ("Adam" in key or "power" in key):
           continue
       tensor = reader.get_tensor(key)
       file_key = remap[key.replace('/','_')]
       val = tensor.shape
       print("%s 0 %s "%(file_key, val))
       flat_tensor = tensor.flatten()
       outputFile.write("%s 0 %s "%(file_key, val))
       outputFile.write(flat_tensor.tobytes())
       outputFile.write("\n");
   outputFile.close()

except Exception as error:
    print(str(error))
