#!/usr/bin/env python3

# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import struct
import argparse
import re

try:
    from tensorflow.python import pywrap_tensorflow as pyTF
except ImportError as err:
    sys.stderr.write("""Error: Failed to import tensorflow module ({})""".format(err))
    sys.exit()

parser = argparse.ArgumentParser(description='TensorFlow to TensorRT Weight Dumper')

parser.add_argument('-m', '--model', required=True, help='The checkpoint file basename, example basename(model.ckpt-766908.data-00000-of-00001) -> model.ckpt-766908')
parser.add_argument('-o', '--output', required=True, help='The weight file to dump all the weights to.')

opt = parser.parse_args()

print( "Outputting the trained weights in TensorRT's in the following format:")
print( "Line 0: <number of buffers N in the file>")
print( "Line 1-N: [buffer name] [buffer type] [number of dims D] [space-sep. shape list: d_1 d_2 ... d_D] <d_1 * d_2 * ... * d_D * sizeof(type) bytes of data>")
print( "Buffer type is hard-coded to 0 (float), i.e. 4 bytes per coefficient")

inputbase = opt.model
outputbase = opt.output

reader = pyTF.NewCheckpointReader(inputbase)
tensor_dict = reader.get_variable_to_shape_map()
out_fn = outputbase + ".weights"
with open(out_fn, 'wb') as output_file:

    # there might be training-related variables in the checkpoint that can be discarded
    exclude_list = ["adam", "global_step", "pooler", "bad_steps", "good_steps", "loss_scale"]
    param_names = [key for key in sorted(tensor_dict) if all([exclude not in key for exclude in exclude_list])]

    count = len(param_names)
    print(count)

    output_file.write('{}\n'.format(count).encode('ASCII'))
    for pn in param_names:
        toks = pn.lower().split('/')
        if 'encoder' in pn:
            assert('layer' in pn)
            l = (re.findall('\d+',pn))[0]
            outname = 'l{}_'.format(l) + '_'.join(toks[3:])
        else:
            outname = '_'.join(toks)

        tensor = reader.get_tensor(pn)
        shape = tensor.shape
        flat_tensor = tensor.flatten()
        shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])

        output_file.write('{} 0 {} '.format(outname, shape_str).encode('ASCII'))
        output_file.write(flat_tensor.tobytes())
        output_file.write('\n'.encode('ASCII'));
        print('Orig.name:', pn,'TRT name:', outname, 'shape:' , shape_str)
