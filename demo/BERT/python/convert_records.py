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

# The reference bert implementation preprocesses the input e.g. a squad dataset
# generating a tf record dataset of token ids, input masks and segment ids.
# Given such a dataset (e.g. eval.tf_record), this script converts the tf records
# into a simpler binary format readable by the bert sample.

import sys
import struct
import argparse
import re

try:
    import tensorflow as tf
    tf.enable_eager_execution()
except ImportError as err:
    sys.stderr.write("""Error: Failed to import tensorflow module ({})""".format(err))
    sys.exit()

parser = argparse.ArgumentParser(description='Convert a BERT TF record dataset to TRT weights format')

parser.add_argument('-i', '--input', required=True, help='The path to the TF record dataset, e.g. eval.tf_record')
parser.add_argument('-o', '--output', required=True, help='The weight file to dump the data  to.')
parser.add_argument('-s', '--seqlen', required=True, help='The sequence length used to generate the tf record dataset')

opt = parser.parse_args()

print("Outputting the trained weights in TensorRT's in the following format:")
print("Line 0: <number of buffers N in the file>")
print("Line 1-N: [buffer name] [buffer type] [number of dims D] [space-sep. shape list: d_1 d_2 ... d_D] <d_1 * d_2 * ... * d_D * sizeof(type) bytes of data>")
print("The buffers are all 1-D int32 arrays (buffer type 3) of the same length S, specified by the seqlen input parameter")
print("The file contains rows that logically belong together in groups of three. They will be named input_ids_<idx>, input_mask_<idx>, segment_ids_<idx>, i.e. tied together by the same idx number.")

inputbase = opt.input
outputbase = opt.output

seq_length=opt.seqlen
name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }     

def _decode_record(record, name_to_features):                                                                                                                                          
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but we expect tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example

raw_dataset = tf.data.TFRecordDataset([inputbase])
out_fn = outputbase + ".weights_int32"
with open(out_fn, 'wb') as output_file:
    
    count = raw_dataset.reduce(0, lambda x,y: x+ 1).numpy()
    print(count)

    output_file.write("{}\n".format(count).encode('ASCII'))
    
    for idx, record in enumerate(raw_dataset):
        dec = _decode_record(record, name_to_features)
        
        for k,v in dec.items():
            a = v.numpy()
            outname = '{}_{}'.format(k, idx)

            shape = a.shape
            shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])

            output_file.write("{} 3 {} ".format(outname, shape_str).encode('ASCII'))
            output_file.write(a.tobytes())
            output_file.write("\n".encode('ASCII'));
        
