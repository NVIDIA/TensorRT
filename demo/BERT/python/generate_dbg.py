# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

import argparse
import os
from os.path import join as jp
import numpy as np
import re
import site

os.system("git clone https://github.com/NVIDIA/DeepLearningExamples.git dle 2> /dev/null || git -C dle pull")
try:
    import tensorflow as tf
except ImportError as err:
    sys.stderr.write("""Error: Failed to import tensorflow module ({})""".format(err))
    sys.exit()

parser = argparse.ArgumentParser(description='Generate debug input/output pairs')

TEST_INPUT_FN = 'test_inputs.weights_int32'
TEST_OUTPUT_FN = 'test_outputs.weights'

parser.add_argument('-o', '--output', required=True, help='The directory to dump the data to. The script will create an input file, {}, and a correponding reference output file, {}.'.format(TEST_INPUT_FN, TEST_OUTPUT_FN))
parser.add_argument('-s', '--seqlen', type=int, required=True, help='The sequence length used to generate the tf record dataset')
parser.add_argument('-b', '--batchsize',type=int, required=True, help='The sequence length used to generate the tf record dataset')
parser.add_argument('-f', '--finetuned', required=True, help='The checkpoint file basename of the fine-tuned model, example basename(model.ckpt-766908.data-00000-of-00001) -> model.ckpt-766908')
parser.add_argument('-p', '--pretrained', required=True, help='The folder containing the bert_config.json, which can be downloaded e.g. from https://github.com/google-research/bert#pre-trained-models or by running download_models.py in dle/TensorFlow/LanguageModeling/BERT/data/pretrained_models_google')
parser.add_argument('-r', '--randomseed',type=int, required=False, default=12345, help='Seed for PRNG')

opt = parser.parse_args()

slen = opt.seqlen
B = opt.batchsize
outputbase = opt.output
bert_path = opt.pretrained
init_checkpoint = opt.finetuned

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

site.addsitedir('dle/TensorFlow/LanguageModeling/BERT')
import run_squad
import modeling
del_all_flags(tf.flags.FLAGS)
tf.flags.FLAGS(['test'])

np.random.seed(opt.randomseed)

bert_config_path = jp(bert_path, 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_path)

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.visible_device_list = ''

with tf.Graph().as_default() as g:

    input_ids = tf.placeholder(tf.int32, name='input_ids', shape=(None, slen))
    input_mask = tf.placeholder(tf.int32, name='input_mask', shape=(None, slen))
    segment_ids = tf.placeholder(tf.int32, name='segment_ids', shape=(None, slen))

    use_one_hot_embeddings=False
    is_training=False

    ###### START run_squad.create_model to get access to bert model
    model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            compute_type=tf.float32)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
            "cls/squad/output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
            "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
            [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)
    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1]) 
    ###### END run_squad.create_model

    tvars = tf.trainable_variables()

    if init_checkpoint:
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    init_op = tf.global_variables_initializer()

sess = tf.Session(graph=g, config=config)
sess.run(init_op)


test_word_ids = np.random.randint(0, bert_config.vocab_size, (B, slen), dtype=np.int32)
test_input_mask = np.ones((B,slen), dtype=np.int32)
test_segment_ids = np.random.randint(0, bert_config.type_vocab_size, (B, slen), dtype=np.int32)

fd = {'input_ids:0' : test_word_ids,
        'input_mask:0':test_input_mask,
        'segment_ids:0':test_segment_ids}

out_emb = sess.run(model.embedding_output, feed_dict=fd)
out_enc = sess.run(model.all_encoder_layers, feed_dict=fd)
out_logits = sess.run(logits, feed_dict=fd)

if not os.path.exists(outputbase):
    print("Output path does not exist. Creating.")
    os.makedirs(outputbase)

outputFileName = jp(outputbase,  TEST_INPUT_FN)

with open(outputFileName, 'wb') as outputFile:
    count = 3
    outputFile.write("{}\n".format(count).encode('ASCII'))
    idx = 0
    for k,v in fd.items():
        outname = '{}_{}'.format(k[:-2], idx)
        shape = v.shape
        shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])
        outputFile.write("{} 3 {} ".format(outname, shape_str).encode('ASCII'))
        outputFile.write(v.tobytes())
        outputFile.write("\n".encode('ASCII'))

outputFileName = jp(outputbase,  TEST_OUTPUT_FN)

with open(outputFileName, 'wb') as outputFile:
    count = 2 + len(out_enc)
    outputFile.write("{}\n".format(count).encode('ASCII'))
    outname = 'embedding_output'
    shape = out_emb.shape
    shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])
    outputFile.write("{} 0 {} ".format(outname, shape_str).encode('ASCII'))
    outputFile.write(out_emb.tobytes())
    outputFile.write("\n".encode('ASCII'))

    outname = 'logits'
    shape = out_logits.shape
    shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])
    outputFile.write("{} 0 {} ".format(outname, shape_str).encode('ASCII'))
    outputFile.write(out_logits.tobytes())
    outputFile.write("\n".encode('ASCII'))

    for it, enc in enumerate(out_enc):
        outname = 'l{}_output'.format(it)
        shape = enc.shape
        shape_str = '{} '.format(len(shape)) + ' '.join([str(d) for d in shape])
        outputFile.write("{} 0 {} ".format(outname, shape_str).encode('ASCII'))
        outputFile.write(enc.tobytes())
        outputFile.write("\n".encode('ASCII'))

