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

import tensorflow as tf
import numpy as np
import os
import sys
import argparse
from copy import deepcopy

"""
    The conversion of a checkpoint from 
    https://github.com/tensorflow/nmt project 
    The conversion was tested using Tensorflow 1.6
"""

def chpt_to_dict_arrays_simple(file_name):
    """
        Convert a checkpoint into into a dictionary of numpy arrays 
        for later use in TensorRT NMT sample.
    """
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    saver = tf.train.import_meta_graph(file_name)
    dir_name = os.path.dirname(os.path.abspath(file_name))
    saver.restore(sess, tf.train.latest_checkpoint(dir_name))

    params = {}
    print ('\nFound the following trainable variables:')
    with sess.as_default():
        variables = tf.trainable_variables()
        for v in variables:
            params[v.name] = v.eval(session=sess)
            print ("{0}    {1}".format(v.name, params[v.name].shape))

    #use default value
    params["forget_bias"] = 1.0
    return params

def chpt_to_dict_arrays():
    """
        Convert a checkpoint into a dictionary of numpy arrays 
        for later use in TensorRT NMT sample.
        git clone https://github.com/tensorflow/nmt.git
    """
    sys.path.append('./nmt')
    from nmt.nmt import add_arguments, create_hparams
    from nmt import attention_model
    from nmt import model_helper
    from nmt.nmt import create_or_load_hparams
    from nmt import utils
    from nmt import model as nmt_model

    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()

    default_hparams = create_hparams(FLAGS)

    hparams = create_or_load_hparams(\
        FLAGS.out_dir, default_hparams, FLAGS.hparams_path, save_hparams=False)

    print (hparams)

    model_creator = None
    if not hparams.attention:
        model_creator = nmt_model.Model
    elif hparams.attention_architecture == "standard":
        model_creator = attention_model.AttentionModel
    else:
        raise ValueError("Unknown model architecture")

    infer_model = model_helper.create_infer_model(model_creator, hparams, scope = None)

    params = {}
    print ('\nFound the following trainable variables:')
    with tf.Session(
        graph=infer_model.graph, config=utils.misc_utils.get_config_proto()) as sess:

        loaded_infer_model = model_helper.load_model(
        infer_model.model, FLAGS.ckpt, sess, "infer")

        variables = tf.trainable_variables()
        for v in variables:
            params[v.name] = v.eval(session=sess)
            print ("{0}    {1}".format(v.name, params[v.name].shape))

    params["forget_bias"] = hparams.forget_bias
    return params

def concatenate_layers(params):
    """Concatenate weights from multiple layers"""

    input_dict_size = params[u'embeddings/encoder/embedding_encoder:0'].shape[0]
    output_dict_size = params[u'embeddings/decoder/embedding_decoder:0'].shape[0]
    print('Input dictionary size: {0}, Output dictionary size: {1}'.format(input_dict_size, output_dict_size))

    layers = 0
    encoder_type = "unidirectional"
    for key in params:
        if "bidirectional_rnn" in key:
            encoder_type = "bidirectional"
        if "basic_lstm_cell" in key:
            layers = layers + 1

    # we have encoder, decoder, kernel and bias
    layers = int(layers / 4)
    print('Layers: {0}, Encoder type: {1}'.format(layers, encoder_type))

    data = {}
    encoder_postfix = u'/basic_lstm_cell/'
    kernel_alias = u'kernel:0'
    bias_alias = u'bias:0'
    # weights, concatenate all layers
    #process encoder
    if encoder_type == 'bidirectional':
        bi_layers = int(layers / 2)
        if bi_layers == 1:
            bifw_encoder_prefix = u'dynamic_seq2seq/encoder/bidirectional_rnn/fw/basic_lstm_cell/'
            bibw_encoder_prefix = u'dynamic_seq2seq/encoder/bidirectional_rnn/bw/basic_lstm_cell/'
            data["encrnnkernel"] = params[bifw_encoder_prefix + kernel_alias] 
            tmp_weights = params[bibw_encoder_prefix + kernel_alias]
            data["encrnnkernel"] = np.concatenate((data["encrnnkernel"], tmp_weights), axis=0)

            data["encrnnbias"] = params[bifw_encoder_prefix + bias_alias]
            tmp_weights = params[bibw_encoder_prefix + bias_alias]
            data["encrnnbias"] = np.concatenate((data["encrnnbias"], tmp_weights), axis=0)
        else:
            bifw_encoder_prefix = u'dynamic_seq2seq/encoder/bidirectional_rnn/fw/multi_rnn_cell/cell_'
            bibw_encoder_prefix = u'dynamic_seq2seq/encoder/bidirectional_rnn/bw/multi_rnn_cell/cell_'

            data["encrnnkernel"] = np.concatenate(tuple(params[bifw_encoder_prefix + str(i) \
                                                    + encoder_postfix + kernel_alias] \
                                                    for i in range(bi_layers)), axis=0)
            tmp_weights = np.concatenate(tuple(params[bibw_encoder_prefix + str(i) \
                                                + encoder_postfix + kernel_alias] \
                                                for i in range(bi_layers)), axis=0)
            data["encrnnkernel"] = np.concatenate((data["encrnnkernel"], tmp_weights), axis=0)

            data["encrnnbias"] = np.concatenate(tuple(params[bifw_encoder_prefix + str(i) \
                                                + encoder_postfix + bias_alias] \
                                                for i in range(bi_layers)), axis=0)
            tmp_weights = np.concatenate(tuple(params[bibw_encoder_prefix + str(i) \
                                        + encoder_postfix + bias_alias] \
                                        for i in range(bi_layers)), axis=0)
            data["encrnnbias"] = np.concatenate((data["encrnnbias"], tmp_weights), axis=0)
    else:
        uni_encoder_prefix = u'dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_'
        data["encrnnkernel"] = np.concatenate(tuple(params[uni_encoder_prefix + str(i) \
                                                + encoder_postfix + kernel_alias] \
                                                for i in range(layers)), axis=0)
        data["encrnnbias"] = np.concatenate(tuple(params[uni_encoder_prefix + str(i) \
                                            + encoder_postfix + bias_alias] \
                                            for i in range(layers)), axis=0)

    data["encembed"] = params[u'embeddings/encoder/embedding_encoder:0']

    #process decoder
    data["decembed"] = params[u'embeddings/decoder/embedding_decoder:0']
    data["decmemkernel"] = params[u'dynamic_seq2seq/decoder/memory_layer/kernel:0']
    data["decattkernel"] = params[u'dynamic_seq2seq/decoder/attention/attention_layer/kernel:0']
    data["decprojkernel"] = params[u'dynamic_seq2seq/decoder/output_projection/kernel:0']

    uni_decoder_prefix = u'dynamic_seq2seq/decoder/attention/multi_rnn_cell/cell_'
    data["decrnnkernel"] = np.concatenate(tuple(params[uni_decoder_prefix + str(i) \
                                            + encoder_postfix + kernel_alias] \
                                            for i in range(layers)), axis=0)
    data["decrnnbias"] = np.concatenate(tuple(params[uni_decoder_prefix + str(i) \
                                        + encoder_postfix + bias_alias] \
                                        for i in range(layers)), axis=0)

    for key in data:
        print("{0} shape: {1}".format(key, data[key].shape))

    num_units = int(data["decrnnkernel"].shape[1] / 4)
    encoder_type_int = 1 if encoder_type == 'bidirectional' else 0
    dimensions = {"layers": layers, 
                  "encoder_type": encoder_type_int, 
                  "num_units":  num_units,
                  "encembed_outputs": data['encembed'].shape[0],
                  "decembed_outputs": data['decembed'].shape[0],
                  }
    return dimensions, data

def convert_rnn_kernel(weights, dimensions, is_decoder_rnn = False):
    """ 
    In place. weights conversion
    TensorFlow weight parameters for BasicLSTMCell
    are formatted as:
    Each [WR][icfo] is hiddenSize sequential elements.
    CellN  Row 0: WiT, WcT, WfT, WoT
    CellN  Row 1: WiT, WcT, WfT, WoT
    ...
    CellN RowM-1: WiT, WcT, WfT, WoT
    CellN RowM+0: RiT, RcT, RfT, RoT
    CellN RowM+1: RiT, RcT, RfT, RoT
    ...
    CellNRow(M+P)-1: RiT, RcT, RfT, RoT
    M - data size
    P - projection size
    TensorRT expects the format to laid out in memory:
    CellN: Wf, Wi, Wc, Wo, Rf, Ri, Rc, Ro

    For the purpose of implementing LSTMP all W and R weights become weights from W
    CellN: Wf, Rf, Wi, Ri, Wc, Rc, Wo, Ro, Empty states

    Update: alternative notation
    Tensorflow documents gates' order in e.g. 
    https:github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py:439
    TF: i = input_gate, j = new_input (cell gate), f = forget_gate, o = output_gate - ijfo
    Need to convert 'ijfo' to 'fijo'
    """

    print("Starting shape: {0}".format(weights.shape))

    num_units = dimensions["num_units"]
    layers = dimensions["layers"]


    new_weights = np.empty([0], dtype=np.float32)
    # if is_decoder_rnn == False:
    if False :
        # we can use decoder path for both, but we leave it for now
        input_size = num_units
        # case encoder
        # (layers * 2 * input_size, 4 * num_units) -> (layers, 2, input_size, 4, num_units))
        weights = np.reshape(weights, (layers, 2, input_size, 4, num_units))
        print("After reshape: {0}".format(weights.shape))

        # reorder/transpose axis to match TensorRT format (layers, 2, 4, num_units, input_size) 
        weights = np.moveaxis(weights, [2, 3, 4], [4, 2, 3])
        print("After moveaxis: {0}".format(weights.shape))

        # then we reorder gates from Tensorflow's 'icfo' into TensorRT's 'fico' order
        input_perm = [ 1, 2, 0, 3 ]
        temp_weights = np.empty([layers, 2, 4, num_units, input_size], dtype=np.float32)
        for i in range(4):
            temp_weights[:, :, input_perm[i], :, :] = weights[:, :, i, :, :]

        weights = deepcopy(temp_weights)
    else:
        offset = 0
        for i in range(layers):
            # first layer has shape (input_size + num_units, 4 * num_units)
            # other layers  (num_units + num_units, 4 * num_units)
            input_size = 2 * num_units if i == 0 and is_decoder_rnn else num_units
            temp_weights_w = np.empty([4, num_units, input_size], dtype=np.float32)
            temp_weights_r = np.empty([4, num_units, num_units], dtype=np.float32)

            layer_weights_w = np.reshape(weights[offset:(offset + input_size), :], (input_size, 4, num_units))
            layer_weights_r = np.reshape(weights[(offset + input_size):(offset + input_size + num_units), :], (num_units, 4, num_units))

            # reorder/transpose axis to match TensorRT format (layers, 2, 4, num_units, input_size)
            layer_weights_w = np.moveaxis(layer_weights_w, [0, 1, 2], [2, 0, 1])
            layer_weights_r = np.moveaxis(layer_weights_r, [0, 1, 2], [2, 0, 1])

            # then we reorder gates from Tensorflow's 'icfo' into TensorRT's 'fico' order
            input_perm = [ 1, 2, 0, 3 ]
            for i in range(4):
                temp_weights_w[input_perm[i], :, :] = layer_weights_w[i, :, :]
                temp_weights_r[input_perm[i], :, :] = layer_weights_r[i, :, :]

            layer_weights_w = deepcopy(temp_weights_w.flatten())
            layer_weights_r = deepcopy(temp_weights_r.flatten())
            new_weights = np.concatenate((new_weights, layer_weights_w, layer_weights_r), axis = 0)

            offset = offset + input_size + num_units

    return new_weights

def convert_rnn_bias(weights, dimensions, forget_bias = 1.0):
    """
    TensorFlow bias parameters for BasicLSTMCell
    are formatted as:
    CellN: Bi, Bc, Bf, Bo

    TensorRT expects the format to be:
    CellN: Wf, Wi, Wc, Wo, Rf, Ri, Rc, Ro

    Since Tensorflow already combines U and W,
    we double the size and set all of U to zero.
    """
    num_units = dimensions["num_units"]
    layers = dimensions["layers"]
    temp_weights = np.zeros([layers, 2 * 4, num_units], dtype=np.float32)
    weights = np.reshape(weights, (layers, 4, num_units))

    # then we reorder gates from Tensorflow's 'icfo' into TensorRT's 'fico' order
    input_perm = [ 1, 2, 0, 3 ]
    for i in range(4):
        temp_weights[:, input_perm[i], :] = weights[:, i, :]
    #  Add a value to f bias to be consistent with the Tensorflow model.
    print("Adding {0} to forget bias".format(forget_bias))
    temp_weights[:, 0, :] = np.add(temp_weights[:, 0, :], forget_bias)
    weights = deepcopy(temp_weights)

    return weights


def convert_weigts(dimensions, data, forget_bias = 1.0):
    """Convert weights from Tensorflow to TensorRT format"""
  
    print("Processing encoder RNN kernel") 
    data["encrnnkernel"] = convert_rnn_kernel(data["encrnnkernel"], dimensions, False)
    
    print("Processing encoder RNN bias")
    data["encrnnbias"] = convert_rnn_bias(data["encrnnbias"], dimensions, forget_bias = forget_bias)

    print("Processing decoder RNN kernel") 
    data["decrnnkernel"] = convert_rnn_kernel(data["decrnnkernel"], dimensions, True)

    print("Processing decoder RNN bias")
    data["decrnnbias"] = convert_rnn_bias(data["decrnnbias"], dimensions, forget_bias = forget_bias)
    
    return data

def save_layer_weights(data, list_keys, dims, footer_string, file_name):
    """
        data          - dictionary with string names as keys and 
                        numpy weights as values
        list_keys     - list of dictionary keys to save
        dims          - list of int values relevant to the layer
                        e.g. tensor dimensions sufficient to extract all the tensors
        footer_string - marker placed at the end of file

        file format: data -> meta_data -> footer
    """

    data_type = data[list_keys[0]].dtype
    #default precision is FP32
    # The values should be compartible with DataType from Nvinfer.h
    data_prec = 1 if data_type == np.dtype('float16') else 0

    meta_data  = np.int32([data_prec] + dims)
    meta_count = np.int32(meta_data.shape[0])

    out_file = open(file_name, 'wb')
    for key in list_keys:
        out_file.write(data[key].tobytes())
    out_file.write(meta_data.tobytes())
    # write footer
    out_file.write(meta_count.tobytes() + bytearray(footer_string, 'ASCII'))

def main(_):

    if len(sys.argv) < 3:
        print ('\nUsage:')
        print ('python {0} <NMT inference parameters> --weightsdir=<case_name_dir>'.format(sys.argv[0]))
        print ("""e.g. \npython {0} --src=en --tgt=vi \\
    --ckpt=/path/to/envi_model/translate.ckpt \\
    --hparams_path=nmt/standard_hparams/iwslt15.json \\ 
    --out_dir=/tmp/envi \\
    --vocab_prefix=/tmp/nmt_data/vocab \\
    --inference_input_file=/tmp/nmt_data/tst2013.en \\
    --inference_output_file=/tmp/envi/output_infer \\
    --inference_ref_file=/tmp/nmt_data/tst2013.vi \\
    --weightsdir=envi""".format(sys.argv[0]))
        print ('\nOR\n')
        print ('python {0} --metafile=</path_to/graph.meta> --weightsdir=<case_name_dir> '.format(sys.argv[0]))
        print ('e.g.\npython {0} --metafile=./translate.ckpt-12000.meta --weightsdir=envi'.format(sys.argv[0]))
        sys.exit()

    nmt_parser = argparse.ArgumentParser()
    nmt_parser.add_argument("--metafile", type=str, default=None,
                      help="Path to the metafile (alternative checkpoint restore, may not work)")
    nmt_parser.add_argument("--weightsdir", type=str, default="weights",
                     help="Output weights directory")
    trt_flags, unparsed = nmt_parser.parse_known_args()

    if trt_flags.metafile == None:
        params = chpt_to_dict_arrays()
    else:
        params = chpt_to_dict_arrays(trt_flags.metafile)

    print('\nLoading the checkpoint...\n')
    
    print('\nConcatenating the weights...')
    dimensions, data = concatenate_layers(params)

    print('\nConverting the weights...')
    # Convert weights to TensorRT format
    data = convert_weigts(dimensions, data, params["forget_bias"])

    print('\nSaving into binary file...')

    case_dir = trt_flags.weightsdir
    if not os.path.isdir(case_dir):
        os.mkdir(case_dir)
    case_dir = case_dir + "/"

    trt_string = u'trtsamplenmt'
    # save embed weights
    save_layer_weights(data, ["encembed"], \
                        [ dimensions["encembed_outputs"], \
                          dimensions["num_units"] ], \
                        trt_string, case_dir + "encembed.bin")
    save_layer_weights(data, ["decembed"], \
                        [ dimensions["decembed_outputs"], \
                        dimensions["num_units"] ], \
                        trt_string, case_dir + "decembed.bin")
    #encrnn
    save_layer_weights(data, ["encrnnkernel", "encrnnbias"], \
                        [ dimensions["encoder_type"], \
                        dimensions["layers"], \
                        dimensions["num_units"] ], \
                        trt_string, case_dir + "encrnn.bin")
    #decrnn
    save_layer_weights(data, ["decrnnkernel", "decrnnbias"], \
                        [ 0, \
                        dimensions["layers"], \
                        dimensions["num_units"] ], \
                        trt_string, case_dir + "decrnn.bin")
    #decprojkernel
    save_layer_weights(data, ["decprojkernel"], \
                        [ dimensions["num_units"], \
                        dimensions["decembed_outputs"] ], \
                        trt_string, case_dir + "decproj.bin")

    #decmemkernel
    save_layer_weights(data, ["decmemkernel"], \
                        [ dimensions["num_units"], \
                        dimensions["num_units"] ], \
                        trt_string, case_dir + "decmem.bin")
                        
    #decattkernel
    # first dimension is 3 * num_units of bi RNN, 2 * num_units otherwise
    save_layer_weights(data, ["decattkernel"], \
                        [ data["decattkernel"].shape[0], \
                        dimensions["num_units"] ], \
                        trt_string, case_dir + "decatt.bin")


if __name__ == "__main__":
    tf.app.run()
