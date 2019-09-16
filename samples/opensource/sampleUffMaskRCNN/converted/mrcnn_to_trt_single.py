#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from keras.models import model_from_json, Model
from keras import backend as K
from keras.layers import Input, Lambda
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from mrcnn.model import *
import mrcnn.model as modellib
from mrcnn.config import Config
import sys
import os
ROOT_DIR = os.path.abspath("./")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
import argparse
import os
import uff


def parse_command_line_arguments(args=None):
    parser = argparse.ArgumentParser(prog='keras_to_trt', description='Convert trained keras .hdf5 model to trt .uff')

    parser.add_argument(
        '-w',
        '--weights',
        type=str,
        default=None,
        required=True,
        help="The checkpoint weights file of keras model."
    )

    parser.add_argument(
        '-o',
        '--output_file',
        type=str,
        default=None,
        required=True,
        help="The path to output .uff file."
    )

    parser.add_argument(
        '-l',
        '--list-nodes',
        action='store_true',
        help="show list of nodes contained in converted pb"
    )

    parser.add_argument(
        '-p',
        '--preprocessor',
        type=str,
        default=False,
        help="The preprocess function for converting tf node to trt plugin"
    )

    return parser.parse_args(args)


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def main(args=None):

    K.set_image_data_format('channels_first')
    K.set_learning_phase(0)

    args = parse_command_line_arguments(args)

    model_weights_path = args.weights
    output_file_path = args.output_file
    list_nodes = args.list_nodes

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=LOG_DIR, config=config).keras_model

    model.load_weights(model_weights_path, by_name=True)


    model_A = Model(inputs=model.input, outputs=model.get_layer('mrcnn_mask').output)
    model_A.summary()

    output_nodes = ['mrcnn_detection', "mrcnn_mask/Sigmoid"]
    convert_model(model_A, output_file_path, output_nodes, preprocessor=args.preprocessor,
                  text=True, list_nodes=list_nodes)


def convert_model(inference_model, output_path, output_nodes=[], preprocessor=None, text=False,
                  list_nodes=False):
    # convert the keras model to pb
    orig_output_node_names = [node.op.name for node in inference_model.outputs]
    print("The output names of tensorflow graph nodes: {}".format(str(orig_output_node_names)))

    sess = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        orig_output_node_names)

    temp_pb_path = "../temp.pb"
    graph_io.write_graph(constant_graph, os.path.dirname(temp_pb_path), os.path.basename(temp_pb_path),
                         as_text=False)

    predefined_output_nodes = output_nodes
    if predefined_output_nodes != []:
        trt_output_nodes = predefined_output_nodes
    else:
        trt_output_nodes = orig_output_node_names

    # convert .pb to .uff
    uff.from_tensorflow_frozen_model(
        temp_pb_path,
        output_nodes=trt_output_nodes,
        preprocessor=preprocessor,
        text=text,
        list_nodes=list_nodes,
        output_filename=output_path,
        debug_mode = False
    )

    os.remove(temp_pb_path)


if __name__ == "__main__":
    main()
